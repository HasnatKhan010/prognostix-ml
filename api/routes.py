"""HTTP routes for the inference API.

Endpoints are deliberately thin: validate through the Pydantic schemas, delegate
to :class:`~src.inference.predictor.RULPredictor`, translate domain errors into
status codes. All prediction logic lives in ``src/`` so the notebooks, the CLI
scripts and the API cannot disagree about how a RUL is produced.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DriftRequest,
    DriftResponse,
    EngineDetail,
    FleetEngine,
    FleetResponse,
    HealthResponse,
    LeaderboardResponse,
    ModelInfo,
    ModelListResponse,
    PredictRequest,
    PredictResponse,
)
from src import __version__
from src.config import get_config
from src.inference.predictor import ModelRegistry, RULPredictor

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Prometheus instrumentation (optional dependency) ---------------------

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    PREDICTIONS = Counter(
        "prognostix_predictions_total",
        "Predictions served, by model and resulting risk band.",
        ["model", "risk_level"],
    )
    PREDICTION_ERRORS = Counter(
        "prognostix_prediction_errors_total",
        "Prediction requests rejected or failed, by reason.",
        ["reason"],
    )
    PREDICTION_LATENCY = Histogram(
        "prognostix_prediction_duration_seconds",
        "Wall-clock time spent producing a prediction.",
        ["model"],
    )
    LAST_RUL = Gauge(
        "prognostix_last_predicted_rul_cycles",
        "Most recent predicted RUL, by model.",
        ["model"],
    )
    METRICS_ENABLED = True
except ImportError:  # pragma: no cover - metrics are optional
    METRICS_ENABLED = False
    CONTENT_TYPE_LATEST = "text/plain"


def _count_error(reason: str) -> None:
    if METRICS_ENABLED:
        PREDICTION_ERRORS.labels(reason=reason).inc()


# --- dependencies ---------------------------------------------------------


def get_registry(request: Request) -> ModelRegistry:
    """Return the app-wide model registry, creating it if the app skipped startup."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        registry = ModelRegistry()
        request.app.state.registry = registry
    return registry


def _resolve_predictor(registry: ModelRegistry, name: str | None) -> RULPredictor:
    """Load a predictor, mapping load failures onto HTTP statuses."""
    try:
        return registry.get(name)
    except FileNotFoundError as exc:
        _count_error("model_missing")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ValueError as exc:
        _count_error("unknown_model")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


# --- service -------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> HealthResponse:
    """Liveness and readiness.

    Reports ``degraded`` when no trained model is on disk - the process is up but
    cannot serve predictions.
    """
    registry = get_registry(request)
    available = registry.available()
    loaded = sorted(registry._predictors)

    scaler_available = (
        get_config().path("artifacts") / get_config().preprocessing.scaler_filename
    ).exists()

    started_at = getattr(request.app.state, "started_at", None)
    return HealthResponse(
        status="ok" if available else "degraded",
        version=__version__,
        models_available=available,
        models_loaded=loaded,
        default_model=registry.default_model,
        scaler_available=scaler_available,
        uptime_seconds=round(time.time() - started_at, 2) if started_at else 0.0,
    )


@router.get("/models", response_model=ModelListResponse, tags=["service"])
def list_models(request: Request) -> ModelListResponse:
    """Describe every trained model found on disk."""
    registry = get_registry(request)
    entries = [ModelInfo(**info) for info in registry.info()]
    return ModelListResponse(
        default_model=registry.default_model,
        count=len(entries),
        models=entries,
    )


@router.get("/leaderboard", response_model=LeaderboardResponse, tags=["service"])
def leaderboard(metric: str = Query("RMSE", description="Metric to rank by.")) -> LeaderboardResponse:
    """Offline evaluation results recorded during training."""
    from src.evaluation.compare import load_leaderboard

    frame = load_leaderboard(config=get_config())
    if frame.empty:
        return LeaderboardResponse(count=0, metric=metric, rows=[])
    if metric not in frame.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown metric {metric!r}. Available: {sorted(frame.columns)}",
        )

    frame = frame.sort_values(metric).replace({np.nan: None})
    return LeaderboardResponse(
        count=len(frame),
        metric=metric,
        best_model=str(frame.iloc[0]["Model"]),
        rows=frame.to_dict(orient="records"),
    )


@router.get("/metrics", tags=["service"], include_in_schema=False)
def metrics() -> Response:
    """Prometheus exposition endpoint."""
    if not METRICS_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="prometheus_client is not installed",
        )
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- prediction -----------------------------------------------------------


@router.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(
    payload: PredictRequest,
    request: Request,
    explain: bool = Query(
        False, description="Include per-cycle attention weights, when supported."
    ),
) -> PredictResponse:
    """Predict RUL for one machine and return the maintenance decision.

    The window must hold exactly the number of cycles the model was trained on
    (30 by default), oldest cycle first.
    """
    registry = get_registry(request)
    predictor = _resolve_predictor(registry, payload.model)
    started = time.perf_counter()

    try:
        window = (
            np.asarray(payload.window, dtype=float)
            if payload.window is not None
            else predictor.window_from_readings(payload.readings or [])
        )
        assessment = predictor.assess(
            window, scaled=payload.scaled, engine_id=payload.engine_id
        )
        attention = (
            predictor.explain(window, scaled=payload.scaled) if explain else None
        )
    except ValueError as exc:
        _count_error("invalid_input")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        _count_error("not_ready")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    if METRICS_ENABLED:
        PREDICTION_LATENCY.labels(model=predictor.model_name).observe(
            time.perf_counter() - started
        )
        PREDICTIONS.labels(
            model=predictor.model_name, risk_level=assessment.risk_level.value
        ).inc()
        LAST_RUL.labels(model=predictor.model_name).set(assessment.rul)

    return PredictResponse(
        engine_id=payload.engine_id,
        model=predictor.model_name,
        rul_cycles=round(assessment.rul, 2),
        health_score=round(assessment.health_score, 2),
        risk_level=assessment.risk_level.value,
        recommended_action=assessment.recommended_action,
        requires_action=assessment.requires_action,
        window_size=predictor.window_size,
        attention=attention,
    )


@router.post("/predict/batch", response_model=BatchPredictResponse, tags=["prediction"])
def predict_batch(payload: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    """Score a fleet in one call and summarise it by risk band.

    Items are grouped by model so each model's windows run as a single batched
    forward pass. One bad item fails the whole request - partial fleet results
    would be worse than an explicit error.
    """
    registry = get_registry(request)

    groups: dict[str, list[int]] = {}
    for index, item in enumerate(payload.items):
        name = item.model or payload.model or registry.default_model
        groups.setdefault(name, []).append(index)

    responses: list[PredictResponse | None] = [None] * len(payload.items)

    for model_name, indices in groups.items():
        predictor = _resolve_predictor(registry, model_name)
        scaled_flags = {payload.items[index].scaled for index in indices}

        try:
            windows = [
                np.asarray(payload.items[index].window, dtype=float)
                if payload.items[index].window is not None
                else predictor.window_from_readings(payload.items[index].readings or [])
                for index in indices
            ]
        except ValueError as exc:
            _count_error("invalid_input")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        except RuntimeError as exc:
            _count_error("not_ready")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc

        shapes = {window.shape for window in windows}
        # A single stacked forward pass needs uniform shapes and one scaling mode.
        batchable = len(shapes) == 1 and len(scaled_flags) == 1

        started = time.perf_counter()
        try:
            if batchable:
                assessments = predictor.assess_batch(
                    np.stack(windows),
                    scaled=scaled_flags.pop(),
                    engine_ids=[payload.items[index].engine_id for index in indices],
                )
            else:
                assessments = [
                    predictor.assess(
                        window,
                        scaled=payload.items[index].scaled,
                        engine_id=payload.items[index].engine_id,
                    )
                    for window, index in zip(windows, indices, strict=False)
                ]
        except ValueError as exc:
            _count_error("invalid_input")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        except RuntimeError as exc:
            _count_error("not_ready")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc

        if METRICS_ENABLED:
            PREDICTION_LATENCY.labels(model=predictor.model_name).observe(
                time.perf_counter() - started
            )

        for index, assessment in zip(indices, assessments, strict=False):
            if METRICS_ENABLED:
                PREDICTIONS.labels(
                    model=predictor.model_name, risk_level=assessment.risk_level.value
                ).inc()
            responses[index] = PredictResponse(
                engine_id=payload.items[index].engine_id,
                model=predictor.model_name,
                rul_cycles=round(assessment.rul, 2),
                health_score=round(assessment.health_score, 2),
                risk_level=assessment.risk_level.value,
                recommended_action=assessment.recommended_action,
                requires_action=assessment.requires_action,
                window_size=predictor.window_size,
            )

    predictions = [response for response in responses if response is not None]
    risk_summary: dict[str, int] = {}
    for response in predictions:
        risk_summary[response.risk_level] = risk_summary.get(response.risk_level, 0) + 1

    return BatchPredictResponse(
        count=len(predictions),
        predictions=predictions,
        risk_summary=risk_summary,
        action_required=sum(1 for response in predictions if response.requires_action),
    )


# --- fleet ---------------------------------------------------------------


def _fleet_frame(request: Request):
    """The processed test frame, loaded once per process.

    Raises ``FileNotFoundError`` when the data has not been prepared; the
    app-level handler turns that into a 503 with the command to run.
    """
    frame = getattr(request.app.state, "fleet_frame", None)
    if frame is None:
        from src.ingestion.loader import load_processed

        config = get_config()
        frame = load_processed(f"test_{config.data.dataset}_processed", config)
        request.app.state.fleet_frame = frame
    return frame


def _score_fleet(predictor: RULPredictor, frame):
    """Score a whole frame, mapping serving-readiness failures onto a 503."""
    try:
        return predictor.predict_frame(frame, scaled=False)
    except RuntimeError as exc:
        _count_error("not_ready")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ValueError as exc:
        _count_error("invalid_input")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@router.get("/fleet", response_model=FleetResponse, tags=["fleet"])
def fleet(
    request: Request,
    model: str | None = Query(None, description="Model to score with."),
    limit: int = Query(200, ge=1, le=1000, description="Maximum machines returned."),
) -> FleetResponse:
    """Score every machine in the reference dataset, most urgent first.

    This is what the dashboard reads: one prediction per machine from its most
    recent window, plus the fleet-level rollup.
    """
    registry = get_registry(request)
    predictor = _resolve_predictor(registry, model)
    config = get_config()
    frame = _fleet_frame(request)

    scored = _score_fleet(predictor, frame)
    if scored.empty:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No machine has enough cycles to fill the model's window",
        )

    data = config.data
    cycles = frame.groupby(data.id_column)[data.time_column].max()
    truth = (
        frame.sort_values(data.time_column)
        .groupby(data.id_column)[data.target_column]
        .last()
        if data.target_column in frame.columns
        else None
    )

    scored = scored.sort_values("rul").head(limit)
    engines = [
        FleetEngine(
            engine_id=int(row["engine_id"]),
            rul_cycles=float(row["rul"]),
            health_score=float(row["health_score"]),
            risk_level=str(row["risk_level"]),
            recommended_action=str(row["recommended_action"]),
            cycles_observed=int(cycles.get(row["engine_id"], 0)),
            actual_rul=(
                float(truth[row["engine_id"]])
                if truth is not None and row["engine_id"] in truth.index
                else None
            ),
        )
        for _, row in scored.iterrows()
    ]

    risk_summary: dict[str, int] = {}
    for engine in engines:
        risk_summary[engine.risk_level] = risk_summary.get(engine.risk_level, 0) + 1

    labelled = [e for e in engines if e.actual_rul is not None]
    return FleetResponse(
        model=predictor.model_name,
        count=len(engines),
        risk_summary=risk_summary,
        action_required=sum(
            1 for engine in engines if engine.risk_level in ("warning", "critical")
        ),
        fleet_health=round(float(np.mean([e.health_score for e in engines])), 2),
        median_rul=round(float(np.median([e.rul_cycles for e in engines])), 2),
        engines=engines,
        source=f"test_{data.dataset}",
        mae=(
            round(
                float(np.mean([abs(e.rul_cycles - e.actual_rul) for e in labelled])), 3
            )
            if labelled
            else None
        ),
    )


@router.get("/fleet/{engine_id}", response_model=EngineDetail, tags=["fleet"])
def engine_detail(
    engine_id: int,
    request: Request,
    model: str | None = Query(None, description="Model to score with."),
    history: int = Query(60, ge=2, le=500, description="Cycles of history returned."),
) -> EngineDetail:
    """One machine's prediction plus the raw sensor history behind it."""
    registry = get_registry(request)
    predictor = _resolve_predictor(registry, model)
    config = get_config()
    data = config.data
    frame = _fleet_frame(request)

    machine = frame[frame[data.id_column] == engine_id].sort_values(data.time_column)
    if machine.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Engine {engine_id} not found in {data.dataset}",
        )

    columns = predictor.feature_columns
    if columns is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata does not record its feature columns",
        )

    try:
        window = machine[columns].to_numpy(dtype=float)[-predictor.window_size :]
        assessment = predictor.assess(window, scaled=False, engine_id=engine_id)
        explanation = predictor.explain(window, scaled=False)
    except ValueError as exc:
        _count_error("invalid_input")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    recent = machine.tail(history)
    return EngineDetail(
        engine_id=engine_id,
        rul_cycles=round(assessment.rul, 2),
        health_score=round(assessment.health_score, 2),
        risk_level=assessment.risk_level.value,
        recommended_action=assessment.recommended_action,
        cycles_observed=int(machine[data.time_column].max()),
        actual_rul=(
            float(machine[data.target_column].iloc[-1])
            if data.target_column in machine.columns
            else None
        ),
        model=predictor.model_name,
        feature_columns=list(columns),
        sensor_history={
            column: [round(float(value), 4) for value in recent[column]]
            for column in columns
        },
        cycles=[int(value) for value in recent[data.time_column]],
        attention=explanation["attention_weights"] if explanation else None,
    )


# --- monitoring -----------------------------------------------------------


@router.post("/monitoring/drift", response_model=DriftResponse, tags=["monitoring"])
def check_drift(payload: DriftRequest, request: Request) -> DriftResponse:
    """Compare live windows against the training distribution.

    Answers the question accuracy metrics cannot answer without ground truth:
    does the incoming data still look like what the model was trained on?
    """
    from monitoring.drift import detect_drift, reference_distribution

    registry = get_registry(request)
    predictor = _resolve_predictor(registry, payload.model)
    config = get_config()

    try:
        current = np.asarray(payload.windows, dtype=float)
        if not payload.scaled:
            if predictor.scaler is None:
                raise RuntimeError("No scaler available to standardise the windows")
            current = predictor.scaler.transform_sequences(current)

        reference, feature_names = reference_distribution(config)
        report = detect_drift(reference, current, feature_names, config)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return DriftResponse(
        status=report.status,
        n_samples=report.n_samples,
        n_features=report.n_features,
        drifted_features=report.drifted_features,
        feature_share=round(report.feature_share, 4),
        details=[feature.to_dict() for feature in report.features],
    )
