"""Pydantic request and response models for the inference API.

The schemas are the API contract *and* its first line of defence: a window with
the wrong number of cycles, a non-finite sensor value or an unknown model name is
rejected here with a 422 before it ever reaches a loaded model.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "BatchPredictRequest",
    "BatchPredictResponse",
    "DriftRequest",
    "DriftResponse",
    "EngineDetail",
    "ErrorResponse",
    "FleetEngine",
    "FleetResponse",
    "HealthResponse",
    "LeaderboardResponse",
    "ModelInfo",
    "ModelListResponse",
    "PredictRequest",
    "PredictResponse",
]

MAX_CYCLES = 1000
MAX_FEATURES = 64
MAX_BATCH = 256

ModelName = Literal["gru", "lstm", "attention", "random_forest", "linear"]
RiskName = Literal["healthy", "watch", "warning", "critical"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PredictRequest(BaseModel):
    """One prediction request for one machine.

    Supply the sensor history either as a numeric matrix (``window``) or as
    named readings (``readings``) - exactly one of the two.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "engine_id": 42,
                    "model": "gru",
                    "readings": [
                        {"sensor_2": 641.8, "sensor_3": 1589.7, "sensor_4": 1400.6}
                    ],
                }
            ]
        },
    )

    engine_id: int | str | None = Field(
        default=None, description="Identifier echoed back in the response."
    )
    model: ModelName | None = Field(
        default=None,
        description="Model to serve with. Defaults to the configured default model.",
    )
    window: list[list[float]] | None = Field(
        default=None,
        description=(
            "Sensor matrix ordered oldest cycle first, shape "
            "(window_size, n_features). Columns must follow the model's feature order."
        ),
    )
    readings: list[dict[str, float]] | None = Field(
        default=None,
        description=(
            "One dictionary per cycle, oldest first, keyed by sensor name "
            "(e.g. 'sensor_2'). Extra sensors are ignored."
        ),
    )
    scaled: bool = Field(
        default=False,
        description=(
            "True only if the values are already standardised with this project's "
            "scaler. Raw readings must leave this False."
        ),
    )

    @field_validator("window")
    @classmethod
    def _check_window(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("window must contain at least one cycle")
        if len(value) > MAX_CYCLES:
            raise ValueError(f"window may not exceed {MAX_CYCLES} cycles")

        width = len(value[0])
        if width == 0:
            raise ValueError("each cycle must contain at least one sensor value")
        if width > MAX_FEATURES:
            raise ValueError(f"each cycle may not exceed {MAX_FEATURES} values")

        for index, row in enumerate(value):
            if len(row) != width:
                raise ValueError(
                    f"ragged window: cycle 0 has {width} values, cycle {index} has {len(row)}"
                )
            if any(not math.isfinite(item) for item in row):
                raise ValueError(f"cycle {index} contains a non-finite value")
        return value

    @field_validator("readings")
    @classmethod
    def _check_readings(
        cls, value: list[dict[str, float]] | None
    ) -> list[dict[str, float]] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("readings must contain at least one cycle")
        if len(value) > MAX_CYCLES:
            raise ValueError(f"readings may not exceed {MAX_CYCLES} cycles")
        for index, reading in enumerate(value):
            if not reading:
                raise ValueError(f"cycle {index} has no sensor values")
            if any(not math.isfinite(item) for item in reading.values()):
                raise ValueError(f"cycle {index} contains a non-finite value")
        return value

    @model_validator(mode="after")
    def _exactly_one_input(self) -> "PredictRequest":
        if (self.window is None) == (self.readings is None):
            raise ValueError("provide exactly one of 'window' or 'readings'")
        return self

    @property
    def n_cycles(self) -> int:
        """Number of cycles supplied."""
        return len(self.window if self.window is not None else self.readings or [])


class PredictResponse(BaseModel):
    """A RUL prediction and the maintenance decision that follows from it."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "engine_id": 42,
                    "model": "gru",
                    "rul_cycles": 87.4,
                    "health_score": 69.9,
                    "risk_level": "watch",
                    "recommended_action": "Increase monitoring frequency and review sensor trends.",
                    "requires_action": False,
                    "window_size": 30,
                }
            ]
        }
    )

    engine_id: int | str | None = None
    model: str = Field(description="Model that produced the prediction.")
    rul_cycles: float = Field(description="Predicted remaining useful life, in cycles.")
    health_score: float = Field(ge=0, le=100, description="0 (failed) to 100 (healthy).")
    risk_level: RiskName
    recommended_action: str
    requires_action: bool = Field(
        description="True from the warning band upwards."
    )
    window_size: int = Field(description="Cycles the model consumed.")
    unit: str = "cycles"
    timestamp: datetime = Field(default_factory=_utc_now)
    attention: dict[str, Any] | None = Field(
        default=None, description="Per-cycle attention weights, when the model exposes them."
    )


class BatchPredictRequest(BaseModel):
    """Several machines scored in one round trip."""

    items: list[PredictRequest] = Field(
        min_length=1, max_length=MAX_BATCH, description=f"Up to {MAX_BATCH} requests."
    )
    model: ModelName | None = Field(
        default=None,
        description="Model applied to every item that does not name its own.",
    )


class BatchPredictResponse(BaseModel):
    """Batch results plus a fleet-level risk rollup."""

    count: int
    predictions: list[PredictResponse]
    risk_summary: dict[str, int] = Field(
        default_factory=dict, description="Machines per risk band."
    )
    action_required: int = Field(
        default=0, description="Machines in the warning band or worse."
    )
    timestamp: datetime = Field(default_factory=_utc_now)


class ModelInfo(BaseModel):
    """Metadata for one trained model."""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    type: str | None = None
    loaded: bool = False
    path: str | None = None
    window_size: int | None = None
    n_features: int | None = None
    feature_columns: list[str] | None = None
    n_parameters: int | None = None
    trained_epochs: int | None = None
    rul_cap: float | None = None
    validation_metrics: dict[str, float] | None = None
    size_mb: float | None = None
    device: str | None = None


class ModelListResponse(BaseModel):
    """Everything servable, and which model answers by default."""

    default_model: str
    count: int
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    """Service liveness for orchestrators and uptime checks."""

    status: Literal["ok", "degraded"] = "ok"
    version: str
    models_available: list[str] = Field(default_factory=list)
    models_loaded: list[str] = Field(default_factory=list)
    default_model: str | None = None
    scaler_available: bool = False
    uptime_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=_utc_now)


class LeaderboardResponse(BaseModel):
    """Offline evaluation results, as recorded during training."""

    count: int
    metric: str = "RMSE"
    best_model: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)


class DriftRequest(BaseModel):
    """Live windows to compare against the training distribution."""

    windows: list[list[list[float]]] = Field(
        min_length=1,
        description="Batch of sensor windows, shape (n, window_size, n_features).",
    )
    model: ModelName | None = Field(
        default=None, description="Model whose feature contract applies."
    )
    scaled: bool = Field(
        default=False, description="True if the windows are already standardised."
    )

    @field_validator("windows")
    @classmethod
    def _check_windows(cls, value: list[list[list[float]]]) -> list[list[list[float]]]:
        if len(value) > MAX_BATCH:
            raise ValueError(f"at most {MAX_BATCH} windows per request")
        widths = {len(row) for window in value for row in window}
        if len(widths) > 1:
            raise ValueError(f"inconsistent feature counts across cycles: {sorted(widths)}")
        return value


class DriftResponse(BaseModel):
    """Population Stability Index / KS outcome per feature, plus a verdict."""

    status: Literal["stable", "warning", "critical"]
    n_samples: int
    n_features: int
    drifted_features: list[str] = Field(default_factory=list)
    feature_share: float = Field(
        default=0.0, description="Share of features flagged as drifted (0-1)."
    )
    details: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utc_now)


class FleetEngine(BaseModel):
    """One machine's current standing in the fleet rollup."""

    engine_id: int | str = Field(..., description="Machine identifier.")
    rul_cycles: float = Field(..., description="Predicted remaining useful life.")
    health_score: float = Field(..., description="0 (failed) to 100 (as-new).")
    risk_level: str = Field(..., description="healthy | watch | warning | critical")
    recommended_action: str = Field(..., description="Maintenance recommendation.")
    cycles_observed: int = Field(..., description="Cycles recorded for this machine.")
    actual_rul: float | None = Field(
        None, description="Ground-truth RUL, when the source data carries labels."
    )


class FleetResponse(BaseModel):
    """Fleet-wide view backing the operations dashboard."""

    model: str
    count: int = Field(..., description="Machines scored.")
    risk_summary: dict[str, int] = Field(..., description="Machine count per risk band.")
    action_required: int = Field(..., description="Machines in warning or critical.")
    fleet_health: float = Field(..., description="Mean health score across the fleet.")
    median_rul: float = Field(..., description="Median predicted RUL in cycles.")
    engines: list[FleetEngine]
    source: str = Field(..., description="Dataset the rollup was computed from.")
    mae: float | None = Field(
        None, description="Mean absolute error, when ground truth is available."
    )

    model_config = ConfigDict(protected_namespaces=())


class EngineDetail(BaseModel):
    """Per-machine detail: prediction plus the sensor history behind it."""

    engine_id: int | str
    rul_cycles: float
    health_score: float
    risk_level: str
    recommended_action: str
    cycles_observed: int
    actual_rul: float | None = None
    model: str
    feature_columns: list[str] = Field(..., description="Sensors the model consumes.")
    sensor_history: dict[str, list[float]] = Field(
        ..., description="Recent raw values per sensor, oldest first."
    )
    cycles: list[int] = Field(..., description="Cycle numbers matching the history.")
    attention: list[float] | None = Field(
        None, description="Per-cycle attention weights, for the attention model."
    )

    model_config = ConfigDict(protected_namespaces=())


class ErrorResponse(BaseModel):
    """Uniform error body for 4xx and 5xx responses."""

    detail: str
    error_type: str | None = None
    timestamp: datetime = Field(default_factory=_utc_now)
