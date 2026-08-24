"""Loading trained models and serving RUL predictions.

The predictor owns the whole inference contract: which sensor columns arrive in
which order, the scaler those columns must pass through, the window length the
model expects, and the health assessment derived from the output. Skipping any
of it silently produces plausible-looking nonsense - a model trained on
standardised inputs given raw sensor values returns numbers, just not correct
ones - so every entry point validates before predicting.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config, get_config, get_device
from src.inference.health_score import HealthAssessment, assess_health
from src.ingestion.validator import validate_window
from src.preprocessing.scaling import ScalerBundle, load_scaler

logger = logging.getLogger(__name__)

__all__ = ["ModelRegistry", "RULPredictor"]

#: Checkpoint filenames tried per model, newest naming first. The
#: ``*_baseline.*`` entries are the artifacts the original notebooks wrote.
MODEL_FILES: dict[str, tuple[str, ...]] = {
    "gru": ("gru.pt", "gru_baseline.pt"),
    "lstm": ("lstm.pt", "lstm_baseline.pt"),
    "attention": ("attention.pt", "attention_baseline.pt"),
    "random_forest": ("random_forest.joblib", "random_forest_baseline.joblib"),
    "linear": (
        "linear.joblib",
        "linear_regression.joblib",
        "linear_regression_baseline.joblib",
    ),
}

TORCH_SUFFIXES = {".pt", ".pth"}


class RULPredictor:
    """Serve RUL predictions from one trained model.

    Parameters
    ----------
    model_name:
        ``gru``, ``lstm``, ``attention``, ``random_forest`` or ``linear``.
        Defaults to ``inference.default_model``.
    model_path:
        Explicit checkpoint path, bypassing the filename search.
    scaler:
        Pre-loaded :class:`~src.preprocessing.scaling.ScalerBundle`. Loaded from
        ``artifacts/`` when omitted.
    lazy:
        Delay loading the weights until the first prediction. The API uses this
        to keep startup fast for models nobody has asked for yet.
    """

    def __init__(
        self,
        model_name: str | None = None,
        config: Config | None = None,
        model_path: str | Path | None = None,
        scaler: ScalerBundle | None = None,
        device: str | None = None,
        lazy: bool = False,
    ):
        self.config = config or get_config()
        self.model_name = model_name or str(self.config.inference.default_model)
        self.model_path = Path(model_path) if model_path else None
        self.device_preference = device or str(self.config.training.get("device", "auto"))

        self._model: Any = None
        self._scaler: ScalerBundle | None = scaler
        self._metadata: dict[str, Any] = {}
        self._device = None

        if not lazy:
            self.load()

    # -- loading ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_torch(self) -> bool:
        """True when the backing model is a PyTorch module."""
        return self.resolve_path().suffix in TORCH_SUFFIXES

    def resolve_path(self) -> Path:
        """Locate the checkpoint for this model name.

        Raises
        ------
        FileNotFoundError
            When no candidate filename exists, listing what was tried.
        """
        if self.model_path is not None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"{self.model_path} not found")
            return self.model_path

        directory = self.config.path("models")
        candidates = MODEL_FILES.get(self.model_name)
        if candidates is None:
            raise ValueError(
                f"Unknown model {self.model_name!r}; choose from {sorted(MODEL_FILES)}"
            )

        for filename in candidates:
            candidate = directory / filename
            if candidate.exists():
                self.model_path = candidate
                return candidate

        raise FileNotFoundError(
            f"No checkpoint for {self.model_name!r} in {directory}. Tried: "
            f"{', '.join(candidates)}. Train it with "
            f"`python scripts/train.py --model {self.model_name}`."
        )

    def load(self) -> RULPredictor:
        """Load the scaler, weights and metadata."""
        path = self.resolve_path()

        # The scaler comes first: it carries the feature names that a bare
        # notebook-era sklearn artifact does not record for itself.
        if self._scaler is None:
            try:
                self._scaler = load_scaler(config=self.config)
            except FileNotFoundError:
                logger.warning(
                    "No scaler bundle in %s - raw inputs cannot be scaled. "
                    "Run scripts/prepare_data.py, or pass pre-scaled windows.",
                    self.config.path("artifacts"),
                )

        if path.suffix in TORCH_SUFFIXES:
            self._load_torch(path)
        else:
            self._load_sklearn(path)

        logger.info(
            "Loaded %s from %s (window=%s, features=%s)",
            self.model_name,
            path.name,
            self.window_size,
            self.n_features,
        )
        return self

    def _load_torch(self, path: Path) -> None:
        from src.models import build_model
        from src.models.common import load_checkpoint

        self._device = get_device(self.device_preference)
        checkpoint = load_checkpoint(path, map_location=self._device)

        model_type = checkpoint.get("model_type", self.model_name)
        input_size = int(checkpoint["input_size"])
        kwargs = dict(checkpoint.get("model_kwargs") or {})
        if not kwargs:
            # Notebook-era checkpoints store constructor arguments flat.
            kwargs = {
                key: checkpoint[key]
                for key in ("hidden_size", "num_layers", "dropout", "num_heads", "bidirectional")
                if key in checkpoint
            }

        model = build_model(model_type, input_size=input_size, **kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self._device).eval()

        self._model = model
        self._metadata = {
            key: value for key, value in checkpoint.items() if key != "model_state_dict"
        }
        self._metadata.setdefault("model_type", model_type)

    def _load_sklearn(self, path: Path) -> None:
        from src.models.baseline.random_forest import TabularRULModel

        feature_columns = self._scaler.feature_columns if self._scaler else None
        bundle = TabularRULModel.load(path, feature_columns=feature_columns)
        self._model = bundle
        self._metadata = {
            "model_type": self.model_name,
            "estimator": type(bundle.estimator).__name__,
            "stats": list(bundle.stats),
            "window_size": bundle.window_size,
            "input_size": len(bundle.feature_columns) if bundle.feature_columns else None,
            **bundle.metadata,
        }

    def _require_model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    # -- contract --------------------------------------------------------

    @property
    def scaler(self) -> ScalerBundle | None:
        return self._scaler

    @property
    def feature_columns(self) -> list[str] | None:
        """Sensor columns the model expects, in order."""
        columns = self._metadata.get("feature_columns")
        if columns:
            return list(columns)
        if self._scaler is not None:
            return list(self._scaler.feature_columns)
        return None

    @property
    def n_features(self) -> int | None:
        """Number of sensor channels per cycle."""
        size = self._metadata.get("input_size")
        if size:
            return int(size)
        columns = self.feature_columns
        return len(columns) if columns else None

    @property
    def window_size(self) -> int:
        """Cycles per input window."""
        for source in (self._metadata.get("window_size"),
                       self._scaler.window_size if self._scaler else None):
            if source:
                return int(source)
        return int(self.config.data.window_size)

    def info(self) -> dict[str, Any]:
        """Serving metadata for the ``/models`` endpoint."""
        metrics = self._metadata.get("metrics") or {}
        return {
            "name": self.model_name,
            "type": self._metadata.get("model_type", self.model_name),
            "path": str(self.model_path) if self.model_path else None,
            "loaded": self.is_loaded,
            "window_size": self.window_size,
            "n_features": self.n_features,
            "feature_columns": self.feature_columns,
            "n_parameters": self._metadata.get("n_parameters"),
            "trained_epochs": self._metadata.get("epochs_run"),
            "rul_cap": self._metadata.get("rul_cap", self.config.data.get("rul_cap")),
            "validation_metrics": metrics.get("val") if isinstance(metrics, dict) else None,
            "device": str(self._device) if self._device else "cpu",
        }

    # -- prediction ------------------------------------------------------

    def predict(
        self, windows: np.ndarray | Sequence, scaled: bool = False, validate: bool = True
    ) -> np.ndarray:
        """Predict RUL for one window or a batch.

        Parameters
        ----------
        windows:
            ``(window_size, n_features)`` or ``(n, window_size, n_features)``.
        scaled:
            Set True only when the values already went through this model's
            scaler - e.g. arrays straight out of ``data/processed``.
        validate:
            Check shape and finiteness before predicting.

        Returns
        -------
        RUL in cycles, clipped at zero (negative remaining life is meaningless).
        """
        model = self._require_model()
        array = np.asarray(windows, dtype=float)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3:
            raise ValueError(
                f"Expected 2-D or 3-D input, got shape {np.shape(windows)}"
            )
        if array.shape[0] == 0:
            return np.array([], dtype=float)

        if validate:
            expected_features = self.n_features or array.shape[2]
            for index, window in enumerate(array):
                report = validate_window(window, self.window_size, expected_features)
                if not report.ok:
                    raise ValueError(f"Window {index}: {'; '.join(report.errors)}")

        if not scaled:
            if self._scaler is None:
                raise RuntimeError(
                    "No scaler available: cannot transform raw sensor values. "
                    "Run scripts/prepare_data.py or pass scaled=True."
                )
            array = self._scaler.transform_sequences(array)

        if self.is_torch:
            from src.models.common import predict as torch_predict

            predictions = torch_predict(model, array, device=self._device)
        else:
            predictions = model.predict(array)

        return np.clip(np.asarray(predictions, dtype=float).ravel(), 0.0, None)

    def predict_one(self, window, scaled: bool = False) -> float:
        """Predict RUL for a single window."""
        return float(self.predict(window, scaled=scaled)[0])

    def window_from_readings(
        self, readings: Iterable[Mapping[str, float]]
    ) -> np.ndarray:
        """Build a numeric window from per-cycle sensor dictionaries.

        Each reading maps sensor name to value; only the model's feature columns
        are kept and they are ordered to match training, so callers may send the
        full 21-sensor payload and let the predictor pick what it needs.
        """
        columns = self.feature_columns
        if columns is None:
            raise RuntimeError(
                "Feature column names are unknown for this model; send a numeric "
                "window instead of named readings."
            )

        frame = pd.DataFrame(list(readings))
        if frame.empty:
            raise ValueError("No readings supplied")
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Readings are missing required sensors: {missing}")

        return frame[columns].to_numpy(dtype=float)

    def predict_from_readings(
        self,
        readings: Iterable[Mapping[str, float]],
        scaled: bool = False,
    ) -> float:
        """Predict from a list of per-cycle sensor dictionaries."""
        return self.predict_one(self.window_from_readings(readings), scaled=scaled)

    def assess(
        self,
        window,
        scaled: bool = False,
        engine_id: int | str | None = None,
    ) -> HealthAssessment:
        """Predict and wrap the result in a health assessment."""
        rul = self.predict_one(window, scaled=scaled)
        return assess_health(
            rul, config=self.config, engine_id=engine_id, model=self.model_name
        )

    def assess_batch(
        self,
        windows,
        scaled: bool = False,
        engine_ids: Sequence[int | str] | None = None,
    ) -> list[HealthAssessment]:
        """Predict a batch and assess each window."""
        predictions = self.predict(windows, scaled=scaled)
        ids = list(engine_ids) if engine_ids is not None else [None] * len(predictions)
        if len(ids) != len(predictions):
            raise ValueError(
                f"engine_ids length {len(ids)} does not match {len(predictions)} windows"
            )
        return [
            assess_health(rul, config=self.config, engine_id=engine_id, model=self.model_name)
            for rul, engine_id in zip(predictions, ids, strict=False)
        ]

    def predict_frame(
        self, frame: pd.DataFrame, scaled: bool = False
    ) -> pd.DataFrame:
        """Score every engine in a cycle-level frame, using its latest window.

        Returns one row per engine with the prediction and its health assessment.
        """
        from src.preprocessing.sequences import last_window_per_engine

        columns = self.feature_columns
        if columns is None:
            raise RuntimeError("Feature column names are unknown for this model")
        data = self.config.data

        windows, engine_ids = last_window_per_engine(
            frame,
            feature_columns=columns,
            window_size=self.window_size,
            id_column=data.id_column,
            time_column=data.time_column,
        )
        if len(windows) == 0:
            return pd.DataFrame(
                columns=["engine_id", "rul", "health_score", "risk_level", "recommended_action"]
            )

        assessments = self.assess_batch(windows, scaled=scaled, engine_ids=engine_ids)
        return pd.DataFrame(
            [
                {
                    "engine_id": assessment.engine_id,
                    "rul": round(assessment.rul, 2),
                    "health_score": round(assessment.health_score, 2),
                    "risk_level": assessment.risk_level.value,
                    "recommended_action": assessment.recommended_action,
                    "model": self.model_name,
                }
                for assessment in assessments
            ]
        )

    def explain(self, window, scaled: bool = False) -> dict[str, Any] | None:
        """Per-cycle attention weights, for models that expose them.

        Returns ``None`` for architectures without attention.
        """
        model = self._require_model()
        if not hasattr(model, "attention_weights"):
            return None
        import torch

        array = np.asarray(window, dtype=float)
        if array.ndim == 2:
            array = array[None, ...]
        if not scaled:
            if self._scaler is None:
                raise RuntimeError("No scaler available to transform raw values")
            array = self._scaler.transform_sequences(array)

        tensor = torch.tensor(array, dtype=torch.float32, device=self._device)
        weights = model.attention_weights(tensor).cpu().numpy()
        return {
            "attention_weights": weights[0].tolist(),
            "most_influential_cycle": int(np.argmax(weights[0])) - self.window_size,
        }


class ModelRegistry:
    """Lazily-loaded pool of predictors, one per model name.

    The API holds a single registry: the default model is loaded at startup and
    the rest - including the 180 MB Random Forest - only when first requested.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._predictors: dict[str, RULPredictor] = {}

    @property
    def default_model(self) -> str:
        return str(self.config.inference.default_model)

    def available(self) -> list[str]:
        """Model names with a checkpoint present on disk."""
        directory = self.config.path("models")
        return [
            name
            for name, filenames in MODEL_FILES.items()
            if any((directory / filename).exists() for filename in filenames)
        ]

    def get(self, name: str | None = None) -> RULPredictor:
        """Return a predictor, loading it on first use."""
        name = name or self.default_model
        if name not in MODEL_FILES:
            raise ValueError(
                f"Unknown model {name!r}; choose from {sorted(MODEL_FILES)}"
            )
        if name not in self._predictors:
            self._predictors[name] = RULPredictor(model_name=name, config=self.config)
        return self._predictors[name]

    def preload(self, names: Iterable[str] | None = None) -> list[str]:
        """Eagerly load models, skipping any that fail.

        A missing checkpoint must not stop the API from starting - the remaining
        models stay serveable and ``/models`` reports what is actually loaded.
        """
        names = list(names if names is not None else self.config.api.get("preload_models", []))
        loaded: list[str] = []
        for name in names:
            try:
                self.get(name)
                loaded.append(name)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Could not preload %s: %s", name, exc)
        return loaded

    def info(self) -> list[dict[str, Any]]:
        """Describe every model on disk without forcing a load."""
        directory = self.config.path("models")
        entries: list[dict[str, Any]] = []
        for name, filenames in MODEL_FILES.items():
            path = next(
                (directory / f for f in filenames if (directory / f).exists()), None
            )
            if path is None:
                continue
            if name in self._predictors:
                entries.append(self._predictors[name].info())
            else:
                entries.append(
                    {
                        "name": name,
                        "type": name,
                        "path": str(path),
                        "loaded": False,
                        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                    }
                )
        return entries

    def unload(self, name: str) -> bool:
        """Drop a loaded predictor, freeing its memory."""
        return self._predictors.pop(name, None) is not None
