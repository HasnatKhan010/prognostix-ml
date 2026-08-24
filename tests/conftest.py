"""Shared fixtures.

Every fixture that writes something points at ``tmp_path``, so the suite never
touches the committed ``data/`` or ``artifacts/`` trees and can run on a clean
checkout with no trained model present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config, load_config

N_SENSORS = 21
CONSTANT_SENSORS = ("sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19")
WINDOW = 10


@pytest.fixture(scope="session")
def base_config() -> Config:
    """The real project configuration (read-only use only)."""
    return load_config()


@pytest.fixture
def config(tmp_path, base_config) -> Config:
    """Project configuration with every output path redirected into ``tmp_path``."""
    payload = base_config.to_dict()
    payload["paths"] = {
        "data_raw": str(tmp_path / "data" / "raw" / "CMAPSS"),
        "data_processed": str(tmp_path / "data" / "processed"),
        "artifacts": str(tmp_path / "artifacts"),
        "models": str(tmp_path / "artifacts" / "models"),
        "figures": str(tmp_path / "artifacts" / "figures"),
        "reports": str(tmp_path / "artifacts" / "reports"),
    }
    payload["data"]["window_size"] = WINDOW
    config = Config(payload)
    for key in payload["paths"]:
        config.path(key).mkdir(parents=True, exist_ok=True)
    return config


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Synthetic CMAPSS-shaped frame: 12 engines, degrading sensors, some constant.

    Engine lengths vary (including one shorter than the window) so windowing and
    padding logic is exercised, and the constant channels mirror the six that
    never move in FD001.
    """
    rng = np.random.default_rng(7)
    rows: list[dict[str, float]] = []

    for engine_id in range(1, 13):
        length = 8 if engine_id == 12 else int(rng.integers(25, 60))
        for cycle in range(1, length + 1):
            wear = cycle / length  # 0 -> 1 across the engine's life
            row: dict[str, float] = {
                "engine_id": engine_id,
                "cycle": cycle,
                "setting_1": float(rng.normal(0, 0.002)),
                "setting_2": float(rng.normal(0, 0.0003)),
                "setting_3": 100.0,
            }
            for sensor in range(1, N_SENSORS + 1):
                name = f"sensor_{sensor}"
                if name in CONSTANT_SENSORS:
                    row[name] = 100.0 + sensor
                else:
                    row[name] = 500.0 + sensor + 25.0 * wear + float(rng.normal(0, 0.4))
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def labelled_frame(raw_frame) -> pd.DataFrame:
    """``raw_frame`` with a linear RUL target."""
    from src.ingestion.loader import add_rul

    return add_rul(raw_frame)


@pytest.fixture
def feature_columns(labelled_frame, config) -> list[str]:
    """The 15 informative sensors, in model order."""
    from src.preprocessing.cleaning import select_feature_columns

    return select_feature_columns(labelled_frame, config)


@pytest.fixture
def sequences(labelled_frame, feature_columns):
    """``(X, y)`` windows built from the synthetic frame."""
    from src.preprocessing.sequences import create_sequences

    return create_sequences(
        labelled_frame, feature_columns, "RUL", window_size=WINDOW
    )


@pytest.fixture
def scaler_bundle(labelled_frame, feature_columns):
    """A StandardScaler bundle fitted on the synthetic frame."""
    from src.preprocessing.scaling import fit_scaler

    return fit_scaler(
        labelled_frame, feature_columns, kind="standard", window_size=WINDOW
    )


@pytest.fixture
def prepared_dataset(config, labelled_frame, feature_columns, scaler_bundle):
    """Write scaled train/val/test ``.npz`` splits and the scaler into ``tmp_path``.

    Mirrors what ``scripts/prepare_data.py`` produces, so anything that reads
    ``data/processed`` can be tested without running the real pipeline.
    """
    from src.ingestion.loader import save_sequences
    from src.preprocessing.scaling import apply_scaler, save_scaler
    from src.preprocessing.sequences import create_sequences, split_by_engine

    engines = labelled_frame["engine_id"].unique()
    groups = {
        "train": engines[:8],
        "val": engines[8:10],
        "test": engines[10:],
    }

    save_scaler(scaler_bundle, config=config)
    written: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, ids in groups.items():
        part = apply_scaler(
            split_by_engine(labelled_frame, ids, "engine_id"), scaler_bundle
        )
        X, y, engine_ids = create_sequences(
            part, scaler_bundle.feature_columns, "RUL", WINDOW, return_ids=True
        )
        save_sequences(name, X, y, config=config, engine_ids=engine_ids)
        written[name] = (X, y)

    return written


@pytest.fixture
def torch_checkpoint(config, prepared_dataset):
    """An untrained GRU checkpoint saved in the temp models directory.

    Untrained weights are deliberate: these tests assert plumbing - shapes,
    scaling, validation, HTTP status codes - not predictive accuracy.
    """
    from src.models import build_model
    from src.models.common import save_checkpoint
    from src.preprocessing.scaling import load_scaler

    X, _ = prepared_dataset["train"]
    input_size = int(X.shape[2])
    model = build_model("gru", input_size=input_size, hidden_size=16, num_layers=1, dropout=0.0)

    return save_checkpoint(
        model,
        config.path("models") / "gru.pt",
        model_type="gru",
        input_size=input_size,
        model_kwargs={"hidden_size": 16, "num_layers": 1, "dropout": 0.0},
        feature_columns=load_scaler(config=config).feature_columns,
        window_size=WINDOW,
        metrics={"val": {"MAE": 20.0, "RMSE": 28.0}},
    )


@pytest.fixture
def predictor(config, torch_checkpoint):
    """A fully wired predictor over the temp checkpoint and scaler."""
    from src.inference.predictor import RULPredictor

    return RULPredictor(model_name="gru", config=config)


@pytest.fixture
def api_client(config, predictor):
    """FastAPI test client whose registry serves the temp predictor.

    The app is not entered as a context manager, so its lifespan never loads the
    committed artifacts; the registry is injected directly instead.
    """
    import time

    from fastapi.testclient import TestClient

    from api.main import app
    from src.inference.predictor import ModelRegistry

    class StubRegistry(ModelRegistry):
        def __init__(self):
            super().__init__(config)
            self._predictors = {predictor.model_name: predictor}

        def available(self):
            return [predictor.model_name]

        def info(self):
            return [predictor.info()]

        def get(self, name=None):
            name = name or self.default_model
            if name == predictor.model_name:
                return predictor
            if name in ("lstm", "attention", "random_forest", "linear"):
                raise FileNotFoundError(f"No checkpoint for {name!r}")
            raise ValueError(f"Unknown model {name!r}")

    client = TestClient(app, raise_server_exceptions=False)
    app.state.registry = StubRegistry()
    app.state.started_at = time.time()
    return client
