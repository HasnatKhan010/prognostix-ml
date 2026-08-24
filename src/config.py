"""Configuration loading, path resolution and run-environment helpers.

Everything in the project reads its settings from ``configs/config.yaml`` through
:func:`load_config`, so a single file drives notebooks, scripts, the API and the
monitoring jobs.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "PROJECT_ROOT",
    "Config",
    "get_config",
    "get_device",
    "load_config",
    "set_seed",
    "setup_logging",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

_CACHED_CONFIG: Config | None = None


class Config(Mapping):
    """Read-only nested mapping with attribute access.

    ``config.data.window_size`` and ``config["data"]["window_size"]`` are
    equivalent; nested dictionaries are wrapped lazily.
    """

    def __init__(self, data: Mapping[str, Any]):
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, Mapping) else value

    def __getattr__(self, key: str) -> Any:
        # Private and dunder names must never route through __getitem__: during
        # unpickling or copying, self._data may not exist yet and the lookup
        # would recurse.
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - attribute protocol
            raise AttributeError(
                f"{key!r} is not defined in the configuration"
            ) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({self._data!r})"

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, deeply-copied ``dict``."""

        def _plain(value: Any) -> Any:
            if isinstance(value, Config):
                return value.to_dict()
            if isinstance(value, Mapping):
                return {k: _plain(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_plain(v) for v in value]
            return value

        return {key: _plain(value) for key, value in self._data.items()}

    # -- convenience -----------------------------------------------------

    def path(self, key: str) -> Path:
        """Resolve ``paths.<key>`` against the project root."""
        return (PROJECT_ROOT / self._data["paths"][key]).resolve()

    @property
    def sensor_columns(self) -> list[str]:
        """All raw sensor column names, before any are dropped."""
        return [f"sensor_{i}" for i in range(1, int(self._data["data"]["n_sensors"]) + 1)]

    @property
    def raw_columns(self) -> list[str]:
        """Column names of a raw CMAPSS ``*.txt`` file, in file order."""
        data = self._data["data"]
        return [
            data["id_column"],
            data["time_column"],
            *data["setting_columns"],
            *self.sensor_columns,
        ]


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Configuration file to read. Defaults to ``configs/config.yaml``, or to
        ``$PROGNOSTIX_CONFIG`` when that environment variable is set.
    """
    if path is None:
        path = os.environ.get("PROGNOSTIX_CONFIG", DEFAULT_CONFIG_PATH)
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{config_path} must contain a YAML mapping at the top level")
    return Config(raw)


def get_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Return a process-wide cached configuration (loaded on first call)."""
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None or path is not None:
        _CACHED_CONFIG = load_config(path)
    return _CACHED_CONFIG


def set_seed(seed: int = 42) -> None:
    """Seed ``random``, ``numpy`` and - when installed - ``torch``."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # torch is optional for the tabular pipeline
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto"):
    """Resolve a ``torch.device`` from ``auto`` / ``cpu`` / ``cuda``."""
    import torch

    if preference == "auto":
        preference = "cuda" if torch.cuda.is_available() else "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        logging.getLogger(__name__).warning("CUDA requested but unavailable; using CPU")
        preference = "cpu"
    return torch.device(preference)


def setup_logging(level: int | str = logging.INFO) -> logging.Logger:
    """Configure root logging once and return the project logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("prognostix")


def ensure_dirs(config: Config) -> None:
    """Create every directory declared under ``paths``."""
    for key in config["paths"]:
        config.path(key).mkdir(parents=True, exist_ok=True)
