"""Prognostix ML - predictive maintenance for industrial equipment."""

from __future__ import annotations

from src.config import (
    PROJECT_ROOT,
    Config,
    get_config,
    get_device,
    load_config,
    set_seed,
    setup_logging,
)

__version__ = "0.1.0"

__all__ = [
    "PROJECT_ROOT",
    "Config",
    "__version__",
    "get_config",
    "get_device",
    "load_config",
    "set_seed",
    "setup_logging",
]
