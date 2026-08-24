"""Diagnostic plots for RUL models.

All functions accept an optional ``save_path`` and return the Matplotlib figure,
so they work both inline in a notebook and headless inside ``scripts/``. A
non-interactive backend is selected automatically when no display is available.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

if not os.environ.get("DISPLAY") and os.name != "nt":  # pragma: no cover - env dependent
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "plot_actual_vs_predicted",
    "plot_engine_trajectory",
    "plot_model_comparison",
    "plot_residuals",
    "plot_training_history",
]

# Sequential-to-categorical accent set; index 0 is the primary series.
PALETTE = ("#2b6cb0", "#dd6b20", "#38a169", "#805ad5", "#d53f8c")
GRID_STYLE = {"alpha": 0.3, "linestyle": ":", "linewidth": 0.8}


def _finish(fig: plt.Figure, save_path: str | Path | None, show: bool) -> plt.Figure:
    """Apply tight layout, optionally save, optionally show."""
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure -> %s", path)
    if show:  # pragma: no cover - interactive only
        plt.show()
    return fig


def plot_training_history(
    history: Mapping[str, Sequence[float]],
    title: str = "Training history",
    ylabel: str = "MSE loss",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot train/validation loss curves.

    A validation curve that flattens while the training curve keeps falling is
    the signal that early stopping should have fired.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, (name, values) in enumerate(history.items()):
        if not len(values):
            continue
        ax.plot(
            range(1, len(values) + 1),
            values,
            label=name.replace("_", " ").title(),
            color=PALETTE[index % len(PALETTE)],
            linewidth=1.8,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(**GRID_STYLE)
    ax.legend(frameon=False)
    return _finish(fig, save_path, show)


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    title: str = "Actual vs predicted RUL",
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Scatter predictions against ground truth with the ideal diagonal."""
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true, pred, alpha=0.35, s=14, color=PALETTE[0], edgecolors="none")

    low = float(min(true.min(), pred.min()))
    high = float(max(true.max(), pred.max()))
    ax.plot([low, high], [low, high], linestyle="--", color="#4a5568", linewidth=1.2)

    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(title)
    ax.grid(**GRID_STYLE)
    ax.set_aspect("equal", adjustable="box")
    return _finish(fig, save_path, show)


def plot_residuals(
    y_true,
    y_pred,
    title: str = "Residual distribution",
    bins: int = 40,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Histogram of residuals plus residuals against true RUL.

    The right panel is the one that matters operationally: a fan that widens at
    high RUL is expected (early degradation is unobservable), but bias near
    RUL=0 means the model misses imminent failures.
    """
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    residuals = true - pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(residuals, bins=bins, color=PALETTE[0], alpha=0.85)
    axes[0].axvline(0, color="#4a5568", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Error (actual - predicted)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{title} (mean {residuals.mean():+.2f})")
    axes[0].grid(**GRID_STYLE)

    axes[1].scatter(true, residuals, alpha=0.3, s=12, color=PALETTE[1], edgecolors="none")
    axes[1].axhline(0, color="#4a5568", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Actual RUL (cycles)")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Error vs actual RUL")
    axes[1].grid(**GRID_STYLE)

    return _finish(fig, save_path, show)


def plot_model_comparison(
    leaderboard: pd.DataFrame,
    metric: str = "RMSE",
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Horizontal bar chart ranking models by one metric (lower is better)."""
    if leaderboard.empty:
        raise ValueError("Leaderboard is empty")
    for column in ("Model", metric):
        if column not in leaderboard.columns:
            raise ValueError(f"Leaderboard has no {column!r} column")

    frame = leaderboard.dropna(subset=[metric]).sort_values(metric, ascending=False)
    best = frame[metric].min()
    colors = [PALETTE[0] if value == best else "#a0aec0" for value in frame[metric]]

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(frame) + 2.2))
    bars = ax.barh(frame["Model"].astype(str), frame[metric], color=colors)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)

    ax.set_xlabel(f"{metric} (cycles, lower is better)")
    ax.set_title(title or f"Model comparison by {metric}")
    ax.grid(axis="x", **GRID_STYLE)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.12)
    return _finish(fig, save_path, show)


def plot_engine_trajectory(
    y_true,
    y_pred,
    engine_id: int | str = "",
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot predicted against actual RUL over one engine's life."""
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    cycles = np.arange(1, len(true) + 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cycles, true, label="Actual RUL", color="#4a5568", linewidth=1.8)
    ax.plot(cycles, pred, label="Predicted RUL", color=PALETTE[0], linewidth=1.8)
    ax.fill_between(cycles, true, pred, color=PALETTE[0], alpha=0.12)

    ax.set_xlabel("Cycle index")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(title or f"RUL trajectory{f' - engine {engine_id}' if engine_id != '' else ''}")
    ax.grid(**GRID_STYLE)
    ax.legend(frameon=False)
    return _finish(fig, save_path, show)
