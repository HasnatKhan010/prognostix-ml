"""Shared PyTorch training loop, early stopping and checkpoint I/O.

The three sequence architectures differ only in their recurrent core, so they
share one trainer. Checkpoints stay backward compatible with the format the
notebooks wrote (``model_state_dict`` plus the constructor arguments) and add
the metadata inference needs: model type, feature order and window size.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

__all__ = [
    "EarlyStopping",
    "TrainingResult",
    "count_parameters",
    "evaluate",
    "fit",
    "load_checkpoint",
    "make_loaders",
    "predict",
    "save_checkpoint",
    "train_one_epoch",
    "validate",
]

CHECKPOINT_VERSION = 1


@dataclass
class TrainingResult:
    """Everything a training run produces besides the weights themselves."""

    model: nn.Module
    history: dict[str, list[float]] = field(default_factory=dict)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    epochs_run: int = 0
    duration_seconds: float = 0.0
    n_parameters: int = 0
    checkpoint_path: Path | None = None
    stopped_early: bool = False


class EarlyStopping:
    """Stop when validation loss stops improving, keeping the best weights.

    RUL models overfit quickly - validation loss typically bottoms out well
    before the configured epoch count - so the best snapshot is held in memory
    and restored at the end rather than keeping whatever the last epoch left.
    """

    def __init__(self, patience: int = 8, min_delta: float = 0.0):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
        self._best_state: dict[str, torch.Tensor] | None = None

    def step(self, loss: float, epoch: int, model: nn.Module) -> bool:
        """Record an epoch. Returns True when this epoch was an improvement."""
        improved = loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = float(loss)
            self.best_epoch = epoch
            self.counter = 0
            self._best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved

    def restore(self, model: nn.Module) -> nn.Module:
        """Load the best-seen weights back into ``model``."""
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
            logger.info(
                "Restored weights from epoch %d (val loss %.4f)",
                self.best_epoch,
                self.best_loss,
            )
        return model


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_loaders(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    """Wrap ``{split: (X, y)}`` arrays in DataLoaders.

    Only the ``train`` loader is shuffled; validation and test keep their order
    so predictions line up with the input rows.
    """
    loaders: dict[str, DataLoader] = {}
    for name, (X, y) in splits.items():
        dataset = TensorDataset(
            torch.tensor(np.asarray(X), dtype=torch.float32),
            torch.tensor(np.asarray(y), dtype=torch.float32),
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_train and name == "train",
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> float:
    """Run one training pass and return the sample-weighted mean loss."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()

        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate without gradients.

    Returns ``(mean_loss, predictions, targets)`` in loader order.
    """
    model.eval()
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item() * X_batch.size(0)

        predictions.append(outputs.detach().cpu().numpy())
        targets.append(y_batch.detach().cpu().numpy())

    mean_loss = total_loss / len(loader.dataset)
    return (
        mean_loss,
        np.concatenate(predictions) if predictions else np.array([]),
        np.concatenate(targets) if targets else np.array([]),
    )


# ``evaluate`` reads better at call sites that ignore the loss.
evaluate = validate


@torch.no_grad()
def predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Predict RUL for a batch of sequences without needing a DataLoader."""
    device = device or next(model.parameters()).device
    model.eval()

    array = np.asarray(X, dtype=np.float32)
    if array.ndim == 2:  # a single window
        array = array[None, ...]

    outputs: list[np.ndarray] = []
    for start in range(0, len(array), batch_size):
        batch = torch.tensor(array[start : start + batch_size], dtype=torch.float32)
        outputs.append(model(batch.to(device)).detach().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.array([])


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | str = "cpu",
    criterion: nn.Module | None = None,
    grad_clip: float | None = 1.0,
    early_stopping_patience: int | None = 8,
    min_delta: float = 0.0,
    lr_scheduler: str = "plateau",
    lr_patience: int = 3,
    lr_factor: float = 0.5,
    verbose: bool = True,
) -> TrainingResult:
    """Train a sequence model and return the run summary.

    Uses Adam with MSE loss, optional gradient clipping, optional
    ``ReduceLROnPlateau`` and optional early stopping with best-weight restore.
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    criterion = criterion or nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = None
    if lr_scheduler == "plateau" and val_loader is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience
        )

    stopper = (
        EarlyStopping(early_stopping_patience, min_delta)
        if early_stopping_patience and val_loader is not None
        else None
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    started = time.perf_counter()
    best_val = float("inf")
    best_epoch = 0
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip
        )
        history["train_loss"].append(train_loss)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        val_loss = float("nan")
        if val_loader is not None:
            val_loss, _, _ = validate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            if scheduler is not None:
                scheduler.step(val_loss)
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch

        if verbose:
            message = f"Epoch {epoch:02d}/{epochs} | train {train_loss:.4f}"
            if val_loader is not None:
                message += f" | val {val_loss:.4f}"
            logger.info(message)

        if stopper is not None:
            stopper.step(val_loss, epoch, model)
            if stopper.should_stop:
                logger.info(
                    "Early stopping at epoch %d; no improvement for %d epoch(s)",
                    epoch,
                    stopper.patience,
                )
                break

    stopped_early = bool(stopper and stopper.should_stop)
    if stopper is not None:
        stopper.restore(model)
        best_val, best_epoch = stopper.best_loss, stopper.best_epoch

    if not history["val_loss"]:
        history.pop("val_loss")

    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        epochs_run=epochs_run,
        duration_seconds=time.perf_counter() - started,
        n_parameters=count_parameters(model),
        stopped_early=stopped_early,
    )


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    model_type: str,
    input_size: int,
    model_kwargs: dict[str, Any] | None = None,
    feature_columns: Iterable[str] | None = None,
    window_size: int | None = None,
    metrics: dict[str, Any] | None = None,
    history: dict[str, list[float]] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save weights plus the metadata needed to rebuild and serve the model.

    The constructor arguments are stored flat (``hidden_size``, ``num_layers``,
    ``dropout``, ...) alongside a nested ``model_kwargs`` copy, which keeps the
    file readable by the loading code in the original notebooks.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_kwargs = dict(model_kwargs or {})

    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "model_type": model_type,
        "input_size": int(input_size),
        "model_kwargs": model_kwargs,
        **model_kwargs,
    }
    if feature_columns is not None:
        payload["feature_columns"] = list(feature_columns)
    if window_size is not None:
        payload["window_size"] = int(window_size)
    if metrics:
        payload["metrics"] = dict(metrics)
    if history:
        payload["history"] = {k: list(v) for k, v in history.items()}
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    logger.info("Saved checkpoint -> %s", path)
    return path


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load a checkpoint dictionary written by :func:`save_checkpoint`.

    Also accepts the leaner files produced by the original notebooks, which
    carry no ``model_type``; callers infer it from the filename in that case.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train the model first: python scripts/train.py"
        )
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"{path} is not a Prognostix checkpoint")
    return payload
