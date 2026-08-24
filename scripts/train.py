#!/usr/bin/env python
"""Train one model - or all of them - on the prepared sequences.

Every model is trained on the same windows with the same seed, so the resulting
leaderboard reflects architecture differences and nothing else.

Usage::

    python scripts/train.py --model gru
    python scripts/train.py --model lstm --epochs 50 --device cuda
    python scripts/train.py --model all --test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, setup_logging  # noqa: E402
from src.evaluation.compare import load_leaderboard  # noqa: E402
from src.models import SKLEARN_MODELS, TORCH_MODELS  # noqa: E402

logger = logging.getLogger("prognostix.train")

CHOICES = ("all", "baselines", *SKLEARN_MODELS, *TORCH_MODELS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gru",
        choices=CHOICES,
        help="Model to train. 'baselines' trains mean/linear/random_forest; "
        "'all' adds the sequence models.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument(
        "--device", default=None, choices=["auto", "cpu", "cuda"], help="Compute device."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Also score the held-out test split (leave off during model selection).",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not write artifacts.")
    parser.add_argument("--no-plots", action="store_true", help="Skip diagnostic figures.")
    args = parser.parse_args(argv)

    setup_logging()
    config = get_config()

    if not (config.path("data_processed") / "train_sequences.npz").exists():
        logger.error(
            "No prepared sequences in %s. Run `python scripts/prepare_data.py` first.",
            config.path("data_processed"),
        )
        return 1

    targets = _resolve_targets(args.model)
    logger.info("Training: %s", ", ".join(targets))

    results: dict[str, dict] = {}
    for name in targets:
        try:
            results[name] = _train_one(name, args, config)
        except FileNotFoundError as exc:
            logger.error("Skipping %s: %s", name, exc)
        except Exception:
            logger.exception("Training %s failed", name)
            if len(targets) == 1:
                return 1

    if not results:
        logger.error("No model trained successfully")
        return 1

    _print_summary(results, config)
    return 0


def _resolve_targets(choice: str) -> list[str]:
    """Expand the ``--model`` flag into concrete model names."""
    if choice == "all":
        return [*SKLEARN_MODELS, *TORCH_MODELS]
    if choice == "baselines":
        return list(SKLEARN_MODELS)
    return [choice]


def _train_one(name: str, args: argparse.Namespace, config) -> dict:
    """Dispatch to the tabular or sequence trainer."""
    if name in SKLEARN_MODELS:
        from src.models.baseline.train import train_baseline

        return train_baseline(
            name,
            config=config,
            evaluate_test=args.test,
            save=not args.no_save,
        )

    from src.models.runner import run_sequence_training

    return run_sequence_training(
        name,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        evaluate_test=args.test,
        save=not args.no_save,
        make_plots=not args.no_plots,
    )


def _print_summary(results: dict[str, dict], config) -> None:
    """Print per-run metrics followed by the refreshed leaderboard."""
    print("\n" + "=" * 68)
    print("TRAINING SUMMARY (validation split)")
    print("=" * 68)
    print(f"{'model':<16}{'MAE':>9}{'RMSE':>9}{'R2':>9}{'NASA':>12}")
    print("-" * 68)
    for name, result in results.items():
        metrics = result["val_metrics"]
        print(
            f"{name:<16}{metrics['MAE']:>9.3f}{metrics['RMSE']:>9.3f}"
            f"{metrics['R2']:>9.3f}{metrics.get('NASAScore', float('nan')):>12.1f}"
        )

    if any(result.get("test_metrics") for result in results.values()):
        print("\nTest split:")
        for name, result in results.items():
            metrics = result.get("test_metrics")
            if metrics:
                print(
                    f"  {name:<14} MAE {metrics['MAE']:.3f} | RMSE {metrics['RMSE']:.3f}"
                )

    leaderboard = load_leaderboard(config=config)
    if not leaderboard.empty:
        columns = [c for c in ("Model", "MAE", "RMSE", "R2", "NASAScore") if c in leaderboard.columns]
        print("\nLeaderboard (artifacts/model_comparison.csv):")
        print(leaderboard[columns].round(3).to_string(index=False))

    print("\nNext step: python scripts/evaluate.py")


if __name__ == "__main__":
    raise SystemExit(main())
