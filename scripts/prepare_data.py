#!/usr/bin/env python
"""Build the modelling dataset from the raw CMAPSS files.

Pipeline::

    raw txt -> validate -> RUL target -> drop constant sensors
            -> split by engine -> fit scaler on train only
            -> sliding windows -> save npz + scaler + processed csv

Two properties matter more than anything else here:

* **The split is by engine.** Consecutive cycles of one machine are nearly
  identical, so splitting rows at random leaks the answer and inflates scores.
* **The scaler is fitted on the training engines only** and then saved. A scaler
  fitted on everything leaks validation statistics into training, and a scaler
  that is thrown away leaves the API unable to reproduce the model's inputs.

With default settings this reproduces the arrays committed under
``data/processed/``: 12407 train / 2612 val / 2612 test windows of shape (30, 15).

Usage::

    python scripts/prepare_data.py
    python scripts/prepare_data.py --window-size 50 --rul-cap 125
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, set_seed, setup_logging  # noqa: E402
from src.ingestion.loader import add_rul, load_raw_split, load_rul_truth  # noqa: E402
from src.ingestion.loader import save_sequences  # noqa: E402
from src.ingestion.validator import validate_raw_frame, validate_sequences  # noqa: E402
from src.preprocessing.cleaning import prepare_frame  # noqa: E402
from src.preprocessing.scaling import apply_scaler, fit_scaler, save_scaler  # noqa: E402
from src.preprocessing.sequences import (  # noqa: E402
    create_sequences,
    split_by_engine,
    split_engines,
)

logger = logging.getLogger("prognostix.prepare")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="CMAPSS subset (default: FD001).")
    parser.add_argument("--window-size", type=int, default=None, help="Cycles per window.")
    parser.add_argument(
        "--rul-cap",
        type=float,
        default=None,
        help="Clip RUL at this many cycles (125 is the common choice).",
    )
    parser.add_argument("--scaler", default=None, choices=["standard", "minmax"])
    parser.add_argument(
        "--no-csv", action="store_true", help="Skip writing the cycle-level CSVs."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on any validation warning, not just errors.",
    )
    args = parser.parse_args(argv)

    setup_logging()
    config = get_config()
    data = config.data
    set_seed(int(config.project.seed))

    dataset = args.dataset or str(data.dataset)
    window_size = int(args.window_size or data.window_size)
    rul_cap = args.rul_cap if args.rul_cap is not None else data.get("rul_cap")
    scaler_kind = args.scaler or str(config.preprocessing.scaler)

    logger.info(
        "Preparing %s | window=%d | rul_cap=%s | scaler=%s",
        dataset,
        window_size,
        rul_cap if rul_cap is not None else "none",
        scaler_kind,
    )

    # --- 1. load and validate ----------------------------------------
    frame = load_raw_split("train", dataset, config)
    report = validate_raw_frame(frame, config).log(logger)
    report.raise_for_errors()
    if args.strict and report.warnings:
        logger.error("Aborting: --strict and %d warning(s)", len(report.warnings))
        return 1

    # --- 2. target and cleaning ---------------------------------------
    frame = add_rul(
        frame,
        id_column=data.id_column,
        time_column=data.time_column,
        target_column=data.target_column,
        cap=rul_cap,
    )
    frame, feature_columns = prepare_frame(frame, config, cap=rul_cap)
    logger.info("Feature columns (%d): %s", len(feature_columns), feature_columns)

    # --- 3. split by engine -------------------------------------------
    engine_ids = frame[data.id_column].unique()
    split_config = data.split
    train_engines, val_engines, test_engines = split_engines(
        engine_ids,
        test_size=float(split_config.test_size),
        val_ratio=float(split_config.val_ratio),
        random_state=int(split_config.random_state),
    )

    splits = {
        "train": split_by_engine(frame, train_engines, data.id_column),
        "val": split_by_engine(frame, val_engines, data.id_column),
        "test": split_by_engine(frame, test_engines, data.id_column),
    }
    _assert_disjoint(splits, data.id_column)

    # --- 4. scale (fitted on train only) ------------------------------
    bundle = fit_scaler(
        splits["train"],
        feature_columns,
        kind=scaler_kind,
        window_size=window_size,
        rul_cap=rul_cap,
        metadata={
            "dataset": dataset,
            "n_train_engines": int(len(train_engines)),
            "n_val_engines": int(len(val_engines)),
            "n_test_engines": int(len(test_engines)),
        },
    )
    scaled = {name: apply_scaler(part, bundle) for name, part in splits.items()}
    save_scaler(bundle, config=config)

    # --- 5. windowing --------------------------------------------------
    summary: dict[str, object] = {
        "dataset": dataset,
        "window_size": window_size,
        "rul_cap": rul_cap,
        "scaler": scaler_kind,
        "feature_columns": feature_columns,
        "splits": {},
    }

    for name, part in scaled.items():
        X, y, engine_index = create_sequences(
            part,
            feature_columns=feature_columns,
            target_column=data.target_column,
            window_size=window_size,
            id_column=data.id_column,
            time_column=data.time_column,
            return_ids=True,
        )
        validate_sequences(
            X, y, window_size=window_size, n_features=len(feature_columns), config=config
        ).log(logger).raise_for_errors()

        save_sequences(name, X, y, config=config, engine_ids=engine_index)
        summary["splits"][name] = {
            "n_windows": int(len(X)),
            "n_engines": int(len(np.unique(engine_index))),
            "rul_min": float(y.min()),
            "rul_max": float(y.max()),
            "rul_mean": round(float(y.mean()), 3),
        }
        logger.info(
            "%-5s -> X=%s y=%s over %d engine(s)",
            name,
            X.shape,
            y.shape,
            len(np.unique(engine_index)),
        )

    # --- 6. cycle-level CSVs and the official test set -----------------
    if not args.no_csv:
        _write_csvs(frame, dataset, config)

    summary_path = config.path("reports") / "dataset_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", summary_path)

    print("\nPrepared:")
    for name, stats in summary["splits"].items():
        print(
            f"  {name:<5} {stats['n_windows']:>6} windows | "
            f"{stats['n_engines']:>3} engines | RUL {stats['rul_min']:.0f}-{stats['rul_max']:.0f}"
        )
    print("\nNext step: python scripts/train.py --model gru")
    return 0


def _assert_disjoint(splits: dict, id_column: str) -> None:
    """Fail loudly if any engine appears in two splits."""
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = set(splits[left][id_column]) & set(splits[right][id_column])
            if overlap:
                raise AssertionError(
                    f"Engine leakage between {left} and {right}: {sorted(overlap)[:10]}"
                )
    logger.info("Verified: no engine appears in more than one split")


def _write_csvs(frame, dataset: str, config) -> None:
    """Write cycle-level CSVs, including the official test set with true RUL.

    Column names follow ``configs/config.yaml`` (``engine_id``, ``cycle``,
    ``setting_*``); earlier notebook exports used ``unit_number`` / ``time_cycles``
    and are overwritten here so the whole project speaks one schema.
    """
    directory = config.path("data_processed")
    directory.mkdir(parents=True, exist_ok=True)
    data = config.data

    train_path = directory / f"train_{dataset}_processed.csv"
    frame.to_csv(train_path, index=False)
    logger.info("Wrote %s (%d rows)", train_path.name, len(frame))

    try:
        test_frame = load_raw_split("test", dataset, config)
        truth = load_rul_truth(dataset, config)
    except FileNotFoundError as exc:
        logger.warning("Skipping test CSV: %s", exc)
        return

    # A test engine's series stops before failure, so its RUL at the last
    # recorded cycle is the value in RUL_<dataset>.txt, not zero.
    final_rul = truth.set_index("engine_id")["RUL"]
    test_frame = add_rul(
        test_frame,
        id_column=data.id_column,
        time_column=data.time_column,
        target_column=data.target_column,
        cap=config.data.get("rul_cap"),
        final_rul=final_rul,
    )

    test_path = directory / f"test_{dataset}_processed.csv"
    test_frame.to_csv(test_path, index=False)
    truth.to_csv(directory / f"RUL_{dataset}_processed.csv", index=False)
    logger.info("Wrote %s (%d rows)", test_path.name, len(test_frame))


if __name__ == "__main__":
    raise SystemExit(main())
