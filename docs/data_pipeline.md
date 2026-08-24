# Data pipeline

From raw CMAPSS text files to windowed arrays a model can train on.

```bash
python scripts/download_data.py    # only if data/raw/CMAPSS is empty
python scripts/prepare_data.py
```

## The raw data

`data/raw/CMAPSS/` holds four sub-datasets. This project defaults to **FD001**:
100 training engines, 100 test engines, one operating condition, one fault mode
(HPC degradation).

Each file is whitespace-separated with no header, 26 columns:

| Columns | Meaning |
|---|---|
| 1 | `engine_id` — machine identifier |
| 2 | `cycle` — time index, 1-based |
| 3-5 | `setting_1..3` — operational settings |
| 6-26 | `sensor_1..21` — sensor measurements |

`train_FD001.txt` runs every engine to failure. `test_FD001.txt` stops each
engine some cycles *before* failure, and `RUL_FD001.txt` gives the true remaining
life at that stopping point — one value per test engine, in engine order.

Column names come from `configs/config.yaml`, so the whole project uses
`engine_id` / `cycle` / `setting_*` rather than the two different naming schemes
the exploratory notebooks used.

## Stage 1 — load and validate

`validate_raw_frame` refuses to let bad data reach a model. It checks the schema,
nulls, non-finite values, duplicate `(engine, cycle)` pairs, per-engine cycle
monotonicity, and reports engines too short to window plus any constant sensors.
Errors abort the run; warnings are logged (`--strict` promotes them to errors).

## Stage 2 — the RUL target

For training data, RUL is linear in cycles:

```
RUL = max(cycle) per engine - cycle
```

so the target counts down to 0 at failure.

For the **test** frame the series is truncated, so `add_rul(..., final_rul=...)`
offsets each engine by its ground-truth remaining life. Without that offset the
last recorded cycle of a test engine would be labelled 0, which is wrong — it had
`RUL_FD001.txt[engine]` cycles left.

### RUL capping

`data.rul_cap` optionally clips the target (125 is the common choice in the
literature). The rationale: an engine at cycle 5 of a 300-cycle life shows no
observable degradation, so training the model to output 295 teaches it to
predict from position rather than from condition.

The default is `null`, which reproduces the arrays and model artifacts committed
in this repository. Set `--rul-cap 125` to train the capped variant — it usually
lowers RMSE, and it changes what the number means, so it is an explicit choice
rather than a hidden default.

## Stage 3 — cleaning

Six FD001 sensors never move: `sensor_1, 5, 10, 16, 18, 19`. They carry no
information, break variance-based scaling, and widen the model input for nothing.
`select_feature_columns` drops them, leaving **15 informative sensors** in a fixed
order — that order is the feature axis of every array downstream, so it is
recorded in the scaler bundle and in each checkpoint.

Also applied here: sorting by `(engine_id, cycle)`, dropping duplicate cycles, and
the optional RUL cap.

## Stage 4 — split by engine

```
100 engines → 70 train / 15 validation / 15 test
```

The split is over **machine IDs**, not rows:

```python
train, held_out = train_test_split(engine_ids, test_size=0.30, random_state=42)
val, test       = train_test_split(held_out,   test_size=0.50, random_state=42)
```

Cycle *n* and cycle *n+1* of the same engine are nearly identical readings. A
random row split therefore trains on the neighbours of everything it is validated
on, and the resulting scores are fiction. `prepare_data.py` asserts the three
engine sets are disjoint and fails loudly if they are not.

## Stage 5 — scaling

A `StandardScaler` is fitted on **training engines only**, then applied to all
three splits. Fitting on the full frame would leak validation statistics into
training.

The fitted scaler is saved to `artifacts/scaler.joblib` as a `ScalerBundle`
carrying:

- the fitted scaler
- `feature_columns` — the 15 sensors, in order
- `kind`, `window_size`, `rul_cap`
- metadata (dataset, engine counts per split)

This is what makes the API possible: raw readings from a live machine must pass
through exactly these statistics.

## Stage 6 — windowing

Models see a fixed-length window, not the whole history:

```python
for engine in frame.groupby("engine_id"):
    for i in range(window_size, len(engine)):
        X.append(features[i - window_size : i])   # cycles [i-30, i)
        y.append(targets[i])                      # RUL at cycle i
```

Windows never cross an engine boundary. Window `i` spans the 30 cycles before `i`
and is labelled with the RUL at cycle `i` — the model predicts the RUL of the
cycle immediately following the window it observed. Since RUL decreases by exactly
one per cycle this is a constant one-cycle offset from "RUL at the window's last
cycle"; it is kept because it is what produced the committed arrays.

Engines shorter than `window_size + 1` cycles contribute nothing and are
reported as a warning.

With FD001 defaults:

| Split | Windows | Shape |
|---|---|---|
| train | 12,407 | (12407, 30, 15) |
| val | 2,612 | (2612, 30, 15) |
| test | 2,612 | (2612, 30, 15) |

## Outputs

| Path | Contents |
|---|---|
| `data/processed/{train,val,test}_sequences.npz` | `X`, `y`, `engine_ids` |
| `data/processed/train_FD001_processed.csv` | Cycle-level frame with RUL |
| `data/processed/test_FD001_processed.csv` | Test frame with ground-truth-offset RUL |
| `data/processed/RUL_FD001_processed.csv` | Ground-truth RUL per test engine |
| `artifacts/scaler.joblib` | The `ScalerBundle` |
| `artifacts/reports/dataset_summary.json` | Split sizes, RUL ranges, feature list |

## Inference-time windowing

`last_window_per_engine` takes one window per machine — its most recent
`window_size` cycles. That is both the official CMAPSS test protocol (one
prediction per engine, scored against `RUL_FD001.txt`) and the shape a live
deployment sends. Machines with fewer than `window_size` cycles are front-padded
with their earliest reading, or skipped with `pad=False`.

## Feature engineering for tabular models

Tree and linear models cannot read a matrix, so `create_statistical_features`
flattens each window into per-sensor aggregations — `mean, std, min, max, last,
trend` — giving `15 × 6 = 90` features. **The block order is part of the saved
model's input contract** and must not be reordered.

`build_tabular_frame` offers the row-wise alternative: rolling means and standard
deviations, lags and per-cycle differences, all computed per engine so no history
leaks between machines.

## Reproducing or changing the pipeline

```bash
python scripts/prepare_data.py                        # defaults; reproduces the committed arrays
python scripts/prepare_data.py --window-size 50       # longer context
python scripts/prepare_data.py --rul-cap 125          # capped target
python scripts/prepare_data.py --dataset FD003        # a different sub-dataset
python scripts/prepare_data.py --strict               # warnings become errors
```

Changing the window size or the feature set invalidates existing checkpoints —
the input width changes. Retrain after any such change.
