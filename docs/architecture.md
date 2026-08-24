# Architecture

Prognostix ML predicts the **Remaining Useful Life (RUL)** of industrial equipment
from multivariate sensor histories. The reference dataset is NASA's CMAPSS
turbofan degradation simulation.

## The shape of the problem

Each machine emits one row per operating cycle: 3 operational settings and 21
sensor readings. In training data every machine runs to failure, so RUL at any
cycle is `last_cycle - current_cycle`. At serving time a machine has not failed
yet, and the question is how many cycles remain.

This makes it a **sequence regression** problem, not a classification or
tabular one: the signal lives in how readings move over time, not in any single
snapshot.

## Layers

```
data/raw/CMAPSS/*.txt
        │
        ▼
src/ingestion         load raw files, attach the RUL target, validate schema
        │
        ▼
src/preprocessing     drop dead sensors, split by engine, fit + save the scaler,
        │             slide fixed windows over each machine's history
        ▼
data/processed/*.npz  (n_windows, window_size, n_features) + targets
        │
        ├──▶ src/features        flatten windows for tabular models; rolling / lag columns
        │
        ▼
src/models            naive · linear · random forest · LSTM · GRU · attention
        │
        ▼
artifacts/models/     checkpoints + the scaler + the leaderboard
        │
        ├──▶ src/evaluation      metrics, figures, model comparison
        │
        ▼
src/inference         RULPredictor: raw sensors → scaled window → RUL → health band
        │
        ├──▶ api/                FastAPI service (/predict, /fleet, /monitoring/drift)
        │      └──▶ frontend/    static operations dashboard
        │
        └──▶ monitoring/         drift · performance · alerts
```

Nothing above `src/` contains prediction logic. The API, the CLI scripts and the
notebooks all call the same functions, so they cannot disagree about how a RUL is
produced.

## Modules

| Path | Responsibility |
|---|---|
| `src/config.py` | Loads `configs/config.yaml`, resolves paths, seeds RNGs, picks the device |
| `src/ingestion/loader.py` | Reads raw CMAPSS files, attaches RUL, loads/saves `.npz` splits |
| `src/ingestion/validator.py` | Schema, null, duplicate, monotonicity and shape checks |
| `src/preprocessing/cleaning.py` | Drops zero-variance sensors, caps RUL, dedupes cycles |
| `src/preprocessing/scaling.py` | Fits and **persists** the scaler as a `ScalerBundle` |
| `src/preprocessing/sequences.py` | Engine-level splits and sliding-window construction |
| `src/features/` | Window aggregations for tabular models; rolling / lag / diff columns |
| `src/models/common.py` | Shared training loop, early stopping, checkpoint I/O |
| `src/models/runner.py` | One training protocol used by all three sequence models |
| `src/evaluation/` | Metrics (including the asymmetric NASA score), figures, leaderboard |
| `src/inference/predictor.py` | Loads a model + scaler, owns the serving contract |
| `src/inference/health_score.py` | RUL → health score → risk band → recommended action |
| `api/` | FastAPI app, routes and Pydantic schemas |
| `monitoring/` | PSI/KS drift detection, performance tracking, alert routing |
| `scripts/` | CLI entry points for the whole lifecycle |
| `frontend/` | Static dashboard (vanilla JS + inline SVG, no build step) |

## Design decisions worth knowing

**Split by engine, never by row.** Consecutive cycles of one machine are nearly
identical. A random row split puts cycle 100 in training and cycle 101 in
validation, which leaks the answer and produces scores that collapse in
production. `split_engines` partitions machine IDs; `scripts/prepare_data.py`
asserts the splits are disjoint.

**The scaler is an artifact, not a step.** It is fitted on training engines only
and saved to `artifacts/scaler.joblib` together with the feature order, window
size and RUL cap. A model trained on standardised inputs and served raw readings
returns plausible numbers that are simply wrong; the `ScalerBundle` makes that
mistake impossible to make silently.

**Feature order is part of the contract.** `select_feature_columns` returns
sensors in a fixed order, and that order is stored in both the scaler bundle and
every checkpoint. The API can therefore accept named readings and put the columns
in the right places itself.

**Checkpoints carry their own metadata.** Each `.pt` file stores the constructor
arguments, feature names, window size, training metrics and history — so serving
code rebuilds the exact architecture without a config file, and the notebook-era
checkpoints (which only carry the constructor arguments) still load.

**One training protocol.** All three sequence models run through
`src/models/runner.py`: same splits, same seed, same optimiser, same metrics. A
leaderboard difference is therefore an architecture difference.

**Errors are actionable.** A missing artifact says which command produces it. The
API starts without a trained model, reports `degraded` on `/health`, and returns
503 with the training command rather than crash-looping.

## Configuration

Every module reads `configs/config.yaml` through `load_config()`. Override the
path with `PROGNOSTIX_CONFIG`. Nothing is hardcoded twice — window size, feature
selection, thresholds and paths all live in that one file.

## Further reading

- [`data_pipeline.md`](data_pipeline.md) — how raw files become model inputs
- [`modeling.md`](modeling.md) — architectures, training and results
- [`api.md`](api.md) — endpoint reference
- [`monitoring.md`](monitoring.md) — drift, performance tracking and alerts
