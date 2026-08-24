
# Prognostix ML

Predictive maintenance for industrial equipment: predict the **Remaining Useful
Life (RUL)** of a machine from its sensor history, turn that number into a
maintenance decision, and keep watching for the day the model stops being valid.

Built on NASA's CMAPSS turbofan degradation dataset, with the full lifecycle in
one repository — data pipeline, six models, an inference API, an operations
dashboard, and production monitoring.

```
data → validate → window → train → evaluate → serve → monitor → retrain
```

## Contents

- [Quick start](#quick-start)
- [What's inside](#whats-inside)
- [Usage](#usage)
- [Results](#results)
- [The API](#the-api)
- [The dashboard](#the-dashboard)
- [Monitoring](#monitoring)
- [Docker](#docker)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Documentation](#documentation)

## Quick start

```bash
git clone https://github.com/HasnatKhan010/prognostix-ml.git
cd prognostix-ml

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# CPU-only torch (the default wheels pull in ~2 GB of CUDA libraries)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python scripts/download_data.py     # skip if data/raw/CMAPSS is populated
python scripts/prepare_data.py      # → data/processed/*.npz + artifacts/scaler.joblib
python scripts/train.py --model gru # → artifacts/models/gru.pt
python scripts/evaluate.py          # → leaderboard + figures

uvicorn api.main:app --reload --port 8000
```

Then open <http://localhost:8000> for the dashboard, or
<http://localhost:8000/docs> for the API.

## What's inside

| Layer | What it does |
|---|---|
| **Data pipeline** (`src/ingestion`, `src/preprocessing`) | Loads raw CMAPSS files, validates the schema, attaches the RUL target, drops dead sensors, splits **by engine**, fits and **saves** the scaler, slides fixed windows |
| **Features** (`src/features`) | Window aggregations for tabular models; rolling, lag and difference columns computed per machine |
| **Models** (`src/models`) | Naive mean · linear regression · random forest · LSTM · GRU · LSTM+attention, all trained through one protocol |
| **Evaluation** (`src/evaluation`) | MAE / RMSE / R² / MAPE / bias plus the asymmetric NASA prognostics score; diagnostic figures; a merged leaderboard |
| **Inference** (`src/inference`) | `RULPredictor` owns the serving contract: feature order, scaling, window length, health banding |
| **API** (`api/`) | FastAPI: `/predict`, `/predict/batch`, `/fleet`, `/monitoring/drift`, `/metrics` |
| **Dashboard** (`frontend/`) | Static operations view — fleet risk tiles, RUL ranking, health meter, sensor trends. No build step |
| **Monitoring** (`monitoring/`) | PSI + KS drift detection, performance-degradation tracking, alert routing |

### Three decisions that make it correct

**Split by engine, never by row.** Cycle *n* and cycle *n+1* of one machine are
nearly identical readings. A random row split trains on the neighbours of
everything it validates on; the score looks great and collapses in production.
`prepare_data.py` partitions machine IDs and asserts the splits are disjoint.

**The scaler is a saved artifact.** Fitted on training engines only, then written
to `artifacts/scaler.joblib` along with the feature order, window size and RUL
cap. A model trained on standardised inputs but served raw readings returns
plausible numbers that are simply wrong — the `ScalerBundle` makes that mistake
impossible to make silently.

**Optimistic errors cost more than conservative ones.** Predicting *more* life
than a machine has left means it fails in service. RMSE cannot see that
asymmetry, so every evaluation also reports `Bias` and the NASA/PHM08 score,
which penalises late predictions on a steeper curve.

## Usage

### Prepare data

```bash
python scripts/prepare_data.py                   # FD001, 30-cycle window
python scripts/prepare_data.py --rul-cap 125     # cap the target (common in the literature)
python scripts/prepare_data.py --window-size 50 --dataset FD003
```

### Train

```bash
python scripts/train.py --model gru
python scripts/train.py --model all              # every model, one leaderboard
python scripts/train.py --model lstm --epochs 60 --lr 5e-4 --device cuda
python scripts/train.py --model baselines        # naive, linear, random forest
```

Models: `mean`, `linear`, `random_forest`, `lstm`, `gru`, `attention`.

### Evaluate

```bash
python scripts/evaluate.py                       # every trained model, test split
python scripts/evaluate.py --split val
python scripts/evaluate.py --official            # CMAPSS competition protocol
```

### Predict

```bash
python scripts/predict.py                        # score the fleet, print a worklist
python scripts/predict.py --alerts               # raise alerts for at-risk machines
python scripts/predict.py --input data/live.csv --model lstm
```

```
====================================================================
MAINTENANCE WORKLIST - model: gru | 100 machine(s)
====================================================================
critical: 8  warning: 21  watch: 30  healthy: 41

  engine       RUL   health  risk         actual    error
--------------------------------------------------------------
      17      11.2      9.0  critical       14.0     -2.8
      64      18.7     15.0  critical       21.0     -2.3
```

### Use it from Python

```python
from src.inference.predictor import RULPredictor

predictor = RULPredictor(model_name="gru")

assessment = predictor.assess(window)          # (30, 15) raw sensor readings
print(assessment.rul, assessment.risk_level.value, assessment.recommended_action)

# or score a whole cycle-level frame, one row per machine
worklist = predictor.predict_frame(frame)
```

## Results

FD001, validation split, uncapped RUL — the committed baselines:

| Model | MAE | RMSE |
|---|---|---|
| Linear regression | 25.59 | 32.81 |
| Random forest | 26.00 | 37.88 |
| Naive mean | 48.63 | 57.81 |

The naive mean is the floor every model must clear: it ignores the sensors
entirely and still reaches ~58 RMSE, because RUL has a strong prior. Note also
that the random forest beats linear regression on MAE while losing on RMSE — it
is usually closer but occasionally badly wrong, which is exactly the wrong shape
for maintenance.

Run `python scripts/train.py --model all` to fill in the sequence models on your
hardware; results land in `artifacts/model_comparison.csv` and figures in
`artifacts/figures/`.

## The API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI at `/docs`. Routes are served under `/api/v1` and, unprefixed, for
health checks and metric scrapers.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | The operations dashboard |
| GET | `/api/info` | Service metadata and endpoint index |
| GET | `/health` | Liveness, loaded models, uptime |
| GET | `/models` | Metadata for every model on disk |
| POST | `/predict` | RUL for one machine |
| POST | `/predict/batch` | Up to 256 machines in one call |
| GET | `/fleet` | Score the reference dataset, most urgent first |
| GET | `/fleet/{id}` | One machine plus its sensor history |
| POST | `/monitoring/drift` | Compare live windows against the training distribution |
| GET | `/leaderboard` | Offline evaluation results |
| GET | `/metrics` | Prometheus exposition |

```bash
curl -X POST localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"engine_id": 42, "readings": [{"sensor_2": 641.82, "sensor_3": 1589.7}]}'
```

```json
{
  "engine_id": 42,
  "model": "gru",
  "rul_cycles": 87.4,
  "health_score": 69.9,
  "risk_level": "watch",
  "recommended_action": "Increase monitoring frequency and review sensor trends.",
  "requires_action": false,
  "window_size": 30
}
```

Send `window` (a numeric matrix in the model's feature order) or `readings`
(dictionaries keyed by sensor name — the service selects and orders the columns
itself). Full reference: [`docs/api.md`](docs/api.md).

The API starts even with no trained model: `/health` reports `degraded` and
prediction routes return 503 naming the command to run, rather than crash-looping.

## The dashboard

`frontend/` is a static operations view — vanilla JS and inline SVG, no build
step, no dependencies. The API serves it at `/`, so no second web server is
needed:

```bash
uvicorn api.main:app --port 8000     # then open http://localhost:8000
```

It shows:

- **Fleet health index** and per-band machine counts
- **RUL ranking** by machine, coloured by risk band, most urgent first
- **Health meter** and **sensor trend** for the selected machine
- Table view for every chart, keyboard-navigable bars, light/dark themes

Every risk band ships an icon and a label alongside its colour, and each chart has
a table-view twin, so nothing is encoded by colour alone.

The header's API field only matters when the dashboard is served from a different
origin than the API (it defaults to the current origin).

## Monitoring

```bash
python scripts/predict.py --alerts        # score the fleet, alert on at-risk machines
python -m monitoring.drift                # input drift vs the training distribution
python -m monitoring.performance          # error vs the training baseline
```

Ground truth for RUL only arrives when a machine fails, so waiting for accuracy
metrics means waiting for the failures the model was meant to prevent. Drift
detection watches the **inputs** instead — PSI plus a KS test per sensor, flagged
only when the shift is both large and statistically significant. Performance
tracking closes the loop later, comparing error against the baseline recorded at
training and appending every run to `artifacts/reports/performance_log.csv`.

Both raise alerts through one router (JSONL log, console, optional webhook).
Details and thresholds: [`docs/monitoring.md`](docs/monitoring.md).

### Health bands

| Band | RUL (cycles) | Action |
|---|---|---|
| `critical` | ≤ 20 | Take out of service and inspect immediately |
| `warning` | ≤ 50 | Schedule maintenance in the next planned window |
| `watch` | ≤ 80 | Increase monitoring, review sensor trends |
| `healthy` | > 80 | No action required |

## Docker

### Hugging Face Spaces (single container)

The root `Dockerfile` builds one container that serves the dashboard at `/` and
the API under `/api/v1`, listening on port 7860 as Spaces requires:

```bash
docker build -t prognostix-space .
docker run --rm -p 7860:7860 prognostix-space
```

Open <http://localhost:7860>. To deploy, push this repository to a Space with
`sdk: docker` — the YAML front matter at the top of this README already declares
`app_port: 7860`.

Trained models are not tracked in git, so the build runs
`scripts/prepare_data.py` and trains a GRU from the committed raw CMAPSS data.
That adds a few minutes to the first build and makes the image self-sufficient.
If a `.pt` checkpoint is present in the build context, training is skipped.

### Local development stack (separate services)

```bash
cd docker
docker compose up --build
```

Dashboard on <http://localhost:8080>, API on <http://localhost:8000>, with nginx
reverse-proxying `/api`. Prepared data and trained models are bind-mounted rather
than baked in, so run `prepare_data.py` and `train.py` on the host first.

## Testing

```bash
pytest                                    # whole suite
pytest tests/test_preprocessing.py -v
pytest --cov=src --cov=api --cov=monitoring
```

The suite is self-contained: fixtures build a synthetic CMAPSS-shaped dataset,
scaler and checkpoint inside `tmp_path`, so it runs on a clean checkout with no
prepared data or trained model, and never touches the committed `data/` or
`artifacts/` trees.

What it covers: engine-split disjointness, scaler leakage, window content and
engine boundaries, the flattened-feature block order, model shapes and gradients,
checkpoint round-trips, early stopping, metric correctness (including the NASA
score's asymmetry), the whole HTTP surface, and drift / performance / alert logic.

## Configuration

Everything reads `configs/config.yaml` — paths, dataset, window size, splits,
model hyperparameters, training settings, health thresholds, drift thresholds:

```yaml
data:
  dataset: FD001
  window_size: 30
  rul_cap: null           # 125 is the common choice; null reproduces the committed artifacts
  split: { test_size: 0.30, val_ratio: 0.50, random_state: 42 }

models:
  gru: { hidden_size: 128, num_layers: 2, dropout: 0.2 }

inference:
  default_model: gru
  health: { max_rul: 125, critical_rul: 20, warning_rul: 50, watch_rul: 80 }
```

Point `PROGNOSTIX_CONFIG` at another file to override it wholesale. See
`.env.example` for environment-level settings.

## Project layout

```
├── api/                 FastAPI app: main, routes, schemas
├── configs/             config.yaml — the single source of settings
├── data/
│   ├── raw/CMAPSS/      original NASA text files
│   └── processed/       windowed .npz splits + cycle-level CSVs
├── artifacts/
│   ├── models/          checkpoints (.pt, .joblib)
│   ├── figures/         evaluation plots
│   ├── reports/         drift, performance log, alerts, dataset summary
│   └── scaler.joblib    the fitted ScalerBundle
├── docker/              Dockerfile.api, Dockerfile.frontend, docker-compose.yml
├── docs/                architecture, data pipeline, modeling, api, monitoring
├── frontend/            static dashboard (index.html, app.js, styles.css)
├── monitoring/          drift.py, performance.py, alerts.py
├── notebooks/           01-06: exploration → forensics → preprocessing → models
├── scripts/             download_data, prepare_data, train, evaluate, predict
├── src/
│   ├── config.py        config loading, paths, seeding, device
│   ├── ingestion/       loader, validator
│   ├── preprocessing/   cleaning, scaling, sequences
│   ├── features/        engineering, rolling, lag
│   ├── models/          baseline/, lstm/, gru/, attention/, common, runner
│   ├── evaluation/      metrics, plots, compare
│   └── inference/       predictor, health_score
└── tests/               preprocessing, features, models, inference, api, monitoring
```

The `notebooks/` directory holds the original exploratory work. It used two
different column-naming schemes (`unit_number`/`time_cycles` in notebook 01,
`engine_id`/`cycle` in 02-06); the package standardises on the latter via
`configs/config.yaml`.

## Documentation

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Layers, modules, design decisions |
| [`docs/data_pipeline.md`](docs/data_pipeline.md) | Raw files → windowed arrays, stage by stage |
| [`docs/modeling.md`](docs/modeling.md) | Architectures, training protocol, metrics, results |
| [`docs/api.md`](docs/api.md) | Endpoint reference with request/response examples |
| [`docs/monitoring.md`](docs/monitoring.md) | Drift, performance tracking, alerts, retraining triggers |

## Dataset

NASA CMAPSS Turbofan Engine Degradation Simulation Data Set (FD001–FD004).
A. Saxena, K. Goebel, D. Simon, N. Eklund, *"Damage Propagation Modeling for
Aircraft Engine Run-to-Failure Simulation"*, PHM 2008.

## License

MIT
