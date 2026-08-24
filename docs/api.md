# API reference

FastAPI service exposing the trained models. Interactive docs at
[`/docs`](http://localhost:8000/docs); the OpenAPI schema at `/openapi.json`.

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

All routes are served under the prefix `/api/v1` (set by `api.prefix` in
`configs/config.yaml`) and also, unprefixed and hidden from the schema, so
container health checks and Prometheus scrapers do not need to know the prefix.

The static dashboard is mounted at `/`. It is mounted last, after every API
route, so the mount only catches paths no route claimed — `/docs`,
`/openapi.json`, `/health` and `/api/...` are unaffected. Service metadata that
used to live at `/` is now at `/api/info`.

## Startup behaviour

The default model loads during startup; everything else loads on first request.
A missing checkpoint is **not** fatal — the process starts, `/health` reports
`degraded`, and prediction routes return 503 with the command that fixes it. A
container that reports its own problem beats one that crash-loops.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | The operations dashboard (static files) |
| GET | `/api/info` | Service metadata and endpoint index |
| GET | `/health` | Liveness, loaded models, uptime |
| GET | `/models` | Metadata for every model on disk |
| GET | `/leaderboard` | Offline evaluation results |
| GET | `/metrics` | Prometheus exposition |
| POST | `/predict` | RUL for one machine |
| POST | `/predict/batch` | RUL for many machines |
| GET | `/fleet` | Score the reference dataset, most urgent first |
| GET | `/fleet/{engine_id}` | One machine's prediction plus sensor history |
| POST | `/monitoring/drift` | Compare live windows against the training distribution |

## POST `/predict`

Send a machine's recent sensor history; get back a RUL and the maintenance
decision that follows from it.

Two input forms — supply **exactly one**:

**`window`** — a numeric matrix, oldest cycle first, columns in the model's
feature order (`GET /models` returns `feature_columns`):

```json
{
  "engine_id": 42,
  "model": "gru",
  "window": [[641.8, 1589.7, 1400.6, "…15 values…"], "…30 rows…"]
}
```

**`readings`** — one dictionary per cycle, keyed by sensor name. The predictor
selects and orders the columns itself, so you can send all 21 sensors and let it
pick the 15 it was trained on:

```json
{
  "engine_id": "pump-3",
  "readings": [{"sensor_2": 641.82, "sensor_3": 1589.70, "…": 0}]
}
```

Response:

```json
{
  "engine_id": 42,
  "model": "gru",
  "rul_cycles": 87.4,
  "health_score": 69.9,
  "risk_level": "watch",
  "recommended_action": "Increase monitoring frequency and review sensor trends.",
  "requires_action": false,
  "window_size": 30,
  "unit": "cycles",
  "timestamp": "2026-08-24T12:00:00Z",
  "attention": null
}
```

Query parameters:

- `explain=true` — include per-cycle attention weights (attention model only;
  `null` for others).

Notes:

- `scaled` defaults to `false`. Leave it false for raw readings — the service
  applies the saved scaler. Set it true only for values already standardised with
  this project's scaler (e.g. arrays from `data/processed`).
- The window must hold **exactly** `window_size` cycles (30 by default).
- Predictions are clipped at 0; negative remaining life is meaningless.

## POST `/predict/batch`

Up to 256 machines per call. Items are grouped by model so each model's windows
run as one batched forward pass.

```json
{
  "model": "gru",
  "items": [
    {"engine_id": 1, "window": [["…"]]},
    {"engine_id": 2, "readings": [{"sensor_2": 641.8}]}
  ]
}
```

The response adds a fleet rollup:

```json
{
  "count": 2,
  "predictions": ["…"],
  "risk_summary": {"critical": 1, "watch": 1},
  "action_required": 1
}
```

One invalid item fails the whole request — partial fleet results are more
dangerous than an explicit error.

## GET `/fleet`

Scores every machine in `data/processed/test_<dataset>_processed.csv` from its most
recent window and returns them most urgent first. This is what the dashboard
reads.

Query parameters: `model`, `limit` (1-1000, default 200).

```json
{
  "model": "gru",
  "count": 100,
  "risk_summary": {"critical": 8, "warning": 21, "watch": 30, "healthy": 41},
  "action_required": 29,
  "fleet_health": 71.4,
  "median_rul": 92.5,
  "mae": 18.3,
  "source": "test_FD001",
  "engines": [
    {
      "engine_id": 17,
      "rul_cycles": 11.2,
      "health_score": 9.0,
      "risk_level": "critical",
      "recommended_action": "Take out of service and inspect immediately.",
      "cycles_observed": 198,
      "actual_rul": 14.0
    }
  ]
}
```

Returns 503 when the data has not been prepared, naming the script to run.

## GET `/fleet/{engine_id}`

One machine's prediction plus the raw sensor history behind it — what the
dashboard's detail panel and sparkline plot. Query parameters: `model`,
`history` (2-500 cycles, default 60). Includes `attention` when the attention
model is serving.

## GET `/health`

```json
{
  "status": "ok",
  "version": "0.1.0",
  "models_available": ["gru", "random_forest"],
  "models_loaded": ["gru"],
  "default_model": "gru",
  "scaler_available": true,
  "uptime_seconds": 142.7
}
```

`status` is `degraded` when no checkpoint is on disk. The endpoint still returns
200, so it works as a liveness probe.

## GET `/models`

Per model: type, path, window size, feature columns, parameter count, recorded
validation metrics, and whether it is currently loaded. Models not yet loaded
report their file size instead of full metadata, so the call never forces a
180 MB read.

## POST `/monitoring/drift`

Compares a batch of live windows against the training distribution using PSI and
a two-sample KS test per feature. See [`monitoring.md`](monitoring.md) for the
thresholds and what the verdict means.

```json
{ "windows": [[["…15 values…"], "…30 cycles…"]], "scaled": false }
```

```json
{
  "status": "warning",
  "n_samples": 300,
  "n_features": 15,
  "drifted_features": ["sensor_11", "sensor_14"],
  "feature_share": 0.133,
  "details": ["…per-feature PSI, KS statistic, p-value, mean shift…"]
}
```

## GET `/metrics`

Prometheus text format:

| Metric | Type | Labels |
|---|---|---|
| `prognostix_predictions_total` | counter | `model`, `risk_level` |
| `prognostix_prediction_errors_total` | counter | `reason` |
| `prognostix_prediction_duration_seconds` | histogram | `model` |
| `prognostix_last_predicted_rul_cycles` | gauge | `model` |

## Errors

Every failure returns the same body shape:

```json
{ "detail": "…", "error_type": "validation_error", "timestamp": "…" }
```

| Status | Meaning |
|---|---|
| 400 | Unknown model name, unknown metric |
| 404 | Engine not in the dataset |
| 422 | Schema violation, wrong window shape, missing sensors, non-finite values |
| 503 | Model or prepared data missing — the message names the command to run |
| 500 | Unhandled error (logged with a stack trace server-side) |

## Client examples

```bash
# health
curl localhost:8000/api/v1/health

# fleet rollup
curl "localhost:8000/api/v1/fleet?limit=5"

# one prediction from a synthetic window
python - <<'PY'
import json, urllib.request
window = [[500.0 + i * 0.1] * 15 for i in range(30)]
request = urllib.request.Request(
    "http://localhost:8000/api/v1/predict",
    data=json.dumps({"engine_id": 1, "window": window}).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(request)))
PY
```

## CORS

`api.cors_origins` defaults to `*` for local development. Narrow it to the
dashboard's origin before exposing the service.
