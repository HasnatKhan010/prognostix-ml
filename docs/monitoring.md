# Monitoring

A RUL model degrades quietly. Ground truth only arrives when a machine actually
fails, so waiting for accuracy metrics means waiting for the failures the model
was supposed to prevent. Two loops run instead:

- **Drift** watches the *inputs* — available immediately, no labels needed.
- **Performance** watches the *outputs* — available later, once truth arrives.

Both raise alerts through one router.

## Drift detection

```bash
python -m monitoring.drift                      # test split vs training distribution
python -m monitoring.drift --current val
python -m monitoring.drift --current path/to/live.npz
```

A model is only valid while incoming sensors resemble what it trained on. New
hardware, a recalibrated probe, or a different operating regime shifts the inputs
and accuracy follows — long before anyone can measure it.

Two complementary tests run per feature:

**PSI (Population Stability Index)** measures *how much* a distribution moved.
Bin edges come from the reference quantiles, so bins hold roughly equal mass and
outliers cannot dominate. Empty bins are floored at 1e-6 to keep the logarithm
finite:

```
PSI = Σ (current% - reference%) × ln(current% / reference%)
```

| PSI | Reading |
|---|---|
| < 0.10 | stable |
| 0.10 - 0.25 | moderate shift — worth watching |
| > 0.25 | significant shift |

**KS test** asks whether the shift is statistically significant at all, which
stops a small sample from looking dramatic.

A feature is flagged only when **both** fire: PSI crosses the warning threshold
*and* the KS p-value is below `ks_alpha`. Magnitude without significance is
noise; significance without magnitude is irrelevant.

Fleet verdict:

| Status | Condition |
|---|---|
| `stable` | few or no features flagged |
| `warning` | flagged share ≥ `feature_share_warning` (default 0.2) |
| `critical` | flagged share ≥ `feature_share_critical` (default 0.4), or any flagged feature has PSI ≥ `psi_critical` |

Thresholds live under `monitoring.drift` in `configs/config.yaml`. The CLI writes
`artifacts/reports/drift_report.json`, prints the worst features by PSI, raises an
alert when the status is not `stable`, and exits non-zero — so it can gate a
scheduled job.

The same check is available over HTTP at `POST /api/v1/monitoring/drift`.

## Performance tracking

```bash
python -m monitoring.performance --split test --model gru
```

Once ground truth exists, `track_performance` scores predictions and compares the
error against the reference recorded at training time. The baseline resolves in
order: configured `baseline_rmse` / `baseline_mae`, then the model's leaderboard
row, then the best leaderboard row.

| Status | Condition |
|---|---|
| `ok` | within `degradation_warning_pct` (default 10%) |
| `warning` | ≥ 10% worse than baseline |
| `critical` | ≥ `degradation_critical_pct` (default 25%) worse |

Below `min_samples` (default 50) no verdict is issued — the report says so in its
`note` rather than declaring a model broken on twenty points.

Every run appends a row to `artifacts/reports/performance_log.csv`, so the trend
is visible over time rather than only the latest snapshot.

## Health scoring

`src/inference/health_score.py` turns a RUL into something a planner can act on:

```
health_score = min(RUL / max_rul, 1) × 100
```

Linear in remaining cycles and saturating at `max_rul` (125), because an engine
with 300 cycles left is not meaningfully healthier than one with 130 — the
degradation is not observable in either.

| Risk band | RUL (cycles) | Recommended action |
|---|---|---|
| `critical` | ≤ 20 | Take out of service and inspect immediately |
| `warning` | ≤ 50 | Schedule maintenance within the next planned window |
| `watch` | ≤ 80 | Increase monitoring frequency, review sensor trends |
| `healthy` | > 80 | No action required |

Thresholds are configurable under `inference.health`. `requires_action` is true
from `warning` upwards.

## Alerts

`monitoring/alerts.py` routes an `Alert` to every configured sink:

| Sink | Behaviour |
|---|---|
| JSONL log | Appends to `artifacts/reports/alerts.jsonl` (always) |
| Console | Logged at a level matching severity (`monitoring.alerts.console`) |
| Webhook | POSTed to `monitoring.alerts.webhook_url` if set |

Severities are `info`, `warning`, `critical`. A webhook failure is logged and
swallowed — a broken sink must not take down the job that raised the alert.

```python
from monitoring.alerts import Alert, AlertManager, Severity

AlertManager().emit(
    Alert(
        title="Critical RUL for engine 17",
        message="Take out of service and inspect immediately.",
        severity=Severity.CRITICAL,
        metric="rul_cycles", value=11.2, threshold=20.0, entity=17,
    )
)
```

`AlertManager().history(limit=50)` reads recent alerts back for a digest.

## Prometheus metrics

The API exposes `/api/v1/metrics` (also `/metrics`):

| Metric | Type | Use |
|---|---|---|
| `prognostix_predictions_total{model,risk_level}` | counter | Volume, and drift in the risk mix |
| `prognostix_prediction_errors_total{reason}` | counter | Rejected/failed requests by cause |
| `prognostix_prediction_duration_seconds{model}` | histogram | Latency |
| `prognostix_last_predicted_rul_cycles{model}` | gauge | Liveness sanity check |

A rising share of `critical` predictions with unchanged inputs is worth
investigating: either the fleet really is degrading, or the model has started
drifting pessimistic.

## A practical schedule

| Cadence | Job |
|---|---|
| Per request | Prometheus counters and latency |
| Hourly | `python scripts/predict.py --alerts` — score the fleet, alert on at-risk machines |
| Daily | `python -m monitoring.drift` — input drift against the training distribution |
| Weekly / as truth arrives | `python -m monitoring.performance` — error vs baseline |
| On `critical` from either | Investigate, then `python scripts/train.py --model gru` and re-evaluate |

## Retraining triggers

Retrain when any of these hold:

1. Drift status is `critical`, or `warning` persists across several runs.
2. Performance degradation is `critical` on a sample of at least `min_samples`.
3. `Bias` in the leaderboard turns systematically positive — the model has become
   optimistic, which is the dangerous direction.
4. The fleet, the sensors, or the maintenance regime changed materially.

After retraining, compare on the same split before promoting:

```bash
python scripts/train.py --model gru
python scripts/evaluate.py --split test
```
