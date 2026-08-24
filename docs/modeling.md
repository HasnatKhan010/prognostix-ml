# Modeling

Six models, one training protocol, one leaderboard.

```bash
python scripts/train.py --model all
python scripts/evaluate.py
```

## Why sequence models

RUL is not readable from a single cycle. Two engines can report identical sensor
values while one is 200 cycles from failure and the other is 20 — what separates
them is the *trajectory*. So the input is a window of 30 consecutive cycles
(`30 × 15` sensor readings) and the output is a single number.

## The models

### Naive mean — the floor

Predicts the mean training RUL for every input, ignoring the sensors entirely.
Any model that cannot beat it by a wide margin has learned the RUL distribution
rather than degradation. On FD001 it reaches ~57.8 RMSE.

### Linear regression

Fitted on the 90 flattened window statistics (`mean, std, min, max, last, trend`
per sensor). Fast, interpretable, and a useful check on whether the deep models
earn their complexity — on FD001 it reaches ~32.8 RMSE, which is a large jump
over the naive floor and shows that most of the signal is in simple aggregates.

### Random forest

Same 90 features, 200 trees. Captures non-linear interactions and gives feature
importances. It is the largest artifact by far (~180 MB), which is why the
serving layer loads it lazily and defaults to the GRU.

### LSTM

Two stacked LSTM layers (128 hidden units, dropout 0.2). The prediction is read
from the final timestep's hidden state, which by construction has seen the whole
window. Three gates per unit let it hold information across long stretches.

### GRU — the default served model

Same shape as the LSTM, but two gates instead of three: ~25% fewer parameters and
faster training, at little accuracy cost on CMAPSS. That trade is why it is the
default in `configs/config.yaml`.

### Attention

An LSTM encoder followed by multi-head self-attention and **additive attention
pooling**. Two things improve over reading the last hidden state:

1. Every cycle can attend directly to every other cycle, so nothing has to
   survive a single-vector bottleneck.
2. The pooling weights are readable. `predictor.explain(window)` returns the
   per-cycle weights, and `/api/v1/fleet/{id}` includes them — the model reports
   which cycles drove its answer, which is the difference between a number a
   planner can act on and one they must take on faith.

## Training protocol

All three sequence models run through `src/models/runner.py`, so leaderboard
differences reflect architecture and nothing else:

| Setting | Value |
|---|---|
| Loss | MSE |
| Optimiser | Adam, lr 1e-3 |
| Batch size | 128 |
| Epochs | 30 (early stopping, patience 8) |
| Gradient clipping | 1.0 |
| LR schedule | `ReduceLROnPlateau`, factor 0.5, patience 3 |
| Seed | 42 (`random`, `numpy`, `torch`) |

**Early stopping keeps the best weights, not the last ones.** Validation loss on
this data typically bottoms out well before epoch 30 and then drifts upward;
`EarlyStopping` snapshots the best state and restores it when training ends.

Everything is configurable in `configs/config.yaml` or per run:

```bash
python scripts/train.py --model lstm --epochs 60 --lr 5e-4 --device cuda
```

## Metrics

| Metric | What it tells you |
|---|---|
| **MAE** | Average error in cycles — the most directly interpretable |
| **RMSE** | Penalises large misses; the primary ranking metric |
| **R²** | Variance explained |
| **MAPE** | Relative error (denominator floored, since RUL reaches 0) |
| **Within10** | Share of predictions within 10 cycles |
| **Bias** | Mean signed error — positive means systematically optimistic |
| **NASAScore** | The asymmetric PHM08 score |

### Why the asymmetric score matters

MAE and RMSE treat a 10-cycle optimistic error the same as a 10-cycle
conservative one. In maintenance they are not the same: predicting *more* life
than a machine has left means it fails in service. The NASA/PHM08 score encodes
that asymmetry:

```
d = predicted - actual
d > 0  (late, optimistic)     → exp(d/10) - 1     ← steeper
d ≤ 0  (early, conservative)  → exp(-d/13) - 1
```

A model can win on RMSE while being systematically optimistic. Read `Bias` and
`NASAScore` alongside it.

## Results (FD001, validation split, uncapped RUL)

| Model | MAE | RMSE |
|---|---|---|
| Linear regression | 25.59 | 32.81 |
| Random forest | 26.00 | 37.88 |
| Naive mean | 48.63 | 57.81 |

These are the committed baseline numbers from `artifacts/baseline_results.csv`.
Run `python scripts/train.py --model all` to fill in the sequence models on your
hardware; results land in `artifacts/model_comparison.csv` and the figures in
`artifacts/figures/`.

Note the ordering: random forest wins on MAE but loses on RMSE, meaning it is
usually closer but occasionally badly wrong. That is exactly the failure mode
RMSE exists to expose, and exactly the wrong shape for maintenance.

## Evaluation modes

```bash
python scripts/evaluate.py                       # windowed test split (default)
python scripts/evaluate.py --official            # CMAPSS competition protocol
```

**Windowed** scores every sliding window — thousands of predictions, stable and
comparable across models.

**Official** follows the competition protocol: one prediction per test engine
from its final window, scored against `RUL_FD001.txt`. Fewer points and harder,
but it is the number the literature quotes. Use it when comparing against
published results.

## Checkpoints

Each `.pt` file stores the weights plus everything needed to rebuild and serve
the model:

```python
{
  "model_state_dict": ...,
  "model_type": "gru",
  "input_size": 15,
  "model_kwargs": {"hidden_size": 128, "num_layers": 2, "dropout": 0.2},
  "hidden_size": 128, ...,          # flat copies, for notebook-era loaders
  "feature_columns": [...],         # the serving contract
  "window_size": 30,
  "metrics": {"val": {...}},
  "history": {"train_loss": [...], "val_loss": [...]},
}
```

The serving layer reads `model_type` and `model_kwargs` to reconstruct the exact
architecture — no config file needed, and no risk of loading weights into a model
that was built differently.

## Adding a model

1. Write the module under `src/models/<name>/model.py`, exposing
   `forward(x) -> (batch,)` for input `(batch, window, features)`.
2. Register it in `TORCH_MODELS` in `src/models/__init__.py`.
3. Add its hyperparameters under `models:` in `configs/config.yaml`.
4. Add a thin `train.py` that calls `run_sequence_training("<name>")`.

It then works with `scripts/train.py --model <name>`, the evaluator, the API and
the dashboard with no further changes.
