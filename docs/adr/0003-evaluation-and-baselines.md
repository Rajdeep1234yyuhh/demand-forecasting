# ADR 0003: Evaluation and baselines

- Status: Accepted
- Date: 2026-10-09

## Context

Every later model decision depends on a trustworthy yardstick. It must avoid look-ahead
leakage, reflect how the model is retrained in production, be comparable with published
M5 results, and run within the same memory budget as ingestion.

## Decision

| Concern | Choice | Reason |
|---|---|---|
| Primary metric | WRMSSE across all 12 hierarchy levels (M5 official) | Scale-free, revenue-weighted, comparable with published results |
| Secondary metrics | WAPE, bottom-level MASE, bias | WAPE reads as a business number; MASE is per-series and scale-free; bias exposes systematic over- or under-stocking |
| Validation | Rolling origin: 3 non-overlapping 28-day folds ending at the last day | Mimics periodic retraining with no leakage; fold 3 is exactly the M5 public-leaderboard window (days 1914-1941) |
| Baselines | Naive, seasonal naive (7 days), 28-day moving average | Standard, cheap; a learned model must beat all three |
| Model contract | `Forecaster` base class; output validated by Pandera; exact coverage check against actuals | A model bug fails loudly instead of skewing a metric |
| Scales and weights | Recomputed per fold from that fold's training data only | Matches the M5 definition and avoids leakage |
| Computation | One read per store; levels rolled up item -> department -> category -> total; state and global item tables accumulated as stores stream | Bounded memory; relies on the hierarchy ingestion validates |
| Correctness | Tested against an independent plain-Python WRMSSE implementation | The vectorized code is not checked against itself |

## Measured

Synthetic data at full M5 shape (59.2M series-days), all states, 3 folds, 3 baselines:

| Version | Total time | Peak memory | Evaluator setup per fold |
|---|---|---|---|
| Per-level scans (14 per fold) | 313 s | 5.76 GB | ~98 s |
| Single-pass roll-up | 173 s | 2.83 GB | ~52 s |
| + column addition when store grids align | 107 s | 1.75 GB | ~30 s |

California only (the in-cluster scope): 45 s, 1.43 GB. All three versions produced the same
metrics to within 1e-15.

## Consequences

- WRMSSE and WAPE can rank models differently. On synthetic data, seasonal naive won on
  WRMSSE (it captures weekly patterns at aggregate levels) while the moving average won on
  WAPE and MASE (it smooths item-level noise). WRMSSE stays the comparison metric, but the
  final model choice is made on inventory cost in Phase 4.
- Series with fewer than two active days in a fold's training data cannot be scaled and are
  excluded; their combined weight is reported as `unscored_weight`.
- Store closures (e.g. Christmas) are not excluded from scoring. None of the default folds
  contain one; revisit if the fold layout changes.
- The aligned-grid fast path checks key equality on every call and falls back to a
  group-by, so it never trades correctness for speed.
- Reports are written per run under `reports/backtest/<data_version>/` and are not
  committed. Experiment tracking moves to MLflow in Phase 3.
