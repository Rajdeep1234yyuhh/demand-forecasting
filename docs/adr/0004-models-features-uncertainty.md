# ADR 0004: Models, features, and uncertainty

- Status: Accepted
- Date: 2026-10-11

## Context

Phase 3 adds classical and learned models to the Phase 2 evaluation. The models must not
leak future information, must train within one free-tier node's memory, must be
reproducible, and must produce quantiles that the Phase 4 inventory policy can use.

## Decision

| Concern | Choice | Reason |
|---|---|---|
| Learned model | LightGBM, one model per store | Peak memory stays at one store's data however many stores there are; stores can train in parallel (Phase 5); one store's failure doesn't block the others |
| Objective | Tweedie, variance power 1.1 | Intermittent counts: many exact zeros and a long right tail |
| Forecast strategy | Direct: every sales feature is lagged by at least the 28-day horizon | One model covers every horizon day; no forecasts are fed back as inputs, so errors don't compound |
| Leakage guard | Config rejects lags shorter than the horizon; feature code refuses them too; a test checks that a day's features are identical before and after its sales are known | Leakage is blocked by construction, not by convention |
| Availability | An item is on the shelf in weeks it has a price; train only on on-shelf days; forecast zero off the shelf | Days before release aren't demand; prices are set in advance, so this is known on the forecast date |
| Categorical encoding | Codes learned per store from training rows; unseen values become missing | Encoding stays a model concern; nothing leaks in from later data |
| Statistical models | statsforecast: Croston SBA, TSB, SES, seasonal ES, each fitted from the series' first sale | Standard intermittent-demand benchmarks, about one second per store |
| Uncertainty | Negative binomial per series: size by method of moments over the last 112 days, mean from any point model | No extra models; quantiles never cross; gives lead-time demand for Phase 4 |
| Tracking | `mlflow-skinny` client logging to an MLflow server through `MlflowClient` batch calls, after the report is written | Light dependency; no global state; a tracking failure never loses results |
| Reproducibility | Deterministic LightGBM with a fixed seed; every run tagged with code version (commit, marked `-dirty` for uncommitted changes), data version, and the full config | Any number in MLflow can be traced and rebuilt |

## Measured

On synthetic data with M5's exact shape (30,490 series x 1,941 days) and M5-like behaviour,
on 2 vCPUs and 7 GB of memory:

| What | Result |
|---|---|
| Features for one store, full history (5.9M rows) | about 8 s, 2.9 GB peak |
| LightGBM, one store, 200 rounds | 99 s |
| statsforecast per store (3,049 series) | Croston SBA 0.4 s, TSB 0.3 s, SES 1.0 s, seasonal ES 1.2 s |
| AutoETS | 0.39 s per series: about 10 hours per 3-fold backtest, so excluded |
| Croston SBA vs TSB, same store and fold | MASE 3.40 vs 0.95, same WAPE |
| Negative binomial calibration on simulated data | coverage within 3 points below to 8 points above each level |
| California backtest, 1 fold, all 8 models | 704 s and 4.3 GB peak; LightGBM took 633 s of it (4 stores, 300 rounds) |
| LightGBM vs best baseline, same fold | WRMSSE 0.31 vs 0.78 (seasonal naive); WAPE 61.9% vs 79.9% |
| LightGBM quantiles at 0.5 / 0.9 / 0.95 | coverage 73% / 95% / 98%; lowest pinball loss of all models at each level |
| Determinism | two separate runs gave identical results (WRMSSE 0.3053) |

The synthetic data has weekday, SNAP, and promotion effects that LightGBM can learn, so these
numbers check the mechanics, not the accuracy on the real M5 data.

Croston's MASE came from items that sold once at launch and never again: the top 1% of
series made up 72% of its MASE. Croston updates only when a sale happens, so it keeps
forecasting the old rate for dead items; TSB lowers the probability of a sale on every
zero day. Both stay in the registry, so every report shows the contrast.

## Consequences

- LightGBM training dominates backtest time. Levers, in order: `num_threads` on bigger
  machines, parallel stores in Phase 5, fewer boosting rounds, a shorter training window.
- Quantiles are conservative for good models: the method of moments uses each series'
  total variance, including variation the model explains (weekdays, promotions). Upgrade
  path: estimate dispersion from backtest residuals instead.
- Coverage of count quantiles sits above the level by construction: with many zero-sales
  days the median is often 0, and an actual 0 counts as covered (73% at the 0.5 level).
  A randomized PIT histogram is the stricter calibration check if one is needed.
- Mean MASE is dominated by low-volume series with tiny scales; read it next to WRMSSE.
- Feature importance is LightGBM's total gain, which favours continuous and high-cardinality
  features. SHAP values are the upgrade if explanations need to be per-forecast.
- Saving and loading models is left to Phase 5, where the pipeline decides whether training
  and prediction run as separate tasks.
- In Kubernetes, `num_threads` must match the pod's CPU limit: OpenMP sees the node's cores.
- The MLflow 3 server starts a job runner and several job consumers by default (about 2 GB
  in total here). Tracking doesn't need them, so `make mlflow` runs one worker with
  `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` (about 0.5 GB); the Phase 7 deployment should
  do the same on the free-tier node.
