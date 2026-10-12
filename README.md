# Demand Forecasting & Inventory Optimization

Forecasts daily unit sales for Walmart's M5 dataset and turns the forecasts into inventory
decisions, with the cost of stockouts and overstock quantified at different service levels.

## Architecture

- **Batch forecasting.** An Airflow pipeline ingests, validates, builds features, trains,
  evaluates, registers models, and publishes forecasts plus inventory recommendations to
  Postgres. The API only reads.
- **Simulated data arrival.** M5 is replayed day by day to exercise retraining and monitoring.
- **Deployment.** Kubernetes on OCI Always Free, Neon Postgres, OCI Object Storage.

Decisions: [ADR 0001](docs/adr/0001-platform-and-tooling.md) (platform), [ADR 0002](docs/adr/0002-data-layer.md) (data layer), [ADR 0003](docs/adr/0003-evaluation-and-baselines.md) (evaluation), [ADR 0004](docs/adr/0004-models-features-uncertainty.md) (models), [ADR 0005](docs/adr/0005-inventory-policy-and-costs.md) (inventory), [ADR 0006](docs/adr/0006-real-data-runs-on-kaggle.md) (full-dataset runs).

## Roadmap

- [x] Phase 0: Foundations (tooling, config, CI, Terraform base)
- [x] Phase 1: Data ingestion, validation, EDA
- [x] Phase 2: Baselines and backtesting
- [x] Phase 3: Statistical and LightGBM models, quantile forecasts
- [x] Phase 4: Inventory optimization and cost impact
- [ ] Real-data benchmark: full M5 on Kaggle
- [ ] Phase 5: Airflow pipeline
- [ ] Phase 6: API and dashboard
- [ ] Phase 7: Infrastructure and deployment
- [ ] Phase 8: Monitoring and write-up

## Local setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
cp .env.example .env
make install   # dependencies + git hooks
make check     # lint, type-check, tests
```

## Data

1. Create a Kaggle API token and save it as `~/.kaggle/kaggle.json`
   (or set `KAGGLE_USERNAME` and `KAGGLE_KEY`).
2. Accept the competition rules at
   https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules.
3. Run:

```bash
make download-data   # raw CSVs into data/raw/m5
make ingest          # validate and publish data/processed/m5 (add --state CA via df-ingest)
make eda             # execute notebooks/01_eda.py into notebooks/01_eda.ipynb
```

Each ingestion writes `manifest.json` with source hashes, row counts, and a `data_version`.
Full ingestion takes about a minute and peaks around 2.6 GB of memory.

## Models and evaluation

```bash
make backtest                                   # every model, rolling-origin folds
uv run df-backtest --models lightgbm tsb        # chosen models only
uv run df-backtest --state ALL                  # every ingested state
```

| Family | Models |
|---|---|
| Baselines | `naive`, `seasonal_naive`, `moving_average` |
| Statistical (statsforecast) | `croston_sba`, `tsb`, `ses`, `seasonal_es` |
| Machine learning | `lightgbm`: one model per store, Tweedie objective, features lagged by the horizon |

- **Point metrics:** WRMSSE across all 12 M5 hierarchy levels (primary), WAPE, MASE, bias.
- **Quantiles:** every model's forecasts become quantiles (0.5, 0.9, 0.95) through a
  negative binomial distribution per series, scored for coverage and pinball loss.
- **Folds:** three non-overlapping 28-day windows ending at the last day; the last one is the
  M5 public-leaderboard window (days 1914-1941).
- **Scope:** defaults to `data.train_subset_state` (CA); `--state ALL` uses every state.
- **Output:** `reports/backtest/<data_version>/<timestamp>/` with `results.parquet`, LightGBM
  feature importance, and a `summary.md` covering metrics, WRMSSE per level, quantile
  calibration, inventory cost, and top features.

LightGBM training dominates run time: about 2.5 minutes per store per fold on 2 vCPUs,
falling with more cores. Peak memory is about 4.3 GB.

## Inventory impact

Accuracy scores don't say what a forecast is worth, so every backtest also turns each model's
forecasts into stocking decisions and prices the outcome:

1. Every 7 days, each item on the shelf is topped up to a target level: enough to cover demand
   until the next delivery, 7 days later, with a chosen probability (the target service level).
2. Actual sales are replayed day by day. Demand beyond the stock on the shelf is lost.
3. Holding stock costs 25% of its price per year; a lost sale costs 40% of the price. These are
   assumptions (M5 has no cost data), set in `configs/inventory.yaml`.

For each model, the report gives the total cost at targets from 80% to 99%, the cheapest
target, the service actually achieved, the fill rate, the stock held, and the saving per year
against `seasonal_naive`.

```bash
make results   # charts: accuracy, stock versus service, cost by target, top features
```

## Real-data run on Kaggle

The full dataset runs in a free Kaggle CPU notebook: Kaggle attaches the M5 data directly, and
its sessions have about 30 GB of memory. The free-tier cluster trains on one state.

1. Commit your changes, then build the code bundle. It is made from the last commit, so it
   never contains `.env` or local data:

   ```bash
   make kaggle-bundle   # dist/kaggle/demand-forecasting-code.zip
   ```

2. On Kaggle, create a private dataset from that zip, and accept the
   [M5 rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules).
3. Create a notebook with **File > Import Notebook** and choose `notebooks/kaggle_run.ipynb`.
   Add both inputs (the M5 competition and your code dataset), set the accelerator to None,
   and turn the internet on.
4. Start it with **Save Version > Save & Run All (Commit)**. It installs the locked environment,
   then runs ingest, EDA, a backtest over every store, and the results charts.
5. Download `m5-run-all-<data_version>.zip` from the version's Output tab and unzip it at the
   repository root. Commit `notebooks/01_eda.ipynb` and `notebooks/02_results.ipynb`; the
   report stays in `reports/`.

Each report folder records the data version, the code commit, the session's CPUs and memory,
and the time and peak memory of each part (`run_info.json`). After editing
`notebooks/kaggle_run.py`, run `make kaggle-notebook` to rebuild the notebook to import.

## Experiment tracking

```bash
make mlflow    # local MLflow server on http://127.0.0.1:5000, data kept in mlruns/
```

Then set `DF_MLFLOW_TRACKING_URI=http://127.0.0.1:5000` in `.env` and run a backtest. Each
backtest logs a parent run (full config, data and code versions, report files) and one child
run per model (metrics averaged over folds, plus `fold/<metric>` for each fold). Without the
variable, tracking is skipped.

## Layout

```
configs/         YAML parameters (merged in order; secrets come from DF_* env vars)
src/forecasting/ Library code
tests/           Unit and integration tests
infra/terraform/ OCI infrastructure (state in Object Storage)
infra/helm/      Kubernetes charts (Phase 7)
notebooks/       EDA, results and Kaggle runner notebooks (Jupytext .py sources)
docs/adr/        Architecture decision records
```
