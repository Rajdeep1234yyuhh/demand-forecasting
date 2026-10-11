# Demand Forecasting & Inventory Optimization

Forecasts daily unit sales for Walmart's M5 dataset and turns the forecasts into inventory
decisions, with the cost of stockouts and overstock quantified at different service levels.

## Architecture

- **Batch forecasting.** An Airflow pipeline ingests, validates, builds features, trains,
  evaluates, registers models, and publishes forecasts plus inventory recommendations to
  Postgres. The API only reads.
- **Simulated data arrival.** M5 is replayed day by day to exercise retraining and monitoring.
- **Deployment.** Kubernetes on OCI Always Free, Neon Postgres, OCI Object Storage.

Decisions: [ADR 0001](docs/adr/0001-platform-and-tooling.md) (platform), [ADR 0002](docs/adr/0002-data-layer.md) (data layer), [ADR 0003](docs/adr/0003-evaluation-and-baselines.md) (evaluation).

## Roadmap

- [x] Phase 0: Foundations (tooling, config, CI, Terraform base)
- [x] Phase 1: Data ingestion, validation, EDA
- [x] Phase 2: Baselines and backtesting
- [ ] Phase 3: Statistical and LightGBM models, quantile forecasts
- [ ] Phase 4: Inventory optimization and cost impact
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

## Evaluation

```bash
make backtest        # rolling-origin backtest of the baselines
uv run df-backtest --state ALL --models seasonal_naive   # override scope or models
```

- **Metrics:** WRMSSE across all 12 M5 hierarchy levels (primary), WAPE, MASE, and bias.
- **Folds:** three non-overlapping 28-day windows ending at the last day; the last one is the
  M5 public-leaderboard window (days 1914-1941).
- **Scope:** defaults to `data.train_subset_state` (CA); `--state ALL` uses every state.
- **Output:** `reports/backtest/<data_version>/<timestamp>/` with `results.parquet` and a
  `summary.md` that includes WRMSSE per hierarchy level.

A full-scale backtest of three baselines takes about two minutes and peaks under 2 GB.

## Layout

```
configs/         YAML parameters (merged in order; secrets come from DF_* env vars)
src/forecasting/ Library code
tests/           Unit and integration tests
infra/terraform/ OCI infrastructure (state in Object Storage)
infra/helm/      Kubernetes charts (Phase 7)
docs/adr/        Architecture decision records
```
