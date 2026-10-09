# Demand Forecasting & Inventory Optimization

Forecasts daily unit sales for Walmart's M5 dataset and turns the forecasts into inventory
decisions, with the cost of stockouts and overstock quantified at different service levels.

## Architecture

- **Batch forecasting.** An Airflow pipeline ingests, validates, builds features, trains,
  evaluates, registers models, and publishes forecasts plus inventory recommendations to
  Postgres. The API only reads.
- **Simulated data arrival.** M5 is replayed day by day to exercise retraining and monitoring.
- **Deployment.** Kubernetes on OCI Always Free, Neon Postgres, OCI Object Storage.

See [ADR 0001](docs/adr/0001-platform-and-tooling.md) for platform decisions.

## Roadmap

- [x] Phase 0: Foundations (tooling, config, CI, Terraform base)
- [ ] Phase 1: Data ingestion, validation, EDA
- [ ] Phase 2: Baselines and backtesting
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

## Layout

```
configs/         YAML parameters (merged in order; secrets come from DF_* env vars)
src/forecasting/ Library code
tests/           Unit and integration tests
infra/terraform/ OCI infrastructure (state in Object Storage)
infra/helm/      Kubernetes charts (Phase 7)
docs/adr/        Architecture decision records
```
