# ADR 0001: Platform and tooling

- Status: Accepted
- Date: 2026-10-06

## Context

Portfolio-grade demand forecasting system on the M5 dataset. Requirements: production-style
deployment on Kubernetes, public read-only access, managed Postgres with point-in-time
recovery, and zero recurring cost.

## Decision

| Concern | Choice | Reason |
|---|---|---|
| Compute | OCI OKE basic cluster on Always Free Ampere A1 (4 OCPU / 24 GB, arm64) | Only free managed Kubernetes with free nodes |
| Database | Neon Postgres, Free plan | Managed, with point-in-time restore on the free tier |
| Object storage | OCI Object Storage, versioned, lifecycle-managed | Free 20 GB, S3-compatible, same cloud |
| Scaling | HPA on API and dashboard; fixed node pool, no cluster autoscaler | Extra nodes would be billed |
| Edge | Cloudflare free proxy, Traefik rate limiting | DDoS protection, TLS, hidden origin |
| Secrets | Sealed Secrets | Free, provider-agnostic, encrypted in git |
| Orchestration | Airflow, KubernetesExecutor | Strongest hiring signal; per-task pod isolation |
| Dashboard | Next.js | Production-grade UI |
| Tracking | MLflow | Experiment tracking and model registry |

## Consequences

- All images must be built for `linux/arm64`.
- Total Ampere A1 usage must never exceed 4 OCPU / 24 GB, or instances are disabled.
- In-cluster training uses one state (CA); full-dataset training runs locally and logs to MLflow.
- Neon limits (0.5 GB storage, 100 CU-hours/month, 6-hour restore window) mean:
  only the latest forecast run and aggregated history are stored in Neon; Airflow's
  metadata DB runs in-cluster; nightly `pg_dump` to Object Storage extends recovery to 30 days.
- Models are recoverable through reproducibility (git SHA, data version, config per MLflow
  run) rather than cross-region replication.
