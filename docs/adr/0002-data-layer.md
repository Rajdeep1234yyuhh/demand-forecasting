# ADR 0002: Data layer

- Status: Accepted
- Date: 2026-10-09

## Context

M5 has 30,490 series over 1,941 days: about 59M series-days. Ingestion must fit the
in-cluster memory budget and ordinary laptops, be reproducible, and never publish a
partial or invalid dataset.

## Decision

| Concern | Choice |
|---|---|
| Engine | Polars, lazy and streaming; native arm64 wheels |
| Validation | Pandera schemas with dtypes fixed at read time and no coercion; cross-table rules for calendar coverage, hierarchy consistency, and price coverage |
| Memory | Reshape and write one store at a time |
| Storage | Parquet, zstd, Hive-partitioned by `store_id`, sorted by `id` and `date` |
| Publishing | Write to a staging directory, then rename into place; restore the previous dataset if the swap fails |
| Versioning | `data_version` = hash of source-file hashes, `TRANSFORM_VERSION`, and the state filter; stored in `manifest.json` |

## Measured

Load test on synthetic data at full M5 shape, on a 3 GB machine:

| Run | Rows | Time | Peak memory |
|---|---|---|---|
| Ingest, all states | 59.2M | 64 s | 2.6 GB |
| Every EDA query, all states | 59.2M | ≤ 7 s each | ≤ 0.35 GB |

Processing a whole state at once peaked above 3.7 GB, which is why ingestion is per store.

## Consequences

- Bump `TRANSFORM_VERSION` whenever the transformation changes, so `data_version` changes too.
- Every MLflow run logs `data_version`, tying each model to the exact data it saw.
- `(id, day)` uniqueness holds by construction and is not re-checked; checking it would
  dominate ingestion time.
- The 1% missing-price threshold is a starting point; revisit after the first real run.
- Store closures (e.g. Christmas) stay in the data and are handled as features, not dropped.
