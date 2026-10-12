# ADR 0006: Full-dataset runs on Kaggle

- Status: Accepted
- Date: 2026-10-12

## Context

The headline results need all of M5: 30,490 series, three 28-day folds, and every model. That
takes hours of CPU and more memory than the places the project runs today. The development
workspace has about 5 GB of memory and no access to Kaggle, and the free-tier cluster trains on
one state to fit its node budget (ADR 0001). Kaggle hosts M5 and offers free CPU notebooks with
about 30 GB of memory (at the time of writing), with competition data attached in place.

## Decision

| Concern | Choice | Reason |
|---|---|---|
| Compute | A Kaggle CPU notebook, started with Save & Run All | Free, enough memory for every store, keeps running in the background; nothing needs a GPU |
| Data | The M5 competition attached as a notebook input and linked into `data/raw/m5` | No download step and no Kaggle API token in the run |
| Code | `make kaggle-bundle`: a `git archive` of the last commit, uploaded as a private Kaggle dataset | Kaggle gets no GitHub access; the bundle can't contain `.env` or local data; the commit is recorded |
| Environment | uv installs `uv.lock` into its own Python 3.12 environment | Same package versions as local and CI, not Kaggle's preinstalled ones |
| Runner | `notebooks/kaggle_run.py`: standard library only, calling the same CLIs as `make` | Runs on whatever Python Kaggle has; no second pipeline to maintain |
| Measurement | Wall time and peak memory of each part, read with `wait4`, saved in `run_info.json` | Phases 5 and 7 size Airflow tasks and pod limits from measured numbers |
| Output | One zip laid out like the repository: the executed notebooks and the report | Unzip at the root and commit the notebooks; the report keeps its data version and commit |

## Consequences

- Full-dataset numbers come from a run started by hand, not from the scheduled pipeline. The
  Phase 5 pipeline runs on the cluster's scope, and every report states its scope.
- ADR 0001 expected full-dataset runs to log to MLflow. Kaggle can't reach a private tracking
  server, so the report folder, with `run_info.json`, is the record of each run.
- Installs need the notebook's internet switched on, which needs a phone-verified Kaggle account.
- Kaggle can change its hardware and limits; each run records the CPUs and memory it had.
- A new code version means a new bundle upload: one command and one upload.
- Peak memory is the largest single process of each part, not the sum of its processes.
