"""Experiment tracking with MLflow.

The project depends only on ``mlflow-skinny``, a lightweight client that logs over HTTP to a
tracking server: ``make mlflow`` locally, the shared server in the cluster later. Runs go
through ``MlflowClient`` with batched calls rather than MLflow's global "fluent" API, so
there is no hidden global state and each run takes a handful of requests.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# MLflow's per-request limits for batch logging.
_MAX_METRICS_PER_BATCH = 1000
_MAX_PARAMS_PER_BATCH = 100
_PARENT_TAG = "mlflow.parentRunId"


@dataclass(frozen=True)
class RunRecord:
    """Everything one tracking run records."""

    name: str
    params: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    # (key, value, step) for metrics with a history, such as one value per fold.
    step_metrics: Sequence[tuple[str, float, int]] = ()
    tags: Mapping[str, str] = field(default_factory=dict)
    artifacts: Sequence[Path] = ()


class Tracker(Protocol):
    def log_run(self, run: RunRecord, *, parent_run_id: str | None = None) -> str:
        """Log one complete run and return its ID."""
        ...


def _chunks[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class MlflowTracker:
    """Logs each run in full (create, batch log, upload, close) through ``MlflowClient``."""

    def __init__(self, tracking_uri: str, experiment: str) -> None:
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._experiment = experiment
        self._experiment_id: str | None = None

    def _get_experiment_id(self) -> str:
        if self._experiment_id is None:
            existing = self._client.get_experiment_by_name(self._experiment)
            if existing is not None:
                self._experiment_id = existing.experiment_id
            else:
                try:
                    self._experiment_id = self._client.create_experiment(self._experiment)
                except MlflowException:
                    # Another process created it first.
                    created = self._client.get_experiment_by_name(self._experiment)
                    if created is None:
                        raise
                    self._experiment_id = created.experiment_id
        return self._experiment_id

    def log_run(self, run: RunRecord, *, parent_run_id: str | None = None) -> str:
        tags = dict(run.tags)
        if parent_run_id is not None:
            tags[_PARENT_TAG] = parent_run_id
        run_id: str = self._client.create_run(
            self._get_experiment_id(), run_name=run.name, tags=tags
        ).info.run_id
        try:
            now = int(time.time() * 1000)
            metrics = [Metric(key, value, now, 0) for key, value in run.metrics.items()]
            metrics += [Metric(key, value, now, step) for key, value, step in run.step_metrics]
            params = [Param(key, value) for key, value in run.params.items()]
            for metric_batch in _chunks(metrics, _MAX_METRICS_PER_BATCH):
                self._client.log_batch(run_id, metrics=metric_batch)
            for param_batch in _chunks(params, _MAX_PARAMS_PER_BATCH):
                self._client.log_batch(run_id, params=param_batch)
            for artifact in run.artifacts:
                self._client.log_artifact(run_id, str(artifact))
        except Exception:
            # Close the run as failed so it doesn't stay "running" in the UI forever.
            self._client.set_terminated(run_id, status="FAILED")
            raise
        self._client.set_terminated(run_id, status="FINISHED")
        return run_id


def log_nested(tracker: Tracker, parent: RunRecord, children: Sequence[RunRecord]) -> str:
    """Log a parent run, then each child under it; return the parent's ID."""
    parent_id = tracker.log_run(parent)
    for child in children:
        tracker.log_run(child, parent_run_id=parent_id)
    return parent_id


def flatten_params(values: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested settings into dotted keys with string values, as MLflow params need."""
    flat: dict[str, str] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, prefix=f"{name}."))
        elif isinstance(value, list | tuple):
            flat[name] = json.dumps(list(value))
        else:
            flat[name] = str(value)
    return flat


def _git(*args: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [git, *args], capture_output=True, text=True, check=True, timeout=10
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip()


def code_version() -> str:
    """The commit that produced a run, marked ``-dirty`` if there were uncommitted changes.

    Container builds have no ``.git`` directory, so they pass the commit in ``DF_GIT_SHA``.
    """
    if sha := os.environ.get("DF_GIT_SHA"):
        return sha
    commit = _git("rev-parse", "HEAD")
    if not commit:
        return "unknown"
    changes = _git("status", "--porcelain", "--untracked-files=no")
    return f"{commit}-dirty" if changes else commit
