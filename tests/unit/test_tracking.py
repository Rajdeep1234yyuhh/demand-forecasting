import shutil
import subprocess
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from forecasting.tracking import MlflowTracker, RunRecord, code_version, flatten_params, log_nested


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A throwaway local MLflow store, so tests need no server.

    MLflow 3 allows the file store only on opt-in; it's fine for tests, while real runs go
    to a tracking server.
    """
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return (tmp_path / "mlruns").as_uri()


class RecordingTracker:
    """Fake tracker that remembers calls instead of talking to MLflow."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, RunRecord, str | None]] = []

    def log_run(self, run: RunRecord, *, parent_run_id: str | None = None) -> str:
        run_id = f"run-{len(self.calls)}"
        self.calls.append((run_id, run, parent_run_id))
        return run_id


def test_flatten_params() -> None:
    nested = {"forecast": {"horizon_days": 28, "quantiles": [0.5, 0.9]}, "env": "dev", "x": None}
    assert flatten_params(nested) == {
        "forecast.horizon_days": "28",
        "forecast.quantiles": "[0.5, 0.9]",
        "env": "dev",
        "x": "None",
    }


def test_log_nested_logs_parent_then_children() -> None:
    tracker = RecordingTracker()
    parent_id = log_nested(tracker, RunRecord("parent"), [RunRecord("a"), RunRecord("b")])
    assert parent_id == "run-0"
    assert [(run.name, parent) for _, run, parent in tracker.calls] == [
        ("parent", None),
        ("a", "run-0"),
        ("b", "run-0"),
    ]


def test_mlflow_tracker_logs_a_complete_run(tracking_uri: str, tmp_path: Path) -> None:
    artifact = tmp_path / "summary.md"
    artifact.write_text("# report\n", encoding="utf-8")
    tracker = MlflowTracker(tracking_uri, "tests")
    parent_id = tracker.log_run(
        RunRecord(
            "parent", params={"horizon": "28"}, tags={"data_version": "abc"}, artifacts=[artifact]
        )
    )
    child_id = tracker.log_run(
        RunRecord(
            "naive",
            metrics={"wrmsse": 1.5},
            step_metrics=[("fold/wrmsse", 1.4, 1), ("fold/wrmsse", 1.6, 2)],
        ),
        parent_run_id=parent_id,
    )

    client = MlflowClient(tracking_uri)
    parent, child = client.get_run(parent_id), client.get_run(child_id)
    assert parent.info.status == child.info.status == "FINISHED"
    assert parent.data.params == {"horizon": "28"}
    assert parent.data.tags["data_version"] == "abc"
    assert [entry.path for entry in client.list_artifacts(parent_id)] == ["summary.md"]
    assert child.data.tags["mlflow.parentRunId"] == parent_id
    assert child.data.metrics["wrmsse"] == 1.5
    history = client.get_metric_history(child_id, "fold/wrmsse")
    assert sorted((metric.step, metric.value) for metric in history) == [(1, 1.4), (2, 1.6)]


def test_batches_stay_within_mlflow_limits(tracking_uri: str) -> None:
    run = RunRecord(
        "big",
        params={f"p{i}": str(i) for i in range(150)},
        metrics={f"m{i}": float(i) for i in range(1200)},
    )
    run_id = MlflowTracker(tracking_uri, "tests").log_run(run)
    data = MlflowClient(tracking_uri).get_run(run_id).data
    assert len(data.params) == 150
    assert len(data.metrics) == 1200


def test_reuses_the_experiment(tracking_uri: str) -> None:
    MlflowTracker(tracking_uri, "tests").log_run(RunRecord("first"))
    MlflowTracker(tracking_uri, "tests").log_run(RunRecord("second"))
    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name("tests")
    assert experiment is not None
    assert len(client.search_runs([experiment.experiment_id])) == 2


def test_failed_run_is_closed_as_failed(tracking_uri: str, tmp_path: Path) -> None:
    tracker = MlflowTracker(tracking_uri, "tests")
    with pytest.raises(OSError, match=r"missing\.md"):
        tracker.log_run(RunRecord("broken", artifacts=[tmp_path / "missing.md"]))
    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name("tests")
    assert experiment is not None
    [run] = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FAILED"


def test_code_version_prefers_build_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DF_GIT_SHA", "abc123")
    assert code_version() == "abc123"


def test_code_version_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DF_GIT_SHA", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert code_version() == "unknown"


def _git(*args: str) -> str:
    git = shutil.which("git")
    assert git is not None
    result = subprocess.run(  # noqa: S603 - fixed arguments, in a temporary repository
        [git, "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_code_version_marks_uncommitted_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DF_GIT_SHA", raising=False)
    monkeypatch.chdir(tmp_path)
    _git("init", "-q")
    (tmp_path / "file.txt").write_text("one\n", encoding="utf-8")
    _git("add", "file.txt")
    _git("commit", "-q", "-m", "initial")
    head = _git("rev-parse", "HEAD")

    assert code_version() == head
    (tmp_path / "file.txt").write_text("two\n", encoding="utf-8")
    assert code_version() == f"{head}-dirty"
