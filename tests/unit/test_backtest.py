from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from mlflow.tracking import MlflowClient

from forecasting.data.dataset import Dataset
from forecasting.data.schemas import DataValidationError
from forecasting.evaluation.backtest import (
    backtest_runs,
    main,
    make_folds,
    run_backtest,
    summarize,
    write_report,
)
from forecasting.models.base import Forecaster
from forecasting.models.baselines import MovingAverage, Naive, SeasonalNaive
from forecasting.models.uncertainty import NegativeBinomialUncertainty
from forecasting.utils.config import get_secrets, load_config
from tests.conftest import N_DAYS, START

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
HORIZON = 3
FOLDS = 2
LAST = START + timedelta(days=N_DAYS - 1)


class Oracle(Forecaster):
    """Test double that forecasts the real future, so every error metric must be zero."""

    name = "oracle"

    def __init__(self, sales: pl.LazyFrame) -> None:
        super().__init__()
        self._sales = sales

    def _fit(self, history: pl.LazyFrame, cutoff: date) -> None:
        return None

    def _predict(self, cutoff: date, horizon: int) -> pl.DataFrame:
        window = pl.col("date").is_between(
            cutoff + timedelta(days=1), cutoff + timedelta(days=horizon)
        )
        return (
            self._sales.filter(window)
            .select("id", "date", pl.col("sales").cast(pl.Float64).alias("forecast"))
            .collect()
        )


class DropsASeries(Oracle):
    name = "drops_a_series"

    def _predict(self, cutoff: date, horizon: int) -> pl.DataFrame:
        forecasts = super()._predict(cutoff, horizon)
        return forecasts.filter(pl.col("id") != "FOODS_1_001_CA_1_evaluation")


def test_make_folds() -> None:
    folds = make_folds(date(2016, 5, 22), 28, 3)
    assert [fold.index for fold in folds] == [1, 2, 3]
    assert [fold.cutoff for fold in folds] == [
        date(2016, 2, 28),
        date(2016, 3, 27),
        date(2016, 4, 24),
    ]
    assert [fold.end for fold in folds] == [date(2016, 3, 27), date(2016, 4, 24), date(2016, 5, 22)]
    assert folds[0].start == date(2016, 2, 29)
    with pytest.raises(ValueError, match="positive"):
        make_folds(date(2016, 5, 22), 0, 3)


def test_backtests_baselines(dataset: Dataset) -> None:
    models = [Naive(), SeasonalNaive(period=7), MovingAverage(window=7)]
    results = run_backtest(dataset, models, horizon=HORIZON, n_folds=FOLDS)
    assert results.height == FOLDS * len(models)
    assert set(results.get_column("model")) == {"naive", "seasonal_naive", "moving_average"}
    assert sorted(set(results.get_column("cutoff"))) == [
        LAST - timedelta(days=2 * HORIZON),
        LAST - timedelta(days=HORIZON),
    ]
    assert {"wrmsse", "wape", "mase", "bias", "seconds", "wrmsse_item_store"} <= set(
        results.columns
    )
    assert results.get_column("wrmsse").is_finite().all()
    assert (results.get_column("wrmsse") > 0).all()


def test_oracle_scores_zero(dataset: Dataset) -> None:
    results = run_backtest(dataset, [Oracle(dataset.sales)], horizon=HORIZON, n_folds=FOLDS)
    assert results.get_column("wrmsse").to_list() == [0.0, 0.0]
    assert results.get_column("wape").to_list() == [0.0, 0.0]


def test_rejects_misaligned_forecasts(dataset: Dataset) -> None:
    with pytest.raises(DataValidationError, match="3 series-days missing"):
        run_backtest(dataset, [DropsASeries(dataset.sales)], horizon=HORIZON, n_folds=FOLDS)


def test_rejects_invalid_runs(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_backtest(dataset, [], horizon=HORIZON, n_folds=FOLDS)
    with pytest.raises(ValueError, match="unique"):
        run_backtest(dataset, [Naive(), Naive()], horizon=HORIZON, n_folds=FOLDS)
    with pytest.raises(ValueError, match="more history"):
        run_backtest(dataset, [Naive()], horizon=HORIZON, n_folds=5)


def test_state_scope(ingested: Path) -> None:
    texas = Dataset.load(ingested, "TX")
    states = texas.sales.select(pl.col("state_id").unique()).collect().get_column("state_id")
    assert states.to_list() == ["TX"]
    assert texas.prices.select(pl.col("store_id").str.starts_with("TX_").all()).collect().item()
    results = run_backtest(texas, [Naive()], horizon=HORIZON, n_folds=FOLDS)
    assert results.height == FOLDS

    with pytest.raises(DataValidationError, match="no sales data"):
        run_backtest(Dataset.load(ingested, "WI"), [Naive()], horizon=HORIZON, n_folds=FOLDS)


def test_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make ingest"):
        Dataset.load(tmp_path)


def test_report(dataset: Dataset, tmp_path: Path) -> None:
    results = run_backtest(
        dataset, [Naive(), Oracle(dataset.sales)], horizon=HORIZON, n_folds=FOLDS
    )
    assert summarize(results).get_column("model").to_list() == ["oracle", "naive"]

    run_dir = write_report(
        results, tmp_path / "reports", data_version="abc123", scope="all states", horizon=HORIZON
    )
    assert run_dir.parent == tmp_path / "reports" / "backtest" / "abc123"
    assert pl.read_parquet(run_dir / "results.parquet").shape == results.shape
    summary = (run_dir / "summary.md").read_text("utf-8")
    assert "Data version: `abc123`" in summary
    assert "| oracle | 0.0000 |" in summary
    assert "## WRMSSE by level" in summary
    assert summary.index("| oracle |") < summary.index("| naive |")
    # No quantiles or feature importance were produced, so neither section appears.
    assert "Quantile calibration" not in summary
    assert "top features" not in summary


def test_scores_quantiles(dataset: Dataset) -> None:
    results = run_backtest(
        dataset,
        [Naive(), MovingAverage(window=7)],
        horizon=HORIZON,
        n_folds=FOLDS,
        uncertainty=NegativeBinomialUncertainty(window_days=7),
        levels=(0.5, 0.9),
    )
    for column in ("q_0.5", "q_0.9"):
        assert results.get_column(f"coverage_{column}").is_between(0, 1).all()
        assert (results.get_column(f"pinball_{column}") >= 0).all()
    # A higher quantile can only cover more of the same actuals.
    assert (results.get_column("coverage_q_0.9") >= results.get_column("coverage_q_0.5")).all()


def test_uncertainty_requires_levels(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="quantile levels"):
        run_backtest(
            dataset,
            [Naive()],
            horizon=HORIZON,
            n_folds=FOLDS,
            uncertainty=NegativeBinomialUncertainty(window_days=7),
        )


def test_report_calibration_and_importance(dataset: Dataset, tmp_path: Path) -> None:
    results = run_backtest(
        dataset,
        [Naive()],
        horizon=HORIZON,
        n_folds=FOLDS,
        uncertainty=NegativeBinomialUncertainty(window_days=7),
        levels=(0.9,),
    )
    importance = pl.DataFrame(
        {
            "store_id": ["CA_1", "CA_1", "TX_1", "TX_1"],
            "feature": ["lag_28", "snap", "lag_28", "snap"],
            "gain": [3.0, 1.0, 1.0, 1.0],
            "share": [0.75, 0.25, 0.5, 0.5],
        }
    )
    run_dir = write_report(
        results,
        tmp_path,
        data_version="abc123",
        scope="CA",
        horizon=HORIZON,
        importance={"lightgbm": importance},
    )
    summary = (run_dir / "summary.md").read_text("utf-8")
    assert "## Quantile calibration" in summary
    assert "| Model | q_0.9 coverage | q_0.9 pinball |" in summary
    assert "## lightgbm: top features, last fold" in summary
    # Shares are averaged over stores: lag_28 (0.75 + 0.5) / 2, snap (0.25 + 0.5) / 2.
    assert "| lag_28 | 62.5% |" in summary
    assert summary.index("| lag_28 |") < summary.index("| snap | 37.5% |")
    assert pl.read_parquet(run_dir / "feature_importance_lightgbm.parquet").equals(importance)


def _cli_config(tmp_path: Path) -> list[str]:
    override = tmp_path / "override.yaml"
    override.write_text(
        f"reports_dir: {tmp_path / 'reports'}\n"
        f"data:\n  root: {tmp_path}\n  max_missing_price_ratio: 0.0\n"
        "forecast:\n  horizon_days: 3\n  backtest_folds: 2\n  moving_average_window: 7\n",
        "utf-8",
    )
    return [str(CONFIGS / "base.yaml"), str(CONFIGS / "inventory.yaml"), str(override)]


def _summaries(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "reports" / "backtest").glob("*/*/summary.md"))


@pytest.mark.usefixtures("ingested")
def test_cli_all_states(tmp_path: Path) -> None:
    assert main(["--config", *_cli_config(tmp_path), "--state", "ALL"]) == 0
    [summary] = _summaries(tmp_path)
    text = summary.read_text("utf-8")
    assert "Scope: all states" in text
    # Every registered model ran, with quantiles, and LightGBM explained its features.
    assert "| q_0.5 coverage | q_0.9 coverage | q_0.95 coverage |" in text
    assert "## lightgbm: top features, last fold" in text
    assert (summary.parent / "feature_importance_lightgbm.parquet").is_file()


@pytest.mark.usefixtures("ingested")
def test_cli_defaults_to_training_subset(tmp_path: Path) -> None:
    # base.yaml sets data.train_subset_state to CA.
    assert main(["--config", *_cli_config(tmp_path), "--models", "naive"]) == 0
    [summary] = _summaries(tmp_path)
    assert "Scope: CA" in summary.read_text("utf-8")


def test_cli_failure_returns_nonzero(tmp_path: Path) -> None:
    assert main(["--config", *_cli_config(tmp_path)]) == 1


def test_backtest_runs(dataset: Dataset, tmp_path: Path) -> None:
    config = load_config(CONFIGS / "base.yaml", CONFIGS / "inventory.yaml")
    results = run_backtest(
        dataset,
        [Naive(), SeasonalNaive(period=7)],
        horizon=HORIZON,
        n_folds=FOLDS,
        uncertainty=NegativeBinomialUncertainty(window_days=7),
        levels=(0.9,),
    )
    run_dir = write_report(results, tmp_path, data_version="abc123", scope="CA", horizon=HORIZON)
    parent, children = backtest_runs(
        results, config=config, data_version="abc123", scope="CA", run_dir=run_dir
    )

    assert parent.params["forecast.horizon_days"] == "28"
    assert parent.tags["data_version"] == "abc123"
    assert parent.tags["code_version"]
    assert {path.name for path in parent.artifacts} == {"results.parquet", "summary.md"}
    assert [child.name for child in children] == ["naive", "seasonal_naive"]

    naive = children[0]
    folds = results.filter(pl.col("model") == "naive")
    assert naive.metrics["wrmsse"] == pytest.approx(folds.get_column("wrmsse").mean())
    assert naive.metrics["coverage_q_0.9"] == pytest.approx(
        folds.get_column("coverage_q_0.9").mean()
    )
    fold_wrmsse = [(step, value) for key, value, step in naive.step_metrics if key == "fold/wrmsse"]
    assert fold_wrmsse == list(zip([1, 2], folds.sort("fold").get_column("wrmsse"), strict=True))
    assert naive.params["model"] == "naive"


def _use_tracking(monkeypatch: pytest.MonkeyPatch, uri: str) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("DF_MLFLOW_TRACKING_URI", uri)
    get_secrets.cache_clear()


@pytest.mark.usefixtures("ingested")
def test_cli_logs_to_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uri = (tmp_path / "mlruns").as_uri()
    _use_tracking(monkeypatch, uri)
    assert main(["--config", *_cli_config(tmp_path), "--models", "naive", "tsb"]) == 0

    client = MlflowClient(uri)
    experiment = client.get_experiment_by_name("demand-forecasting")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    parents = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
    children = [run for run in runs if "mlflow.parentRunId" in run.data.tags]
    assert [run.info.run_name for run in parents] == ["backtest CA"]
    assert sorted(run.info.run_name for run in children) == ["naive", "tsb"]
    assert all(run.info.status == "FINISHED" for run in runs)


@pytest.mark.usefixtures("ingested")
def test_cli_reports_tracking_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_tracking(monkeypatch, "unsupported-scheme://nowhere")
    assert main(["--config", *_cli_config(tmp_path), "--models", "naive"]) == 1
    # The backtest itself finished, so its report is still written.
    assert len(_summaries(tmp_path)) == 1
