from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from forecasting.data.ingest import run_ingest
from forecasting.data.schemas import DataValidationError
from forecasting.evaluation.backtest import (
    Dataset,
    main,
    make_folds,
    run_backtest,
    summarize,
    write_report,
)
from forecasting.models.base import Forecaster
from forecasting.models.baselines import MovingAverage, Naive, SeasonalNaive
from forecasting.utils.config import DataConfig
from tests.conftest import N_DAYS, START, M5Fixture

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


@pytest.fixture
def ingested(m5: M5Fixture, data_config: DataConfig) -> Path:
    m5.write(data_config.raw_dir)
    run_ingest(data_config)
    return data_config.processed_dir


@pytest.fixture
def dataset(ingested: Path) -> Dataset:
    return Dataset.load(ingested)


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
    assert "Scope: all states" in summary.read_text("utf-8")


@pytest.mark.usefixtures("ingested")
def test_cli_defaults_to_training_subset(tmp_path: Path) -> None:
    # base.yaml sets data.train_subset_state to CA.
    assert main(["--config", *_cli_config(tmp_path), "--models", "naive"]) == 0
    [summary] = _summaries(tmp_path)
    assert "Scope: CA" in summary.read_text("utf-8")


def test_cli_failure_returns_nonzero(tmp_path: Path) -> None:
    assert main(["--config", *_cli_config(tmp_path)]) == 1
