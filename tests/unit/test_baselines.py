from datetime import date, timedelta

import polars as pl
import pytest

from forecasting.data.schemas import DataValidationError
from forecasting.models.base import Forecaster, NotFittedError, future_dates
from forecasting.models.baselines import (
    BASELINES,
    MovingAverage,
    Naive,
    SeasonalNaive,
    make_baseline,
)
from forecasting.utils.config import ForecastConfig

START = date(2016, 1, 1)
VALUES = {"a": [5, 1, 2, 3, 4, 6, 8, 0, 2, 4], "b": [0, 0, 0, 0, 0, 0, 1, 1, 2, 3]}


def _history() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "id": [sid for sid, values in VALUES.items() for _ in values],
            "date": [START + timedelta(days=i) for values in VALUES.values() for i in range(10)],
            "sales": [value for values in VALUES.values() for value in values],
        },
        schema_overrides={"sales": pl.Int16},
    ).lazy()


def _series(forecasts: pl.DataFrame, series_id: str) -> list[float]:
    return forecasts.filter(pl.col("id") == series_id).sort("date")["forecast"].to_list()


def test_future_dates() -> None:
    dates = future_dates(date(2016, 2, 28), 2)
    assert dates.get_column("date").to_list() == [date(2016, 2, 29), date(2016, 3, 1)]
    assert dates.get_column("step").to_list() == [1, 2]


def test_naive_repeats_last_value() -> None:
    forecasts = Naive().fit(_history()).predict(3)
    assert forecasts.columns == ["id", "date", "forecast"]
    assert _series(forecasts, "a") == [4.0, 4.0, 4.0]
    assert _series(forecasts, "b") == [3.0, 3.0, 3.0]
    assert forecasts.get_column("date").min() == START + timedelta(days=10)


def test_seasonal_naive_cycles_last_season() -> None:
    forecasts = SeasonalNaive(period=3).fit(_history()).predict(5)
    # Last three days of "a" are 0, 2, 4.
    assert _series(forecasts, "a") == [0.0, 2.0, 4.0, 0.0, 2.0]


def test_seasonal_naive_matches_weekday_a_week_earlier() -> None:
    forecasts = SeasonalNaive(period=7).fit(_history()).predict(7)
    assert _series(forecasts, "a") == [float(value) for value in VALUES["a"][-7:]]


def test_moving_average() -> None:
    forecasts = MovingAverage(window=4).fit(_history()).predict(2)
    assert _series(forecasts, "a") == pytest.approx([3.5, 3.5])
    assert _series(forecasts, "b") == pytest.approx([1.75, 1.75])


def test_requires_full_window() -> None:
    with pytest.raises(ValueError, match="needs 28 days"):
        MovingAverage(window=28).fit(_history())


def test_requires_every_series_every_day() -> None:
    last_day = START + timedelta(days=9)
    gappy = _history().filter(~((pl.col("id") == "a") & (pl.col("date") == last_day)))
    with pytest.raises(ValueError, match="every series"):
        MovingAverage(window=3).fit(gappy)


def test_must_fit_before_use() -> None:
    with pytest.raises(NotFittedError):
        Naive().predict(3)
    with pytest.raises(NotFittedError):
        _ = Naive().recent


def test_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="horizon"):
        Naive().fit(_history()).predict(0)
    with pytest.raises(ValueError, match="window"):
        MovingAverage(window=0)
    with pytest.raises(ValueError, match="empty"):
        Naive().fit(_history().filter(pl.lit(value=False)))


class _Negative(Forecaster):
    name = "negative"

    def _fit(self, history: pl.LazyFrame, cutoff: date) -> None:
        return None

    def _predict(self, cutoff: date, horizon: int) -> pl.DataFrame:
        return future_dates(cutoff, horizon).with_columns(
            pl.lit("a").alias("id"), pl.lit(-1.0).alias("forecast")
        )


def test_forecasts_are_validated() -> None:
    with pytest.raises(DataValidationError, match="negative forecasts"):
        _Negative().fit(_history()).predict(2)


def test_make_baseline() -> None:
    config = ForecastConfig(
        horizon_days=28,
        quantiles=(0.5,),
        backtest_folds=3,
        seasonal_period=7,
        moving_average_window=14,
    )
    models = {name: make_baseline(name, config) for name in BASELINES}
    assert {name: model.name for name, model in models.items()} == {n: n for n in BASELINES}
    seasonal, average = models["seasonal_naive"], models["moving_average"]
    assert isinstance(seasonal, SeasonalNaive)
    assert seasonal.window == 7
    assert isinstance(average, MovingAverage)
    assert average.window == 14
    with pytest.raises(ValueError, match="unknown baseline"):
        make_baseline("prophet", config)
