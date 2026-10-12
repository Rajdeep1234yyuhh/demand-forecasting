from datetime import date, timedelta

import polars as pl
import pytest

from forecasting.data.dataset import Dataset
from forecasting.evaluation.backtest import run_backtest
from forecasting.models.base import Forecaster, NotFittedError
from forecasting.models.statistical import (
    TSB,
    CrostonSBA,
    SeasonalExponentialSmoothing,
    SimpleExponentialSmoothing,
)

START = date(2016, 1, 1)
DAYS = 30
SERIES = {
    # Launches on day 10, then sells 4 a day: the forecast level should be about 4.
    ("CA_1", "late_launch"): [0] * 10 + [4] * (DAYS - 10),
    ("CA_1", "never_sold"): [0] * DAYS,
    # A second store, to check stores are processed and combined.
    ("TX_1", "one_sale_at_end"): [0] * (DAYS - 1) + [6],
    # Sold once on day 0, then never again: the item has effectively stopped selling.
    ("TX_1", "went_dead"): [3] + [0] * (DAYS - 1),
}


def _history() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "id": [sid for (_, sid), values in SERIES.items() for _ in values],
            "store_id": [store for (store, _), values in SERIES.items() for _ in values],
            "date": [START + timedelta(days=i) for values in SERIES.values() for i in range(DAYS)],
            "sales": [value for values in SERIES.values() for value in values],
        },
        schema_overrides={"sales": pl.Int16},
    ).lazy()


def _by_series(forecasts: pl.DataFrame) -> dict[str, list[float]]:
    grouped = forecasts.sort("date").group_by("id", maintain_order=True).agg("forecast")
    return dict(grouped.iter_rows())


ALL_MODELS = [CrostonSBA(), TSB(), SimpleExponentialSmoothing(), SeasonalExponentialSmoothing(7)]


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda model: model.name)
def test_forecasts_every_series_for_the_whole_horizon(model: Forecaster) -> None:
    forecasts = model.fit(_history()).predict(5)
    assert forecasts.height == len(SERIES) * 5
    assert forecasts.get_column("date").min() == START + timedelta(days=DAYS)
    by_series = _by_series(forecasts)
    assert by_series["never_sold"] == [0.0] * 5


def test_leading_zeros_are_ignored() -> None:
    by_series = _by_series(SimpleExponentialSmoothing().fit(_history()).predict(3))
    assert by_series["late_launch"] == pytest.approx([4.0] * 3)


def test_croston_sba_applies_bias_correction() -> None:
    # Constant demand of 4 every day: Croston's estimate is 4, and the Syntetos-Boylan
    # correction multiplies it by (1 - alpha / 2) with alpha = 0.1.
    by_series = _by_series(CrostonSBA().fit(_history()).predict(2))
    assert by_series["late_launch"] == pytest.approx([3.8, 3.8])


def test_tsb_decays_dead_items_but_croston_does_not() -> None:
    croston = _by_series(CrostonSBA().fit(_history()).predict(1))["went_dead"][0]
    tsb = _by_series(TSB().fit(_history()).predict(1))["went_dead"][0]
    # Croston only updates on sales, so it keeps forecasting the old rate.
    assert croston == pytest.approx(0.95 * 3)
    # TSB lowers the probability of a sale on each of the 29 zero days.
    assert tsb < 0.2 * croston


def test_short_series_fall_back_to_mean_demand() -> None:
    # One active day is shorter than a 7-day season, so the model returns NaN.
    by_series = _by_series(SeasonalExponentialSmoothing(7).fit(_history()).predict(2))
    assert by_series["one_sale_at_end"] == [6.0, 6.0]


def test_must_fit_before_predict() -> None:
    with pytest.raises(NotFittedError):
        CrostonSBA().predict(3)


def test_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="season_length"):
        SeasonalExponentialSmoothing(0)
    with pytest.raises(ValueError, match="smoothing weights"):
        TSB(alpha_probability=0)


def test_runs_in_a_backtest(dataset: Dataset) -> None:
    results = run_backtest(dataset, ALL_MODELS, horizon=3, n_folds=2)
    assert results.height == 2 * len(ALL_MODELS)
    assert results.get_column("wrmsse").is_finite().all()
