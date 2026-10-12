import math
from datetime import timedelta

import polars as pl
import pytest

from forecasting.data.dataset import Dataset
from forecasting.evaluation.backtest import run_backtest
from forecasting.features.build import feature_columns
from forecasting.models.base import NotFittedError
from forecasting.models.lgbm import CategoryEncoder, LightGBMForecaster
from forecasting.utils.config import FeatureConfig, LightGBMConfig
from tests.conftest import START

HORIZON = 3
CUTOFF = START + timedelta(days=9)
FEATURES = FeatureConfig(lags=(3, 4), rolling_windows=(2, 3))
# Small and fast: the fixture has only a few dozen training rows.
PARAMS = LightGBMConfig(
    objective="tweedie",
    tweedie_variance_power=1.1,
    learning_rate=0.3,
    num_leaves=4,
    min_data_in_leaf=1,
    feature_fraction=1.0,
    bagging_fraction=1.0,
    bagging_freq=0,
    lambda_l2=0.0,
    num_boost_round=5,
    num_threads=1,
    seed=1,
)


def _model(dataset: Dataset, prices: pl.LazyFrame | None = None) -> LightGBMForecaster:
    return LightGBMForecaster(
        PARAMS,
        FEATURES,
        calendar=dataset.calendar,
        prices=dataset.prices if prices is None else prices,
        horizon=HORIZON,
    )


def _history(dataset: Dataset) -> pl.LazyFrame:
    return dataset.sales.filter(pl.col("date") <= CUTOFF)


def test_forecasts_every_series_for_the_horizon(dataset: Dataset) -> None:
    forecasts = _model(dataset).fit(_history(dataset)).predict(HORIZON)
    assert forecasts.height == 4 * HORIZON
    assert forecasts.get_column("date").min() == CUTOFF + timedelta(days=1)
    assert (forecasts.get_column("forecast") >= 0).all()


def test_items_off_the_shelf_are_forecast_as_zero(dataset: Dataset) -> None:
    forecasts = _model(dataset).fit(_history(dataset)).predict(HORIZON)
    never_priced = forecasts.filter(pl.col("id") == "HOBBIES_1_001_TX_1_evaluation")
    assert never_priced.get_column("forecast").to_list() == [0.0] * HORIZON
    on_shelf = forecasts.filter(pl.col("id") == "FOODS_1_001_CA_1_evaluation")
    assert (on_shelf.get_column("forecast") > 0).all()


def test_training_is_deterministic(dataset: Dataset) -> None:
    first = _model(dataset).fit(_history(dataset)).predict(HORIZON)
    second = _model(dataset).fit(_history(dataset)).predict(HORIZON)
    assert first.equals(second)


def test_shorter_horizons_are_a_prefix(dataset: Dataset) -> None:
    model = _model(dataset).fit(_history(dataset))
    full = model.predict(HORIZON)
    assert model.predict(1).equals(full.filter(pl.col("date") == CUTOFF + timedelta(days=1)))


def test_rejects_horizon_beyond_the_feature_shift(dataset: Dataset) -> None:
    model = _model(dataset).fit(_history(dataset))
    with pytest.raises(ValueError, match="3-day horizon"):
        model.predict(HORIZON + 1)


def test_feature_importance(dataset: Dataset) -> None:
    importance = _model(dataset).fit(_history(dataset)).feature_importance()
    assert set(importance.get_column("store_id")) == {"CA_1", "TX_1"}
    for store in ("CA_1", "TX_1"):
        rows = importance.filter(pl.col("store_id") == store)
        assert sorted(rows.get_column("feature")) == sorted(feature_columns(FEATURES))
        total = rows.get_column("share").sum()
        assert total == pytest.approx(1.0) or total == 0.0


def test_must_fit_first(dataset: Dataset) -> None:
    model = _model(dataset)
    with pytest.raises(NotFittedError):
        model.predict(HORIZON)
    with pytest.raises(NotFittedError):
        model.feature_importance()


def test_store_without_shelf_history_fails_clearly(dataset: Dataset) -> None:
    no_prices = dataset.prices.filter(pl.lit(value=False))
    with pytest.raises(ValueError, match="no training rows"):
        _model(dataset, prices=no_prices).fit(_history(dataset))


def test_rejects_invalid_horizon(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="horizon"):
        LightGBMForecaster(
            PARAMS, FEATURES, calendar=dataset.calendar, prices=dataset.prices, horizon=0
        )


def test_runs_in_a_backtest(dataset: Dataset) -> None:
    results = run_backtest(dataset, [_model(dataset)], horizon=HORIZON, n_folds=2)
    assert results.height == 2
    assert results.get_column("wrmsse").is_finite().all()


def test_category_encoder() -> None:
    training = pl.DataFrame({"cat": ["b", "a", None, "b"], "x": [1, 2, 3, 4]})
    encoder = CategoryEncoder.fit(training, ["cat"])
    assert encoder.vocabularies == {"cat": ("a", "b")}
    unseen = pl.DataFrame({"cat": ["a", "c", None, "b"], "x": [5, 6, 7, 8]})
    matrix = encoder.matrix(unseen, ["x", "cat"])
    assert matrix.dtype.name == "float32"
    assert matrix[:, 0].tolist() == [5.0, 6.0, 7.0, 8.0]
    codes = matrix[:, 1].tolist()
    assert codes[0] == 0.0
    assert math.isnan(codes[1])
    assert math.isnan(codes[2])
    assert codes[3] == 1.0
