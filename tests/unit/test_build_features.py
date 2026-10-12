from datetime import timedelta

import polars as pl
import pytest

from forecasting.data.ingest import to_long
from forecasting.features.build import (
    CATEGORICAL_FEATURES,
    KEY_COLUMNS,
    build_features,
    check_consecutive_days,
    feature_columns,
    make_panel,
    split_rows,
)
from forecasting.utils.config import FeatureConfig
from tests.conftest import N_DAYS, SERIES, START, M5Fixture

HORIZON = 3
CUTOFF = START + timedelta(days=N_DAYS - 1 - HORIZON)
CONFIG = FeatureConfig(lags=(3, 4), rolling_windows=(2, 3))


@pytest.fixture
def history(m5: M5Fixture) -> pl.DataFrame:
    return to_long(m5.sales, m5.day_columns, m5.calendar)


def _build(m5: M5Fixture, panel: pl.DataFrame, state: str = "CA") -> pl.DataFrame:
    return build_features(panel, m5.calendar, m5.prices, state=state, config=CONFIG, shift=HORIZON)


def test_make_panel(history: pl.DataFrame) -> None:
    panel = make_panel(history, CUTOFF, HORIZON)
    # N_DAYS - HORIZON known days up to the cutoff, plus HORIZON future days.
    assert panel.height == len(SERIES) * N_DAYS
    future = panel.filter(pl.col("date") > CUTOFF)
    assert future.height == len(SERIES) * HORIZON
    assert future.get_column("sales").null_count() == future.height
    # Known sales after the cutoff were in the input but must not survive.
    assert panel.filter(pl.col("date") <= CUTOFF).get_column("sales").null_count() == 0
    assert panel.get_column("sales").dtype == pl.Float32


def test_make_panel_rejects_bad_input(history: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="horizon"):
        make_panel(history, CUTOFF, 0)
    with pytest.raises(ValueError, match="no history"):
        make_panel(history, START - timedelta(days=1), HORIZON)


def test_columns_and_types(m5: M5Fixture, history: pl.DataFrame) -> None:
    features = _build(m5, make_panel(history, CUTOFF, HORIZON))
    assert features.columns == [*KEY_COLUMNS, *feature_columns(CONFIG)]
    for name in feature_columns(CONFIG):
        expected = pl.String if name in CATEGORICAL_FEATURES else pl.Float32
        assert features.schema[name] == expected, name
    assert features.height == len(SERIES) * N_DAYS


def test_forecast_rows_match_rows_built_later_with_actuals(
    m5: M5Fixture, history: pl.DataFrame
) -> None:
    """Train/serve consistency: a day's features don't change once its sales are known."""
    at_forecast_time = _build(m5, make_panel(history, CUTOFF, HORIZON))
    last_day = START + timedelta(days=N_DAYS - 1)
    later = _build(m5, make_panel(history, last_day, HORIZON))
    columns = feature_columns(CONFIG)
    future_days = pl.col("date").is_between(CUTOFF + timedelta(days=1), last_day)
    expected = later.filter(future_days).select("id", "date", *columns)
    assert at_forecast_time.filter(future_days).select("id", "date", *columns).equals(expected)


def test_price_and_release_features(m5: M5Fixture, history: pl.DataFrame) -> None:
    features = _build(m5, make_panel(history, CUTOFF, HORIZON))
    first_day = features.filter(pl.col("date") == START)
    foods = first_day.filter(pl.col("id") == "FOODS_1_001_CA_1_evaluation").row(0, named=True)
    assert foods["sell_price"] == pytest.approx(2.5)
    assert foods["days_since_release"] == 0
    # The unsold TX_1 hobby item has no prices at all, so it is never on the shelf.
    never = features.filter(pl.col("id") == "HOBBIES_1_001_TX_1_evaluation")
    assert never.get_column("sell_price").null_count() == never.height
    assert never.get_column("days_since_release").null_count() == never.height


def test_split_rows(m5: M5Fixture, history: pl.DataFrame) -> None:
    features = _build(m5, make_panel(history, CUTOFF, HORIZON))
    training, forecast = split_rows(features, CUTOFF)
    assert forecast.height == len(SERIES) * HORIZON
    assert training.get_column("date").max() == CUTOFF
    assert training.get_column("sales").null_count() == 0
    # Only items that were on the shelf train; the never-priced item does not.
    assert "HOBBIES_1_001_TX_1_evaluation" not in training.get_column("id").to_list()


def test_requires_one_row_per_day(m5: M5Fixture, history: pl.DataFrame) -> None:
    panel = make_panel(history, CUTOFF, HORIZON)
    gap = panel.filter(pl.col("date") != START + timedelta(days=2))
    with pytest.raises(ValueError, match="4 series do not have exactly one row per day"):
        _build(m5, gap)
    duplicate = pl.concat([panel, panel.head(1)])
    with pytest.raises(ValueError, match="1 series do not"):
        check_consecutive_days(duplicate)
