from datetime import date

import polars as pl
import pytest

from forecasting.features.calendar import CALENDAR_FEATURES, calendar_features
from tests.conftest import N_DAYS, M5Fixture


def test_one_row_per_date_with_expected_columns(m5: M5Fixture) -> None:
    features = calendar_features(m5.calendar, "CA")
    assert features.columns == ["date", "wm_yr_wk", *CALENDAR_FEATURES]
    assert features.height == N_DAYS
    assert features.get_column("date").is_unique().all()


def test_iso_weekday_day_and_month(m5: M5Fixture) -> None:
    first = calendar_features(m5.calendar, "CA").row(0, named=True)
    # 2011-01-29 was a Saturday: 6 in ISO numbering, where Monday is 1.
    assert first["date"] == date(2011, 1, 29)
    assert first["day_of_week"] == 6
    assert first["day_of_month"] == 29
    assert first["month"] == 1


@pytest.mark.parametrize("state", ["CA", "TX", "WI"])
def test_snap_comes_from_the_requested_state(m5: M5Fixture, state: str) -> None:
    features = calendar_features(m5.calendar, state)
    assert (
        features.get_column("snap").to_list() == m5.calendar.get_column(f"snap_{state}").to_list()
    )


def test_events(m5: M5Fixture) -> None:
    events = calendar_features(m5.calendar, "TX").filter(pl.col("event_name").is_not_null())
    assert events.select("event_name", "event_type").rows() == [("SuperBowl", "Sporting")]


def test_rejects_unknown_state(m5: M5Fixture) -> None:
    with pytest.raises(ValueError, match="state must be one of"):
        calendar_features(m5.calendar, "NY")
