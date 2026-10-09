from datetime import date, timedelta

import polars as pl
import pytest

from forecasting.analysis.eda import (
    daily_totals,
    demand_classification,
    hierarchy_counts,
    price_change_share,
    snap_lift,
    trim_leading_zeros,
    weekday_profile,
    zero_share,
)
from forecasting.data.ingest import to_long
from tests.conftest import N_DAYS, M5Fixture


def _long(series: dict[str, list[int]]) -> pl.DataFrame:
    start = date(2011, 1, 29)
    return pl.DataFrame(
        {
            "id": [sid for sid, values in series.items() for _ in values],
            "date": [
                start + timedelta(days=i) for values in series.values() for i in range(len(values))
            ],
            "sales": [v for values in series.values() for v in values],
        }
    )


@pytest.fixture
def sales(m5: M5Fixture) -> pl.DataFrame:
    return to_long(m5.sales, m5.day_columns, m5.calendar)


def test_hierarchy_counts(sales: pl.DataFrame) -> None:
    counts = hierarchy_counts(sales)
    assert counts == {
        "state_id": 2,
        "store_id": 2,
        "cat_id": 2,
        "dept_id": 2,
        "item_id": 2,
        "id": 4,
    }


def test_zero_share() -> None:
    assert zero_share(_long({"a": [0, 1, 0, 2]})) == 0.5


def test_trim_leading_zeros_drops_prelaunch_and_unsold() -> None:
    trimmed = trim_leading_zeros(_long({"a": [0, 0, 3, 0], "never": [0, 0, 0, 0]})).collect()
    assert trimmed.get_column("sales").to_list() == [3, 0]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([3, 2, 4, 3, 5, 2, 3, 4], "smooth"),
        ([1, 10, 1, 10, 1, 10], "erratic"),
        ([0, 0, 1, 0, 0, 2, 0, 0, 0, 1], "intermittent"),
        ([0, 0, 10, 0, 0, 1, 0, 0, 10, 0, 0, 1], "lumpy"),
    ],
)
def test_demand_classification(values: list[int], expected: str) -> None:
    result = demand_classification(_long({"s": values}))
    assert result.get_column("demand_class").item() == expected


def test_demand_classification_metrics() -> None:
    result = demand_classification(_long({"s": [0, 0, 1, 0, 0, 2, 0, 0, 0, 1]})).row(0, named=True)
    assert result["periods"] == 8
    assert result["nonzero_periods"] == 3
    assert result["adi"] == pytest.approx(8 / 3)
    assert result["cv2"] == pytest.approx((0.57735 / (4 / 3)) ** 2, rel=1e-4)


def test_single_sale_has_zero_cv2() -> None:
    result = demand_classification(_long({"s": [0, 5, 0]}))
    assert result.get_column("cv2").item() == 0.0


def test_daily_totals(sales: pl.DataFrame) -> None:
    totals = daily_totals(sales)
    assert totals.height == N_DAYS
    assert totals.get_column("sales").sum() == sales.get_column("sales").sum()
    by_state = daily_totals(sales, by=["state_id"])
    assert by_state.height == N_DAYS * 2


def test_weekday_profile_index_averages_to_one(sales: pl.DataFrame) -> None:
    overall = weekday_profile(sales)
    assert overall.height == 7
    assert overall.get_column("index").mean() == pytest.approx(1.0)
    by_state = weekday_profile(sales, by=["state_id"])
    means = by_state.group_by("state_id").agg(pl.col("index").mean())
    assert means.get_column("index").to_list() == pytest.approx([1.0, 1.0])


def test_snap_lift(sales: pl.DataFrame, m5: M5Fixture) -> None:
    lift = snap_lift(sales, m5.calendar).row(0, named=True)
    assert lift["state_id"] == "CA"
    # CA FOODS sales on SNAP days (offsets 2-4) vs the other 11 days.
    snap = [4, 3, 5]
    other = [3, 2, 2, 3, 4, 3, 2, 4, 3, 5, 3]
    assert lift["mean_snap"] == pytest.approx(sum(snap) / 3)
    assert lift["lift"] == pytest.approx((sum(snap) / 3) / (sum(other) / 11) - 1)
    assert snap_lift(sales, m5.calendar, category=None).height == 2


def test_price_change_share(m5: M5Fixture) -> None:
    # CA_1 HOBBIES changes price between weeks; CA_1 FOODS and TX_1 FOODS do not.
    assert price_change_share(m5.prices) == pytest.approx(1 / 3)
