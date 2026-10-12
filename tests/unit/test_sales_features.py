from collections.abc import Mapping, Sequence
from datetime import date, timedelta

import polars as pl
import pytest

from forecasting.features.lags import sales_feature_names, sales_features

START = date(2016, 1, 1)
SALES = [3.0, 0.0, 0.0, 5.0, 1.0, 0.0, 2.0, 0.0, 4.0, 0.0]


def _frame(series: Mapping[str, Sequence[float | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [sid for sid, values in series.items() for _ in values],
            "date": [
                START + timedelta(days=i) for values in series.values() for i in range(len(values))
            ],
            "sales": [value for values in series.values() for value in values],
        },
        schema={"id": pl.String, "date": pl.Date, "sales": pl.Float32},
    )


def _features(frame: pl.DataFrame, *, shift: int = 2) -> pl.DataFrame:
    exprs = sales_features(lags=[2, 3], windows=[3], shift=shift)
    return frame.with_columns(exprs).sort("id", "date")


def test_names_match_expressions() -> None:
    exprs = sales_features(lags=[28, 35], windows=[7, 28], shift=28)
    names = [expr.meta.output_name() for expr in exprs]
    assert names == sales_feature_names([28, 35], [7, 28])
    assert names == [
        "lag_28",
        "lag_35",
        "rolling_mean_7",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_28",
        "days_since_sale",
    ]


def test_lags() -> None:
    features = _features(_frame({"a": SALES}))
    assert features.get_column("lag_2").to_list() == [None, None, *SALES[:-2]]
    assert features.get_column("lag_3").to_list() == [None, None, None, *SALES[:-3]]


def test_rolling_statistics_use_only_shifted_sales() -> None:
    features = _features(_frame({"a": SALES}))
    # Row 5 (shift 2, window 3) sees days 1-3: 0, 0, 5.
    row = features.row(5, named=True)
    assert row["rolling_mean_3"] == pytest.approx(5 / 3)
    assert row["rolling_std_3"] == pytest.approx(pl.Series([0.0, 0.0, 5.0]).std())
    # Not enough shifted history yet.
    assert features.get_column("rolling_mean_3").to_list()[:4] == [None] * 4


def test_days_since_sale() -> None:
    features = _features(_frame({"a": SALES}))
    # Row 6 (2016-01-07) looks at day 4 and earlier; the last sale was day 4 (index 4).
    assert features.row(6, named=True)["days_since_sale"] == 2
    # Row 3 looks at day 1 and earlier; the last sale was day 0.
    assert features.row(3, named=True)["days_since_sale"] == 3
    # No sale on or before the reference day yet.
    assert features.row(1, named=True)["days_since_sale"] is None


def test_series_do_not_mix_and_order_does_not_matter() -> None:
    frame = _frame({"a": SALES, "b": [9.0] * len(SALES)})
    expected = _features(frame)
    lags_of_b = expected.filter(pl.col("id") == "b").get_column("lag_2").drop_nulls()
    assert lags_of_b.unique().to_list() == [9.0]
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=3)
    assert _features(shuffled).equals(expected)


def test_future_rows_only_need_known_sales() -> None:
    """With the shift equal to the horizon, unknown future sales never feed a feature."""
    horizon = 2
    known, future = SALES[:-horizon], [None] * horizon
    at_forecast_time = _features(_frame({"a": [*known, *future]}), shift=horizon)
    with_actuals = _features(_frame({"a": SALES}), shift=horizon)
    feature_cols = sales_feature_names([2, 3], [3])
    assert at_forecast_time.select(feature_cols).equals(with_actuals.select(feature_cols))


@pytest.mark.parametrize(
    ("lags", "shift", "message"), [([1, 3], 2, "shorter"), ([3], 0, "positive")]
)
def test_rejects_leaky_or_invalid_settings(lags: list[int], shift: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sales_features(lags=lags, windows=[3], shift=shift)
