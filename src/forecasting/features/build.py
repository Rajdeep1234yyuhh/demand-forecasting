"""Assemble model-ready feature tables.

A panel holds each series' sales up to a cutoff, followed by ``horizon`` future days whose
sales are unknown (null). ``build_features`` turns the whole panel into features in one
pass, so training rows and forecast rows come from the same code and the same data.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from forecasting.features.calendar import (
    CALENDAR_CATEGORICALS,
    CALENDAR_FEATURES,
    calendar_features,
)
from forecasting.features.lags import sales_feature_names, sales_features
from forecasting.features.price import PRICE_FEATURES, price_features
from forecasting.utils.config import FeatureConfig

PANEL_COLUMNS = ("id", "item_id", "dept_id", "cat_id", "store_id", "date", "sales")
KEY_COLUMNS = ("id", "store_id", "date", "sales")
CATEGORICAL_FEATURES = ("item_id", "dept_id", "cat_id", *CALENDAR_CATEGORICALS)

# An item is on the shelf in weeks it has a price; M5 has no price before an item's release.
ON_SHELF = pl.col("sell_price").is_not_null()


def feature_columns(config: FeatureConfig) -> list[str]:
    """Every feature column, categorical ones first, in a fixed order."""
    numeric_calendar = [name for name in CALENDAR_FEATURES if name not in CALENDAR_CATEGORICALS]
    return [
        *CATEGORICAL_FEATURES,
        *numeric_calendar,
        *PRICE_FEATURES,
        "days_since_release",
        *sales_feature_names(config.lags, config.rolling_windows),
    ]


def make_panel(history: pl.DataFrame, cutoff: date, horizon: int) -> pl.DataFrame:
    """Sales up to ``cutoff`` plus ``horizon`` future days per series, with null sales.

    Rows after ``cutoff`` in ``history`` are dropped, so later sales can never leak in.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    known = history.filter(pl.col("date") <= cutoff).select(
        *PANEL_COLUMNS[:-1], pl.col("sales").cast(pl.Float32)
    )
    if known.is_empty():
        raise ValueError(f"no history on or before {cutoff}")
    future_dates = pl.date_range(
        cutoff + timedelta(days=1), cutoff + timedelta(days=horizon), eager=True
    ).alias("date")
    future = (
        known.select(PANEL_COLUMNS[:-2])
        .unique()
        .join(future_dates.to_frame(), how="cross")
        .with_columns(pl.lit(None, dtype=pl.Float32).alias("sales"))
    )
    return pl.concat([known, future]).sort("id", "date")


def check_consecutive_days(panel: pl.DataFrame) -> None:
    """Raise unless every series has exactly one row per day between its first and last."""
    span = (pl.col("date").max() - pl.col("date").min()).dt.total_days() + 1
    broken = (
        panel.group_by("id")
        .agg((span != pl.len()).alias("gap"), pl.col("date").is_duplicated().any().alias("dup"))
        .filter(pl.col("gap") | pl.col("dup"))
    )
    if not broken.is_empty():
        example = broken.sort("id").item(0, "id")
        raise ValueError(
            f"{broken.height} series do not have exactly one row per day, e.g. {example}"
        )


def build_features(
    panel: pl.DataFrame,
    calendar: pl.DataFrame,
    prices: pl.DataFrame | pl.LazyFrame,
    *,
    state: str,
    config: FeatureConfig,
    shift: int,
) -> pl.DataFrame:
    """Return the panel's keys and every feature, one row per series per day.

    Categorical features stay as strings (encoding is the model's job); numeric features
    are Float32 to halve memory. ``shift`` is the forecast horizon.
    """
    check_consecutive_days(panel)
    columns = feature_columns(config)
    numeric = [name for name in columns if name not in CATEGORICAL_FEATURES]
    released_on = pl.col("date").filter(ON_SHELF).min().over("id")
    return (
        panel.lazy()
        .join(calendar_features(calendar, state).lazy(), on="date", how="left")
        .join(price_features(prices).lazy(), on=["store_id", "item_id", "wm_yr_wk"], how="left")
        .with_columns(
            *sales_features(lags=config.lags, windows=config.rolling_windows, shift=shift),
            (pl.col("date") - released_on).dt.total_days().alias("days_since_release"),
        )
        .select(
            *KEY_COLUMNS,
            *CATEGORICAL_FEATURES,
            *(pl.col(name).cast(pl.Float32) for name in numeric),
        )
        .sort("id", "date")
        .collect()
    )


def split_rows(features: pl.DataFrame, cutoff: date) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split into training rows and forecast rows.

    Training rows are known sales on or before ``cutoff`` while the item was on the shelf;
    days before release are not demand, so they are left out. Forecast rows are every day
    after ``cutoff``.
    """
    training = features.filter(
        (pl.col("date") <= cutoff) & ON_SHELF & pl.col("sales").is_not_null()
    )
    forecast = features.filter(pl.col("date") > cutoff)
    return training, forecast
