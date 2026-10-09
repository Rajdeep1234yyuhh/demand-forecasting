"""Exploratory analysis of long-format sales data.

Functions accept eager or lazy frames and return small, eager summaries ready to plot.
Queries run on Polars' streaming engine, so full-dataset scans stay within modest memory.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

Frame = pl.DataFrame | pl.LazyFrame

# Syntetos-Boylan cut-offs for demand classification.
ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49
_STATE_SNAP_COLUMNS = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}


def hierarchy_counts(sales: Frame) -> dict[str, int]:
    """Number of distinct members at each level of the hierarchy.

    Deduplicates to one row per series first: ``unique`` streams, whereas ``n_unique`` over
    the full table would materialize every row.
    """
    levels = ["state_id", "store_id", "cat_id", "dept_id", "item_id", "id"]
    series = sales.lazy().select(levels).unique().collect(engine="streaming")
    return {level: series.get_column(level).n_unique() for level in levels}


def zero_share(sales: Frame) -> float:
    """Share of series-days with zero units sold."""
    return float(
        sales.lazy().select((pl.col("sales") == 0).mean()).collect(engine="streaming").item()
    )


def trim_leading_zeros(sales: Frame) -> pl.LazyFrame:
    """Drop each series' rows before its first sale, i.e. before the item was stocked.

    Pre-launch zeros are not demand; leaving them in inflates intermittency.
    Series that never sell are dropped entirely.
    """
    lazy = sales.lazy()
    first_sale = (
        lazy.filter(pl.col("sales") > 0)
        .group_by("id")
        .agg(pl.col("date").min().alias("first_sale"))
    )
    return (
        lazy.join(first_sale, on="id", how="inner")
        .filter(pl.col("date") >= pl.col("first_sale"))
        .drop("first_sale")
    )


def demand_classification(
    sales: Frame, *, adi_cutoff: float = ADI_CUTOFF, cv2_cutoff: float = CV2_CUTOFF
) -> pl.DataFrame:
    """Classify each series as smooth, erratic, intermittent, or lumpy.

    ADI is the average interval between non-zero sales; CV2 is the squared coefficient of
    variation of non-zero sale sizes. Both are computed from the first sale onwards.
    """
    nonzero = pl.col("sales").filter(pl.col("sales") > 0)
    adi, cv2 = pl.col("adi"), pl.col("cv2")
    return (
        trim_leading_zeros(sales)
        .group_by("id")
        .agg(
            pl.len().alias("periods"),
            (pl.col("sales") > 0).sum().alias("nonzero_periods"),
            nonzero.mean().alias("mean_size"),
            nonzero.std().fill_null(0.0).alias("std_size"),
        )
        .with_columns(
            (pl.col("periods") / pl.col("nonzero_periods")).alias("adi"),
            ((pl.col("std_size") / pl.col("mean_size")) ** 2).alias("cv2"),
        )
        .with_columns(
            pl.when((adi < adi_cutoff) & (cv2 < cv2_cutoff))
            .then(pl.lit("smooth"))
            .when(adi < adi_cutoff)
            .then(pl.lit("erratic"))
            .when(cv2 < cv2_cutoff)
            .then(pl.lit("intermittent"))
            .otherwise(pl.lit("lumpy"))
            .alias("demand_class")
        )
        .select("id", "periods", "nonzero_periods", "adi", "cv2", "demand_class")
        .sort("id")
        .collect(engine="streaming")
    )


def daily_totals(sales: Frame, by: Sequence[str] = ()) -> pl.DataFrame:
    """Total units per day, optionally split by hierarchy columns."""
    keys = ["date", *by]
    return (
        sales.lazy()
        .group_by(keys)
        .agg(pl.col("sales").sum().cast(pl.Int64))
        .sort(keys)
        .collect(engine="streaming")
    )


def weekday_profile(sales: Frame, by: Sequence[str] = ()) -> pl.DataFrame:
    """Mean daily units by weekday (1 = Monday), with an index where 1.0 is the group mean."""
    keys = [*by, "weekday"]
    profile = (
        daily_totals(sales, by)
        .lazy()
        .with_columns(pl.col("date").dt.weekday().alias("weekday"))
        .group_by(keys)
        .agg(pl.col("sales").mean().alias("mean_sales"))
    )
    group_mean = pl.col("mean_sales").mean()
    if by:
        group_mean = group_mean.over(by)
    return (
        profile.with_columns((pl.col("mean_sales") / group_mean).alias("index"))
        .sort(keys)
        .collect(engine="streaming")
    )


def snap_lift(sales: Frame, calendar: Frame, category: str | None = "FOODS") -> pl.DataFrame:
    """Naive SNAP effect per state: mean daily units on SNAP days vs other days, minus one.

    SNAP days fall in the first half of each month, so this lift is confounded with
    month-position effects. Treat it as a hypothesis for modeling, not a causal estimate.
    """
    lazy = sales.lazy()
    if category is not None:
        lazy = lazy.filter(pl.col("cat_id") == category)
    daily = lazy.group_by("state_id", "date").agg(pl.col("sales").sum().cast(pl.Int64))

    snap = pl.lit(None, dtype=pl.Int8)
    for state, column in _STATE_SNAP_COLUMNS.items():
        snap = pl.when(pl.col("state_id") == state).then(pl.col(column)).otherwise(snap)

    joined = daily.join(
        calendar.lazy().select("date", *_STATE_SNAP_COLUMNS.values()), on="date"
    ).with_columns(snap.alias("snap"))
    return (
        joined.group_by("state_id")
        .agg(
            pl.col("sales").filter(pl.col("snap") == 1).mean().alias("mean_snap"),
            pl.col("sales").filter(pl.col("snap") == 0).mean().alias("mean_non_snap"),
        )
        .with_columns((pl.col("mean_snap") / pl.col("mean_non_snap") - 1).alias("lift"))
        .sort("state_id")
        .collect(engine="streaming")
    )


def price_change_share(prices: Frame) -> float:
    """Share of store-item pairs whose price changes at least once over the history."""
    query = (
        prices.lazy()
        .group_by("store_id", "item_id")
        .agg((pl.col("sell_price").n_unique() > 1).alias("changed"))
        .select(pl.col("changed").mean())
    )
    return float(query.collect(engine="streaming").item())
