"""Calendar features. All are known in advance, so they are valid for any forecast day."""

from __future__ import annotations

import polars as pl

from forecasting.data.schemas import STATES

CALENDAR_FEATURES = (
    "day_of_week",
    "day_of_month",
    "month",
    "snap",
    "event_name",
    "event_type",
)
CALENDAR_CATEGORICALS = ("event_name", "event_type")


def calendar_features(calendar: pl.DataFrame, state: str) -> pl.DataFrame:
    """One row per date: weekday, day of month, month, the state's SNAP flag, and events.

    ``wm_yr_wk`` is kept as the key for joining weekly prices. Only the first event of a
    day is used; M5 has a handful of days with two.
    """
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}, got {state!r}")
    date = pl.col("date")
    return calendar.select(
        "date",
        "wm_yr_wk",
        # 1 = Monday ... 7 = Sunday (ISO), rather than M5's own Saturday-first ``wday``.
        date.dt.weekday().cast(pl.Int8).alias("day_of_week"),
        date.dt.day().cast(pl.Int8).alias("day_of_month"),
        date.dt.month().cast(pl.Int8).alias("month"),
        pl.col(f"snap_{state}").alias("snap"),
        pl.col("event_name_1").alias("event_name"),
        pl.col("event_type_1").alias("event_type"),
    )
