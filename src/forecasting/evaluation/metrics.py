"""Point-forecast accuracy metrics over aligned ``actual`` and ``forecast`` columns."""

from __future__ import annotations

import math

import polars as pl


def _totals(frame: pl.DataFrame) -> tuple[float, float, float]:
    """Return (sum of absolute errors, sum of signed errors, sum of actuals)."""
    error = pl.col("forecast") - pl.col("actual")
    row = frame.select(
        error.abs().sum().alias("abs_error"),
        error.sum().alias("error"),
        pl.col("actual").sum().alias("actual"),
    ).row(0)
    return float(row[0]), float(row[1]), float(row[2])


def wape(frame: pl.DataFrame) -> float:
    """Weighted absolute percentage error: total absolute error over total actual demand.

    Unlike MAPE it stays defined on zero-demand days, because it divides once by the total
    rather than by each point. Returns NaN when total actual demand is zero.
    """
    abs_error, _, actual = _totals(frame)
    return abs_error / actual if actual else math.nan


def bias(frame: pl.DataFrame) -> float:
    """Net over-forecast (positive) or under-forecast (negative) as a share of actual demand.

    Returns NaN when total actual demand is zero.
    """
    _, error, actual = _totals(frame)
    return error / actual if actual else math.nan
