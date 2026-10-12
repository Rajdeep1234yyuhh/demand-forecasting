"""Sales-history features: lags, rolling statistics, and days since the last sale.

Every feature reads sales at least ``shift`` days before the row's own date. With ``shift``
set to the forecast horizon, all features of every forecast row are computable on the
forecast date, so one model covers the whole horizon without feeding its own forecasts
back in as inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def sales_feature_names(lags: Sequence[int], windows: Sequence[int]) -> list[str]:
    """Names of the features ``sales_features`` produces, in the same order."""
    return [
        *(f"lag_{lag}" for lag in lags),
        *(f"rolling_mean_{window}" for window in windows),
        *(f"rolling_std_{window}" for window in windows),
        "days_since_sale",
    ]


def sales_features(*, lags: Sequence[int], windows: Sequence[int], shift: int) -> list[pl.Expr]:
    """Expressions computing each series' sales features, in date order.

    They assume one row per series per day with no gaps, so that shifting by ``k`` rows
    means ``k`` days back; ``build_features`` checks this before using them.
    """
    if shift <= 0:
        raise ValueError("shift must be positive")
    short = [lag for lag in lags if lag < shift]
    if short:
        raise ValueError(f"lags {short} are shorter than the shift of {shift} days")

    def per_series(expr: pl.Expr) -> pl.Expr:
        return expr.over("id", order_by="date")

    sales = pl.col("sales")
    known = sales.shift(shift)
    last_sale_date = pl.when(sales > 0).then(pl.col("date")).forward_fill().shift(shift)
    expressions = [
        *(per_series(sales.shift(lag)) for lag in lags),
        *(per_series(known.rolling_mean(window)) for window in windows),
        *(per_series(known.rolling_std(window)) for window in windows),
        per_series((pl.col("date") - last_sale_date).dt.total_days()),
    ]
    names = sales_feature_names(lags, windows)
    return [expr.alias(name) for expr, name in zip(expressions, names, strict=True)]
