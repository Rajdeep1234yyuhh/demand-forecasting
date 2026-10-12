"""Quantile forecasts from any point forecast, using a negative binomial count distribution.

Daily unit sales are counts, mostly small, and burstier than a Poisson distribution allows.
The negative binomial adds one parameter per series, the size ``r``:

    variance = mean + mean**2 / r

A small ``r`` means bursty sales; as ``r`` grows the distribution approaches Poisson.

Each series' size is estimated by the method of moments over its last ``window_days`` of
training data. The mean comes from whichever point model is used, so every model gets
quantiles without retraining, under the same uncertainty assumptions. Quantiles never cross,
and the distribution also yields demand over a lead time (Phase 4), which separate daily
quantile models cannot: daily quantiles don't add up.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Self

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import stats

from forecasting.models.base import NotFittedError

FloatArray = npt.NDArray[np.float64]
# A variance needs at least two observations.
MIN_WINDOW_DAYS = 2


def quantile_column(level: float) -> str:
    """Column name for a quantile level, e.g. ``q_0.9``."""
    return f"q_{level:g}"


def _quantiles(level: float, mean: FloatArray, size: FloatArray) -> FloatArray:
    """Quantile ``level`` per row: Poisson where ``size`` is NaN, negative binomial elsewhere."""
    result = np.zeros_like(mean)
    positive = mean > 0
    poisson = positive & np.isnan(size)
    negative_binomial = positive & ~np.isnan(size)
    result[poisson] = stats.poisson.ppf(level, mean[poisson])
    r = size[negative_binomial]
    result[negative_binomial] = stats.nbinom.ppf(level, r, r / (r + mean[negative_binomial]))
    return result


class NegativeBinomialUncertainty:
    """Per-series dispersion learned from recent history, applied to any point forecast."""

    def __init__(self, window_days: int) -> None:
        if window_days < MIN_WINDOW_DAYS:
            raise ValueError(f"window_days must be at least {MIN_WINDOW_DAYS}")
        self.window_days = window_days
        self._sizes: pl.DataFrame | None = None

    @property
    def sizes(self) -> pl.DataFrame:
        """Size ``r`` per series; null where sales are not overdispersed (Poisson)."""
        if self._sizes is None:
            raise NotFittedError("negative binomial uncertainty must be fitted first")
        return self._sizes

    def fit(self, history: pl.LazyFrame) -> Self:
        """Estimate each series' size from the last ``window_days`` up to the latest date."""
        cutoff = history.select(pl.col("date").max()).collect(engine="streaming").item()
        if cutoff is None:
            raise ValueError("history is empty")
        start = cutoff - timedelta(days=self.window_days - 1)
        sales = pl.col("sales").cast(pl.Float64)
        mean, variance = pl.col("mean"), pl.col("variance")
        self._sizes = (
            history.filter(pl.col("date").is_between(start, cutoff))
            .group_by("id")
            .agg(sales.mean().alias("mean"), sales.var().alias("variance"))
            .select(
                "id",
                pl.when((variance > mean) & (mean > 0))
                .then(mean.pow(2) / (variance - mean))
                .alias("size"),
            )
            .sort("id")
            .collect(engine="streaming")
        )
        return self

    def quantiles(self, forecasts: pl.DataFrame, levels: Sequence[float]) -> pl.DataFrame:
        """Return ``forecasts`` with a quantile column added for each level."""
        if not levels:
            raise ValueError("at least one quantile level is required")
        if any(not 0 < level < 1 for level in levels):
            raise ValueError("quantile levels must be strictly between 0 and 1")
        joined = forecasts.join(self.sizes, on="id", how="left").sort("id", "date")
        mean = joined.get_column("forecast").cast(pl.Float64).to_numpy()
        size = joined.get_column("size").fill_null(np.nan).cast(pl.Float64).to_numpy()
        return joined.drop("size").with_columns(
            pl.Series(quantile_column(level), _quantiles(level, mean, size)) for level in levels
        )
