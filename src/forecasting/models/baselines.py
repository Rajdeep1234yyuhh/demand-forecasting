"""Baseline forecasters. A learned model must beat these to justify its complexity."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from forecasting.models.base import Forecaster, NotFittedError, future_dates
from forecasting.utils.config import ForecastConfig

BASELINES = ("naive", "seasonal_naive", "moving_average")


class _RecentWindowForecaster(Forecaster):
    """Base for baselines that only need each series' last ``window`` days."""

    def __init__(self, window: int) -> None:
        super().__init__()
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window
        self._recent: pl.DataFrame | None = None

    @property
    def recent(self) -> pl.DataFrame:
        if self._recent is None:
            raise NotFittedError(f"{self.name} must be fitted first")
        return self._recent

    def _fit(self, history: pl.LazyFrame, cutoff: date) -> None:
        start = cutoff - timedelta(days=self.window - 1)
        recent = (
            history.filter(pl.col("date").is_between(start, cutoff))
            .select("id", "date", pl.col("sales").cast(pl.Float64))
            .collect(engine="streaming")
        )
        days = recent.get_column("date").n_unique()
        if days < self.window:
            raise ValueError(f"{self.name} needs {self.window} days of history, got {days}")
        if recent.height != recent.get_column("id").n_unique() * self.window:
            raise ValueError(
                f"{self.name} needs every series on each of the last {self.window} days"
            )
        self._recent = recent


class Naive(_RecentWindowForecaster):
    """Repeats each series' last observed value."""

    name = "naive"

    def __init__(self) -> None:
        super().__init__(window=1)

    def _predict(self, cutoff: date, horizon: int) -> pl.DataFrame:
        last = self.recent.select("id", pl.col("sales").alias("forecast"))
        return last.join(future_dates(cutoff, horizon), how="cross")


class SeasonalNaive(_RecentWindowForecaster):
    """Repeats each series' last ``period`` days, e.g. the same weekday last week."""

    name = "seasonal_naive"

    def __init__(self, period: int = 7) -> None:
        super().__init__(window=period)

    def _predict(self, cutoff: date, horizon: int) -> pl.DataFrame:
        period = self.window
        # Phase 1..period identifies the position in the season: the oldest day in the
        # window is phase 1, the cutoff day is phase ``period``.
        source = self.recent.select(
            "id",
            ((pl.col("date") - pl.lit(cutoff)).dt.total_days() + period).alias("phase"),
            pl.col("sales").alias("forecast"),
        )
        targets = future_dates(cutoff, horizon).with_columns(
            ((pl.col("step") - 1) % period + 1).alias("phase")
        )
        return targets.join(source, on="phase")


class MovingAverage(_RecentWindowForecaster):
    """Forecasts each series' mean over its last ``window`` days."""

    name = "moving_average"

    def __init__(self, window: int = 28) -> None:
        super().__init__(window=window)

    def _predict(self, cutoff: date, horizon: int) -> pl.DataFrame:
        means = self.recent.group_by("id").agg(pl.col("sales").mean().alias("forecast"))
        return means.join(future_dates(cutoff, horizon), how="cross")


def make_baseline(name: str, config: ForecastConfig) -> Forecaster:
    """Build a baseline by name, with parameters from ``config``."""
    match name:
        case "naive":
            return Naive()
        case "seasonal_naive":
            return SeasonalNaive(period=config.seasonal_period)
        case "moving_average":
            return MovingAverage(window=config.moving_average_window)
    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINES}")
