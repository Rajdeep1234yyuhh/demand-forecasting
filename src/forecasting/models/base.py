"""Forecaster interface shared by baselines and learned models.

Every model fits on history up to a cutoff and forecasts every series for the next
``horizon`` days. Outputs are validated against one schema, so evaluation never has to
guess at a model's output, and a model bug fails loudly instead of skewing a metric.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import ClassVar, Protocol, Self, runtime_checkable

import pandera.polars as pa
import polars as pl

from forecasting.data.schemas import validate


class NotFittedError(RuntimeError):
    """Raised when a model is used before it has been fitted."""


class ForecastSchema(pa.DataFrameModel):
    id: pl.String
    date: pl.Date
    forecast: pl.Float64 = pa.Field(ge=0)

    class Config:
        strict = True
        unique = ["id", "date"]  # noqa: RUF012 - pandera config


def future_dates(cutoff: date, horizon: int) -> pl.DataFrame:
    """Forecast dates after ``cutoff``, numbered by ``step`` from 1 to ``horizon``."""
    steps = list(range(1, horizon + 1))
    return pl.DataFrame(
        {"date": [cutoff + timedelta(days=step) for step in steps], "step": steps},
        schema={"date": pl.Date, "step": pl.Int64},
    )


class Forecaster(ABC):
    """Fit on history up to a cutoff, then forecast every series ``horizon`` days ahead."""

    name: ClassVar[str]

    def __init__(self) -> None:
        self._cutoff: date | None = None

    @property
    def cutoff(self) -> date:
        """Last day of the fitted history."""
        if self._cutoff is None:
            raise NotFittedError(f"{self.name} must be fitted first")
        return self._cutoff

    def fit(self, history: pl.LazyFrame) -> Self:
        """Fit on long-format history with at least ``id``, ``date``, and ``sales``."""
        cutoff = history.select(pl.col("date").max()).collect(engine="streaming").item()
        if cutoff is None:
            raise ValueError("history is empty")
        self._fit(history, cutoff)
        self._cutoff = cutoff
        return self

    def predict(self, horizon: int) -> pl.DataFrame:
        """Return validated forecasts with columns ``id``, ``date``, and ``forecast``."""
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        forecasts = self._predict(self.cutoff, horizon).select("id", "date", "forecast")
        return validate(forecasts.sort("id", "date"), ForecastSchema, f"{self.name} forecasts")

    @abstractmethod
    def _fit(self, history: pl.LazyFrame, cutoff: date) -> None: ...

    @abstractmethod
    def _predict(self, cutoff: date, horizon: int) -> pl.DataFrame: ...


@runtime_checkable
class ExplainsFeatures(Protocol):
    """A fitted model that can report how much each feature contributed."""

    def feature_importance(self) -> pl.DataFrame:
        """Rows of ``store_id``, ``feature``, ``gain``, and ``share`` of gain per store."""
        ...
