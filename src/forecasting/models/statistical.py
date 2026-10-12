"""Classical per-series models from statsforecast.

Each series is modelled on its own, from its first sale onwards. Days before an item first
sells carry no demand signal, and leading zeros would pull level estimates down. Series
that never sold are forecast as zero. Work runs one store at a time to bound memory.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import date

import polars as pl
from statsforecast import StatsForecast
from statsforecast import models as sf_models

from forecasting.models.base import Forecaster, NotFittedError, future_dates

STATISTICAL_MODELS = ("croston_sba", "tsb", "ses", "seasonal_es")
_FORECAST = "forecast"


class _StatsForecastModel(Forecaster):
    """Shared fitting loop: statsforecast fits and forecasts every series in one call."""

    def __init__(self) -> None:
        super().__init__()
        self._history: pl.LazyFrame | None = None

    @abstractmethod
    def _model(self) -> object:
        """The statsforecast model, with its output column named ``forecast``."""

    def _fit(self, history: pl.LazyFrame, cutoff: date) -> None:
        # statsforecast's ``forecast`` fits and predicts in one memory-efficient pass, so
        # fitting only keeps a lazy reference; the work happens in ``predict``.
        self._history = history.filter(pl.col("date") <= cutoff).select(
            "id", "store_id", "date", "sales"
        )

    def _predict(self, cutoff: date, horizon: int) -> pl.DataFrame:
        if self._history is None:
            raise NotFittedError(f"{self.name} must be fitted first")
        history = self._history
        stores = (
            history.select(pl.col("store_id").unique().sort())
            .collect(engine="streaming")
            .get_column("store_id")
            .to_list()
        )
        return pl.concat(
            [
                self._predict_store(history.filter(pl.col("store_id") == store), cutoff, horizon)
                for store in stores
            ]
        )

    def _predict_store(
        self, store_history: pl.LazyFrame, cutoff: date, horizon: int
    ) -> pl.DataFrame:
        history = store_history.select("id", "date", pl.col("sales").cast(pl.Float64)).collect(
            engine="streaming"
        )
        active = history.filter(pl.col("sales").cum_max().over("id", order_by="date") > 0)
        forecasts = pl.DataFrame(schema={"id": pl.String, "date": pl.Date, _FORECAST: pl.Float64})
        if not active.is_empty():
            raw = StatsForecast(
                models=[self._model()], freq="1d", n_jobs=1, fallback_model=sf_models.Naive()
            ).forecast(df=active.rename({"id": "unique_id", "date": "ds", "sales": "y"}), h=horizon)
            if not isinstance(raw, pl.DataFrame):
                raise TypeError(f"statsforecast returned {type(raw).__name__}, not polars")
            # Some models return NaN when a series is shorter than they need (e.g. less
            # than one season); those fall back to the series' mean demand since launch.
            fallback = active.group_by("id").agg(pl.col("sales").mean().alias("fallback"))
            forecasts = (
                raw.select(
                    pl.col("unique_id").alias("id"),
                    pl.col("ds").cast(pl.Date).alias("date"),
                    pl.col(_FORECAST).cast(pl.Float64),
                )
                .join(fallback, on="id")
                .select(
                    "id",
                    "date",
                    pl.col(_FORECAST).fill_nan(None).fill_null(pl.col("fallback")).clip(0.0),
                )
            )
        never_sold = history.select("id").unique().join(active.select("id"), on="id", how="anti")
        zeros = never_sold.join(future_dates(cutoff, horizon).select("date"), how="cross")
        return pl.concat([forecasts, zeros.with_columns(pl.lit(0.0).alias(_FORECAST))])


class CrostonSBA(_StatsForecastModel):
    """Croston's method with the Syntetos-Boylan bias correction, for intermittent demand.

    It smooths demand sizes and intervals between sales separately, so long runs of zeros
    don't drag the level of sizes down.
    """

    name = "croston_sba"

    def _model(self) -> object:
        return sf_models.CrostonSBA(alias=_FORECAST)


class TSB(_StatsForecastModel):
    """Teunter-Syntetos-Babai: smooths demand size and the probability of a sale each day.

    Unlike Croston, the probability also decays on zero days, so an item that stopped
    selling sees its forecast fall towards zero instead of staying at its old rate.
    """

    name = "tsb"

    def __init__(self, alpha_demand: float = 0.1, alpha_probability: float = 0.1) -> None:
        super().__init__()
        if not (0 < alpha_demand <= 1 and 0 < alpha_probability <= 1):
            raise ValueError("smoothing weights must be in (0, 1]")
        self.alpha_demand = alpha_demand
        self.alpha_probability = alpha_probability

    def _model(self) -> object:
        return sf_models.TSB(
            alpha_d=self.alpha_demand, alpha_p=self.alpha_probability, alias=_FORECAST
        )


class SimpleExponentialSmoothing(_StatsForecastModel):
    """Exponentially weighted level, with the smoothing weight fitted per series."""

    name = "ses"

    def _model(self) -> object:
        return sf_models.SimpleExponentialSmoothingOptimized(alias=_FORECAST)


class SeasonalExponentialSmoothing(_StatsForecastModel):
    """Exponential smoothing per position in the season, e.g. per weekday."""

    name = "seasonal_es"

    def __init__(self, season_length: int = 7) -> None:
        super().__init__()
        if season_length <= 0:
            raise ValueError("season_length must be positive")
        self.season_length = season_length

    def _model(self) -> object:
        return sf_models.SeasonalExponentialSmoothingOptimized(
            season_length=self.season_length, alias=_FORECAST
        )
