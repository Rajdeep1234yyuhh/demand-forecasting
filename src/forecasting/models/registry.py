"""Build any forecaster by name.

Callers (the backtest CLI now, the pipeline later) ask for models by name, so adding a
model family means adding entries here instead of editing every caller.
"""

from __future__ import annotations

from collections.abc import Callable

from forecasting.data.dataset import Dataset
from forecasting.models.base import Forecaster
from forecasting.models.baselines import BASELINES, make_baseline
from forecasting.models.lgbm import LightGBMForecaster
from forecasting.models.statistical import (
    TSB,
    CrostonSBA,
    SeasonalExponentialSmoothing,
    SimpleExponentialSmoothing,
)
from forecasting.utils.config import AppConfig

# Every builder takes the full config and the dataset; each model uses what it needs.
Builder = Callable[[AppConfig, Dataset], Forecaster]


def _baseline(name: str) -> Builder:
    def build(config: AppConfig, dataset: Dataset) -> Forecaster:
        return make_baseline(name, config.forecast)

    return build


def _croston_sba(config: AppConfig, dataset: Dataset) -> Forecaster:
    return CrostonSBA()


def _tsb(config: AppConfig, dataset: Dataset) -> Forecaster:
    return TSB()


def _ses(config: AppConfig, dataset: Dataset) -> Forecaster:
    return SimpleExponentialSmoothing()


def _seasonal_es(config: AppConfig, dataset: Dataset) -> Forecaster:
    return SeasonalExponentialSmoothing(season_length=config.forecast.seasonal_period)


def _lightgbm(config: AppConfig, dataset: Dataset) -> Forecaster:
    return LightGBMForecaster(
        config.models.lightgbm,
        config.features,
        calendar=dataset.calendar,
        prices=dataset.prices,
        horizon=config.forecast.horizon_days,
    )


_BUILDERS: dict[str, Builder] = {
    **{name: _baseline(name) for name in BASELINES},
    "croston_sba": _croston_sba,
    "tsb": _tsb,
    "ses": _ses,
    "seasonal_es": _seasonal_es,
    "lightgbm": _lightgbm,
}
MODEL_NAMES: tuple[str, ...] = tuple(_BUILDERS)


def make_model(name: str, config: AppConfig, dataset: Dataset) -> Forecaster:
    """Build the model registered as ``name``."""
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}") from None
    return builder(config, dataset)
