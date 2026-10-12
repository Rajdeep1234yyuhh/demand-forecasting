"""Per-store LightGBM models trained on the leakage-safe feature table.

One model per store keeps peak memory at about one store's data however many stores there
are, and lets stores train in parallel later, one pipeline task each. Features are built
with a shift equal to the horizon, so one model forecasts every day of the horizon directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl

from forecasting.features.build import (
    CATEGORICAL_FEATURES,
    ON_SHELF,
    build_features,
    feature_columns,
    make_panel,
    split_rows,
)
from forecasting.models.base import Forecaster, NotFittedError
from forecasting.utils.config import FeatureConfig, LightGBMConfig


@dataclass(frozen=True)
class CategoryEncoder:
    """Stable integer codes for categorical features, learned from training rows.

    Codes follow the sorted vocabulary. Values unseen in training and nulls become NaN,
    which LightGBM treats as missing.
    """

    vocabularies: Mapping[str, tuple[str, ...]]

    @classmethod
    def fit(cls, frame: pl.DataFrame, columns: Sequence[str]) -> CategoryEncoder:
        return cls(
            {
                column: tuple(frame.get_column(column).drop_nulls().unique().sort().to_list())
                for column in columns
            }
        )

    def matrix(self, frame: pl.DataFrame, columns: Sequence[str]) -> npt.NDArray[np.float32]:
        """Float32 matrix of ``columns`` in order, with categories replaced by their codes."""
        encoded = [
            pl.col(column)
            .replace_strict(
                self.vocabularies[column],
                range(len(self.vocabularies[column])),
                default=None,
                return_dtype=pl.Int32,
            )
            .cast(pl.Float32)
            if column in self.vocabularies
            else pl.col(column).cast(pl.Float32)
            for column in columns
        ]
        return frame.select(encoded).to_numpy()


@dataclass(frozen=True)
class StoreModel:
    booster: lgb.Booster
    encoder: CategoryEncoder
    # Feature rows for the days after the cutoff, built together with the training rows.
    forecast_rows: pl.DataFrame


class LightGBMForecaster(Forecaster):
    """One LightGBM model per store, with forecasts set to zero while an item is off the shelf."""

    name = "lightgbm"

    def __init__(
        self,
        params: LightGBMConfig,
        features: FeatureConfig,
        *,
        calendar: pl.DataFrame,
        prices: pl.LazyFrame,
        horizon: int,
    ) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self._params = params
        self._features = features
        self._calendar = calendar
        self._prices = prices
        self._horizon = horizon
        self._columns = feature_columns(features)
        self._stores: dict[str, StoreModel] = {}

    @property
    def stores(self) -> Mapping[str, StoreModel]:
        if not self._stores:
            raise NotFittedError(f"{self.name} must be fitted first")
        return self._stores

    def _fit(self, history: pl.LazyFrame, cutoff: date) -> None:
        stores = (
            history.select(pl.col("store_id").unique().sort())
            .collect(engine="streaming")
            .get_column("store_id")
            .to_list()
        )
        self._stores = {
            store: self._fit_store(history.filter(pl.col("store_id") == store), store, cutoff)
            for store in stores
        }

    def _fit_store(self, history: pl.LazyFrame, store: str, cutoff: date) -> StoreModel:
        panel = make_panel(history.collect(engine="streaming"), cutoff, self._horizon)
        features = build_features(
            panel,
            self._calendar,
            self._prices.filter(pl.col("store_id") == store),
            state=store.split("_", maxsplit=1)[0],
            config=self._features,
            shift=self._horizon,
        )
        training, forecast_rows = split_rows(features, cutoff)
        if training.is_empty():
            raise ValueError(f"store {store} has no training rows on or before {cutoff}")
        encoder = CategoryEncoder.fit(training, CATEGORICAL_FEATURES)
        dataset = lgb.Dataset(
            encoder.matrix(training, self._columns),
            label=training.get_column("sales").to_numpy(),
            feature_name=self._columns,
            categorical_feature=list(CATEGORICAL_FEATURES),
            free_raw_data=True,
        )
        booster = lgb.train(
            self._params.training_params(),
            dataset,
            num_boost_round=self._params.num_boost_round,
        )
        return StoreModel(booster, encoder, forecast_rows.select("id", "date", *self._columns))

    def _predict(self, cutoff: date, horizon: int) -> pl.DataFrame:
        if horizon > self._horizon:
            raise ValueError(
                f"features were built for a {self._horizon}-day horizon; "
                f"forecasting {horizon} days would need sales not known at the cutoff"
            )
        last_day = cutoff + timedelta(days=horizon)
        parts = []
        for model in self.stores.values():
            rows = model.forecast_rows.filter(pl.col("date") <= last_day)
            predicted = model.booster.predict(model.encoder.matrix(rows, self._columns))
            parts.append(
                rows.select(
                    "id",
                    "date",
                    pl.when(ON_SHELF)
                    .then(pl.Series(np.asarray(predicted, dtype=np.float64)).clip(0.0))
                    .otherwise(0.0)
                    .alias("forecast"),
                )
            )
        return pl.concat(parts)

    def feature_importance(self) -> pl.DataFrame:
        """Total gain per feature for each store, with each store's shares summing to one."""
        frames = [
            pl.DataFrame(
                {
                    "store_id": store,
                    "feature": model.booster.feature_name(),
                    "gain": model.booster.feature_importance(importance_type="gain"),
                }
            )
            for store, model in self.stores.items()
        ]
        gain = pl.col("gain").cast(pl.Float64)
        return (
            pl.concat(frames)
            .with_columns(gain, (gain / gain.sum().over("store_id")).fill_nan(0.0).alias("share"))
            .sort("store_id", "gain", descending=[False, True])
        )
