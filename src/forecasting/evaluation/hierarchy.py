"""Hierarchical evaluation with the M5 competition metric, WRMSSE.

WRMSSE scores forecasts at all 12 aggregation levels of the hierarchy, from total sales down
to single item-store series:

- each series' RMSE is divided by its in-sample one-step naive error (RMSSE), so series of
  different volumes are comparable;
- each series is weighted by its share of revenue over the last ``weight_days`` of training,
  so errors on valuable series count more;
- the 12 level scores are averaged with equal weight.

Scales use the full training history, so they are computed once per backtest fold, in a
single pass over the data. Each store's item-level daily sales are read once and rolled up
(item -> department -> category -> store); they are also summed into per-state and global
item-level tables, which roll up the same way. Memory stays bounded by a few item-by-day
tables. This relies on the hierarchy that ingestion validates: an item determines its
department and category, and a store determines its state.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from forecasting.evaluation.metrics import bias, wape

HIERARCHY_COLUMNS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
_ALL_KEY = "_all"


@dataclass(frozen=True)
class Level:
    name: str
    keys: tuple[str, ...]

    @property
    def group_keys(self) -> list[str]:
        """Grouping columns. The top level groups on a constant column."""
        return list(self.keys) if self.keys else [_ALL_KEY]

    @property
    def partition(self) -> str | None:
        """Finest column within which this level can be computed independently."""
        for column in ("store_id", "state_id"):
            if column in self.keys:
                return column
        return None


LEVELS: tuple[Level, ...] = (
    Level("total", ()),
    Level("state", ("state_id",)),
    Level("store", ("store_id",)),
    Level("category", ("cat_id",)),
    Level("department", ("dept_id",)),
    Level("state_category", ("state_id", "cat_id")),
    Level("state_department", ("state_id", "dept_id")),
    Level("store_category", ("store_id", "cat_id")),
    Level("store_department", ("store_id", "dept_id")),
    Level("item", ("item_id",)),
    Level("item_state", ("item_id", "state_id")),
    Level("item_store", ("item_id", "store_id")),
)
BOTTOM_LEVEL = LEVELS[-1]

# Product hierarchy from finest to coarsest; each level rolls up from the one before it.
_PRODUCT_KEYS = ("item_id", "dept_id", "cat_id")
_ROLL_UPS: tuple[tuple[str, str | None], ...] = (
    ("item_id", "dept_id"),
    ("dept_id", "cat_id"),
    ("cat_id", None),
)
_ITEM_DAILY_SCHEMA = {"item_id": pl.String, "date": pl.Date, "sales": pl.Float64}


def _keyed(frame: pl.DataFrame, level: Level) -> pl.DataFrame:
    """Add the constant grouping column the top level needs."""
    return frame if level.keys else frame.with_columns(pl.lit("all").alias(_ALL_KEY))


def _product_key(level: Level) -> str | None:
    return next((key for key in _PRODUCT_KEYS if key in level.keys), None)


def _series_scales(daily: pl.DataFrame, keys: Sequence[str]) -> pl.DataFrame:
    """Mean squared and absolute one-step naive errors, from each series' first sale onwards.

    Days before the first sale are not demand (the item was not stocked), so including them
    would understate the scale. Series with fewer than two active days get a null scale.
    """
    sales = pl.col("sales").sort_by("date")
    step = sales.filter(sales.cum_max() > 0).diff()
    scales = [step.pow(2).mean().alias("scale_sq"), step.abs().mean().alias("scale_abs")]
    return daily.group_by(keys).agg(scales) if keys else daily.select(scales)


def _roll_up(
    daily: pl.DataFrame, child: str, parent: str | None, products: pl.DataFrame
) -> pl.DataFrame:
    """Sum daily sales from ``child`` up to ``parent``; ``None`` means one total series."""
    if parent is None:
        return daily.group_by("date").agg(pl.col("sales").sum())
    mapping = products.select(child, parent).unique()
    return daily.join(mapping, on=child).group_by(parent, "date").agg(pl.col("sales").sum())


def _scope_scales(
    items_daily: pl.DataFrame,
    products: pl.DataFrame,
    scope: Mapping[str, str],
    partition: str | None,
) -> dict[str, pl.DataFrame]:
    """Scales for every level within one scope: a store, a state, or everything.

    ``items_daily`` has one row per item per day in the scope. ``scope`` holds the keys that
    identify it, such as ``{"store_id": "CA_1", "state_id": "CA"}``.
    """
    dailies: dict[str | None, pl.DataFrame] = {"item_id": items_daily}
    for child, parent in _ROLL_UPS:
        dailies[parent] = _roll_up(dailies[child], child, parent, products)

    scales: dict[str, pl.DataFrame] = {}
    for level in LEVELS:
        if level.partition != partition:
            continue
        key = _product_key(level)
        fixed = {name: value for name, value in scope.items() if name in level.keys}
        if not level.keys:
            fixed = {_ALL_KEY: "all"}
        scales[level.name] = _series_scales(dailies[key], [key] if key else []).with_columns(
            pl.lit(value).alias(name) for name, value in fixed.items()
        )
    return scales


def _accumulate(total: pl.DataFrame, part: pl.DataFrame) -> pl.DataFrame:
    """Add one scope's item-level daily sales into a running total.

    Both frames are sorted by item and date. When their keys match row for row, as when
    stores carry the same items over the same days, sales add as columns; otherwise this
    falls back to a group-by, which is correct for any keys but several times slower.
    """
    if total.is_empty():
        return part
    keys = ["item_id", "date"]
    if total.select(keys).equals(part.select(keys)):
        return total.with_columns(pl.col("sales") + part.get_column("sales"))
    return pl.concat([total, part]).group_by(keys).agg(pl.col("sales").sum()).sort(keys)


def compute_scales(train: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    """Per-series scales for every level, keyed by level name, in one pass over the data."""
    products = train.select(*_PRODUCT_KEYS).unique().collect(engine="streaming")
    stores = (
        train.select("state_id", "store_id")
        .unique()
        .collect(engine="streaming")
        .group_by("state_id")
        .agg(pl.col("store_id").sort())
        .sort("state_id")
    )
    if stores.is_empty():
        raise ValueError("training history is empty")

    parts: dict[str, list[pl.DataFrame]] = defaultdict(list)
    everything = pl.DataFrame(schema=_ITEM_DAILY_SCHEMA)
    for state, store_ids in stores.iter_rows():
        state_items = pl.DataFrame(schema=_ITEM_DAILY_SCHEMA)
        for store in store_ids:
            # Filtering on the partition column reads only this store's file.
            store_items = (
                train.filter(pl.col("store_id") == store)
                .select("item_id", "date", pl.col("sales").cast(pl.Float64))
                .sort("item_id", "date")
                .collect(engine="streaming")
            )
            scope = {"store_id": store, "state_id": state}
            for name, frame in _scope_scales(store_items, products, scope, "store_id").items():
                parts[name].append(frame)
            state_items = _accumulate(state_items, store_items)
        scope = {"state_id": state}
        for name, frame in _scope_scales(state_items, products, scope, "state_id").items():
            parts[name].append(frame)
        everything = _accumulate(everything, state_items)
    for name, frame in _scope_scales(everything, products, {}, None).items():
        parts[name].append(frame)
    return {level.name: pl.concat(parts[level.name]) for level in LEVELS}


def compute_weights(
    train: pl.LazyFrame,
    calendar: pl.DataFrame,
    prices: pl.LazyFrame,
    *,
    cutoff: date,
    weight_days: int,
) -> dict[str, pl.DataFrame]:
    """Each series' share of revenue over the last ``weight_days`` up to ``cutoff``.

    Weights sum to 1 within every level. Sales with no matching price (bounded by the
    ingestion gate) contribute no revenue.
    """
    if weight_days <= 0:
        raise ValueError("weight_days must be positive")
    window_start = cutoff - timedelta(days=weight_days - 1)
    price_keys = ["store_id", "item_id", "wm_yr_wk"]
    revenue = (
        train.filter(pl.col("date").is_between(window_start, cutoff))
        .join(calendar.lazy().select("date", "wm_yr_wk"), on="date")
        .join(prices.select(*price_keys, "sell_price"), on=price_keys, how="left")
        .select(
            *HIERARCHY_COLUMNS,
            (pl.col("sales") * pl.col("sell_price").fill_null(0.0)).alias("revenue"),
        )
        .collect(engine="streaming")
    )
    total = float(revenue.get_column("revenue").sum())
    if total <= 0:
        raise ValueError(f"no revenue in the {weight_days} days before {cutoff}; cannot weight")
    return {
        level.name: _keyed(revenue, level)
        .group_by(level.group_keys)
        .agg((pl.col("revenue").sum() / total).alias("weight"))
        for level in LEVELS
    }


@dataclass(frozen=True)
class Evaluation:
    wrmsse: float
    wape: float
    mase: float
    bias: float
    # Weight of series that could not be scored (fewer than two active days), averaged
    # over levels. Near zero in practice; a large value means WRMSSE is understated.
    unscored_weight: float
    wrmsse_by_level: Mapping[str, float]

    def as_row(self) -> dict[str, float]:
        """Flat mapping for a results table."""
        return {
            "wrmsse": self.wrmsse,
            "wape": self.wape,
            "mase": self.mase,
            "bias": self.bias,
            "unscored_weight": self.unscored_weight,
            **{f"wrmsse_{name}": value for name, value in self.wrmsse_by_level.items()},
        }


_REQUIRED_COLUMNS = frozenset({*HIERARCHY_COLUMNS, "date", "actual", "forecast"})


@dataclass(frozen=True)
class HierarchicalEvaluator:
    """Scores bottom-level forecasts against actuals for one training cutoff."""

    scales: Mapping[str, pl.DataFrame]
    weights: Mapping[str, pl.DataFrame]

    @classmethod
    def from_history(
        cls,
        train: pl.LazyFrame,
        calendar: pl.DataFrame,
        prices: pl.LazyFrame,
        *,
        weight_days: int,
    ) -> HierarchicalEvaluator:
        cutoff = train.select(pl.col("date").max()).collect(engine="streaming").item()
        if cutoff is None:
            raise ValueError("training history is empty")
        return cls(
            scales=compute_scales(train),
            weights=compute_weights(
                train, calendar, prices, cutoff=cutoff, weight_days=weight_days
            ),
        )

    def score(self, predictions: pl.DataFrame) -> Evaluation:
        """Score predictions with hierarchy columns, ``date``, ``actual``, and ``forecast``."""
        missing = _REQUIRED_COLUMNS - set(predictions.columns)
        if missing:
            raise ValueError(f"predictions are missing columns: {sorted(missing)}")
        if predictions.select(pl.col("actual", "forecast").null_count()).sum_horizontal().item():
            raise ValueError("predictions contain null actuals or forecasts")

        by_level: dict[str, float] = {}
        unscored = 0.0
        for level in LEVELS:
            by_level[level.name], level_unscored = self._score_level(predictions, level)
            unscored += level_unscored
        return Evaluation(
            wrmsse=sum(by_level.values()) / len(LEVELS),
            wape=wape(predictions),
            mase=self._mase(predictions),
            bias=bias(predictions),
            unscored_weight=unscored / len(LEVELS),
            wrmsse_by_level=by_level,
        )

    def _score_level(self, predictions: pl.DataFrame, level: Level) -> tuple[float, float]:
        keys = level.group_keys
        series = (
            _keyed(predictions, level)
            .group_by([*keys, "date"])
            .agg(pl.col("actual").sum(), pl.col("forecast").sum())
            .group_by(keys)
            .agg(((pl.col("forecast") - pl.col("actual")) ** 2).mean().alias("mse"))
            .join(self.scales[level.name], on=keys, how="left")
            .join(self.weights[level.name], on=keys, how="left")
            .with_columns(
                pl.col("weight").fill_null(0.0),
                (pl.col("scale_sq") > 0).fill_null(value=False).alias("scored"),
            )
        )
        rmsse = (pl.col("mse") / pl.col("scale_sq")).sqrt()
        score, unscored = series.select(
            (pl.col("weight") * rmsse).filter(pl.col("scored")).sum().alias("score"),
            pl.col("weight").filter(~pl.col("scored")).sum().alias("unscored"),
        ).row(0)
        return float(score), float(unscored)

    def _mase(self, predictions: pl.DataFrame) -> float:
        """Mean over bottom-level series of MAE divided by the in-sample naive MAE."""
        keys = BOTTOM_LEVEL.group_keys
        value = (
            predictions.group_by(keys)
            .agg((pl.col("forecast") - pl.col("actual")).abs().mean().alias("mae"))
            .join(self.scales[BOTTOM_LEVEL.name], on=keys, how="inner")
            .filter(pl.col("scale_abs") > 0)
            .select((pl.col("mae") / pl.col("scale_abs")).mean())
            .item()
        )
        return math.nan if value is None else float(value)
