"""WRMSSE tests, checked against an independent plain-Python implementation."""

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date, timedelta
from typing import Any

import polars as pl
import pytest

from forecasting.data.ingest import to_long
from forecasting.evaluation.hierarchy import (
    BOTTOM_LEVEL,
    LEVELS,
    HierarchicalEvaluator,
    _accumulate,
    compute_scales,
    compute_weights,
)
from tests.conftest import START, M5Fixture

CUTOFF_DAY = 11
HORIZON = 3
CUTOFF = START + timedelta(days=CUTOFF_DAY - 1)

Row = dict[str, Any]
Key = tuple[Any, ...]


def _key(row: Row, keys: Sequence[str]) -> Key:
    return tuple(row[column] for column in keys)


def _scale(history: dict[date, float], *, squared: bool) -> float | None:
    """Mean one-step naive error from the first sale onwards; None if undefined."""
    values = [history[day] for day in sorted(history)]
    first = next((i for i, value in enumerate(values) if value > 0), None)
    if first is None:
        return None
    steps = [b - a for a, b in itertools.pairwise(values[first:])]
    if not steps:
        return None
    return sum(step * step if squared else abs(step) for step in steps) / len(steps)


def _reference(
    train: Iterable[Row],
    predictions: Iterable[Row],
    prices: dict[tuple[str, str, int], float],
    weeks: dict[date, int],
    weight_days: int,
) -> tuple[list[float], float]:
    """WRMSSE per level and bottom-level MASE, computed with dictionaries and loops."""
    train, predictions = list(train), list(predictions)
    window_start = max(row["date"] for row in train) - timedelta(days=weight_days - 1)
    level_scores = []
    for level in LEVELS:
        history: dict[Key, dict[date, float]] = defaultdict(lambda: defaultdict(float))
        revenue: dict[Key, float] = defaultdict(float)
        for row in train:
            key = _key(row, level.keys)
            history[key][row["date"]] += row["sales"]
            if row["date"] >= window_start:
                price = prices.get((row["store_id"], row["item_id"], weeks[row["date"]]), 0.0)
                revenue[key] += row["sales"] * price
        total_revenue = sum(revenue.values())

        actual: dict[Key, dict[date, float]] = defaultdict(lambda: defaultdict(float))
        forecast: dict[Key, dict[date, float]] = defaultdict(lambda: defaultdict(float))
        for row in predictions:
            key = _key(row, level.keys)
            actual[key][row["date"]] += row["actual"]
            forecast[key][row["date"]] += row["forecast"]

        score = 0.0
        for key, days in actual.items():
            scale = _scale(history[key], squared=True)
            if not scale:
                continue
            mse = sum((forecast[key][day] - days[day]) ** 2 for day in days) / len(days)
            score += revenue[key] / total_revenue * math.sqrt(mse / scale)
        level_scores.append(score)

    bottom_history: dict[Key, dict[date, float]] = defaultdict(dict)
    for row in train:
        bottom_history[_key(row, BOTTOM_LEVEL.keys)][row["date"]] = row["sales"]
    errors: dict[Key, list[float]] = defaultdict(list)
    for row in predictions:
        errors[_key(row, BOTTOM_LEVEL.keys)].append(abs(row["forecast"] - row["actual"]))
    ratios = []
    for key, series_errors in errors.items():
        scale = _scale(bottom_history[key], squared=False)
        if scale:
            ratios.append(sum(series_errors) / len(series_errors) / scale)
    return level_scores, sum(ratios) / len(ratios)


@pytest.fixture
def long(m5: M5Fixture) -> pl.DataFrame:
    return to_long(m5.sales, m5.day_columns, m5.calendar)


def _split(long: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = long.filter(pl.col("date") <= CUTOFF)
    test = long.filter(pl.col("date").is_between(CUTOFF, CUTOFF + timedelta(days=HORIZON)))
    return train, test.filter(pl.col("date") > CUTOFF)


def _predictions(test: pl.DataFrame) -> pl.DataFrame:
    """Attach varied, deterministic forecasts so every level has non-zero error."""
    frame = test.with_columns(pl.col("sales").cast(pl.Float64).alias("actual")).drop("sales")
    values = [(i * 7 % 5) * 0.5 for i in range(frame.height)]
    return frame.with_columns(pl.Series("forecast", values, dtype=pl.Float64))


def _evaluator(
    train: pl.DataFrame, calendar: pl.DataFrame, prices: pl.DataFrame
) -> HierarchicalEvaluator:
    return HierarchicalEvaluator.from_history(
        train.lazy(), calendar, prices.lazy(), weight_days=HORIZON
    )


def test_levels_follow_m5() -> None:
    assert len(LEVELS) == 12
    assert len({level.name for level in LEVELS}) == 12
    assert BOTTOM_LEVEL.keys == ("item_id", "store_id")
    assert {level.partition for level in LEVELS} == {None, "state_id", "store_id"}
    # Roll-ups assume each level has at most one product key.
    product_keys = {"item_id", "dept_id", "cat_id"}
    assert all(len(product_keys.intersection(level.keys)) <= 1 for level in LEVELS)


def test_matches_reference_implementation(m5: M5Fixture, long: pl.DataFrame) -> None:
    train, test = _split(long)
    predictions = _predictions(test)
    evaluation = _evaluator(train, m5.calendar, m5.prices).score(predictions)

    prices = {
        (row["store_id"], row["item_id"], row["wm_yr_wk"]): row["sell_price"]
        for row in m5.prices.iter_rows(named=True)
    }
    weeks = {row["date"]: row["wm_yr_wk"] for row in m5.calendar.iter_rows(named=True)}
    expected_levels, expected_mase = _reference(
        train.iter_rows(named=True), predictions.iter_rows(named=True), prices, weeks, HORIZON
    )
    assert list(evaluation.wrmsse_by_level.values()) == pytest.approx(expected_levels)
    assert evaluation.wrmsse == pytest.approx(sum(expected_levels) / len(LEVELS))
    assert evaluation.wrmsse > 0
    assert evaluation.mase == pytest.approx(expected_mase)
    # The never-selling series has no revenue, so excluding it costs no weight.
    assert evaluation.unscored_weight == 0.0


def test_perfect_forecast_scores_zero(m5: M5Fixture, long: pl.DataFrame) -> None:
    train, test = _split(long)
    perfect = _predictions(test).with_columns(pl.col("actual").alias("forecast"))
    evaluation = _evaluator(train, m5.calendar, m5.prices).score(perfect)
    assert evaluation.wrmsse == 0.0
    assert evaluation.wape == 0.0
    assert evaluation.mase == 0.0
    assert evaluation.bias == 0.0


def test_weights_sum_to_one_per_level(m5: M5Fixture, long: pl.DataFrame) -> None:
    train, _ = _split(long)
    weights = compute_weights(
        train.lazy(), m5.calendar, m5.prices.lazy(), cutoff=CUTOFF, weight_days=HORIZON
    )
    for level in LEVELS:
        assert weights[level.name].get_column("weight").sum() == pytest.approx(1.0)


def test_weights_need_revenue(m5: M5Fixture, long: pl.DataFrame) -> None:
    train, _ = _split(long)
    no_sales = train.with_columns(pl.lit(0, dtype=pl.Int16).alias("sales")).lazy()
    with pytest.raises(ValueError, match="no revenue"):
        compute_weights(no_sales, m5.calendar, m5.prices.lazy(), cutoff=CUTOFF, weight_days=3)
    with pytest.raises(ValueError, match="positive"):
        compute_weights(train.lazy(), m5.calendar, m5.prices.lazy(), cutoff=CUTOFF, weight_days=0)


def test_reports_unscored_weight(m5: M5Fixture, long: pl.DataFrame) -> None:
    """A series that first sells on the cutoff day has revenue but no defined scale."""
    train, test = _split(long)
    launch = (pl.col("id") == "HOBBIES_1_001_TX_1_evaluation") & (pl.col("date") == CUTOFF)
    train = train.with_columns(
        pl.when(launch).then(pl.lit(4, dtype=pl.Int16)).otherwise(pl.col("sales")).alias("sales")
    )
    new_prices = pl.DataFrame(
        {
            "store_id": ["TX_1", "TX_1"],
            "item_id": ["HOBBIES_1_001", "HOBBIES_1_001"],
            "wm_yr_wk": [11101, 11102],
            "sell_price": [5.0, 5.0],
        },
        schema_overrides={"wm_yr_wk": pl.Int32},
    )
    prices = pl.concat([m5.prices, new_prices])
    evaluation = _evaluator(train, m5.calendar, prices).score(_predictions(test))
    assert 0 < evaluation.unscored_weight < 1
    assert math.isfinite(evaluation.wrmsse)


def test_single_day_history_scores_nothing(m5: M5Fixture, long: pl.DataFrame) -> None:
    train = long.filter(pl.col("date") == START)
    test = long.filter(pl.col("date").is_between(START + timedelta(days=1), START + timedelta(3)))
    evaluator = HierarchicalEvaluator.from_history(
        train.lazy(), m5.calendar, m5.prices.lazy(), weight_days=1
    )
    evaluation = evaluator.score(_predictions(test))
    assert evaluation.wrmsse == 0.0
    assert evaluation.unscored_weight == pytest.approx(1.0)
    assert math.isnan(evaluation.mase)


def test_score_validates_input(m5: M5Fixture, long: pl.DataFrame) -> None:
    train, test = _split(long)
    evaluator = _evaluator(train, m5.calendar, m5.prices)
    predictions = _predictions(test)
    with pytest.raises(ValueError, match="missing columns"):
        evaluator.score(predictions.drop("forecast"))
    with pytest.raises(ValueError, match="null"):
        evaluator.score(predictions.with_columns(pl.lit(None, dtype=pl.Float64).alias("forecast")))


def test_rejects_empty_history(m5: M5Fixture, long: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        _evaluator(long.clear(), m5.calendar, m5.prices)


def test_compute_scales_rejects_empty_history(long: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_scales(long.clear().lazy())


def test_accumulate_aligned_and_misaligned_tables() -> None:
    next_day = START + timedelta(days=1)
    base = pl.DataFrame(
        {"item_id": ["x", "x", "y"], "date": [START, next_day, START], "sales": [1.0, 2.0, 3.0]}
    )
    # Same keys row for row: sales add as columns.
    aligned = base.with_columns(pl.col("sales") * 10)
    assert _accumulate(base, aligned).get_column("sales").to_list() == [11.0, 22.0, 33.0]
    # Different keys: falls back to a group-by and keeps the result sorted.
    other = pl.DataFrame({"item_id": ["y", "z"], "date": [START, START], "sales": [1.0, 5.0]})
    assert _accumulate(base, other).rows() == [
        ("x", START, 1.0),
        ("x", next_day, 2.0),
        ("y", START, 4.0),
        ("z", START, 5.0),
    ]
    assert _accumulate(base.clear(), base).equals(base)
