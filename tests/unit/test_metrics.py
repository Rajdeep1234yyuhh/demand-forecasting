import math

import polars as pl
import pytest

from forecasting.evaluation.metrics import bias, coverage, pinball_loss, wape


def _frame(actual: list[float], forecast: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"actual": actual, "forecast": forecast},
        schema={"actual": pl.Float64, "forecast": pl.Float64},
    )


def test_wape() -> None:
    # Absolute errors 1 + 1 + 2 = 4 over total demand 10.
    assert wape(_frame([2, 3, 5], [3, 2, 7])) == pytest.approx(0.4)


def test_bias_sign() -> None:
    assert bias(_frame([2, 3, 5], [3, 2, 7])) == pytest.approx(0.2)
    assert bias(_frame([4, 4], [3, 3])) == pytest.approx(-0.25)


def test_undefined_without_demand() -> None:
    frame = _frame([0, 0], [1, 0])
    assert math.isnan(wape(frame))
    assert math.isnan(bias(frame))


def test_pinball_loss() -> None:
    frame = _frame([10, 10], [8, 13]).rename({"forecast": "q"})
    # Under by 2 costs 0.9 * 2; over by 3 costs 0.1 * 3.
    assert pinball_loss(frame, "q", 0.9) == pytest.approx((0.9 * 2 + 0.1 * 3) / 2)
    assert pinball_loss(frame, "q", 0.5) == pytest.approx((0.5 * 2 + 0.5 * 3) / 2)


def test_coverage() -> None:
    frame = _frame([1, 5, 9, 3], [4, 4, 4, 3]).rename({"forecast": "q"})
    assert coverage(frame, "q") == pytest.approx(0.5)
