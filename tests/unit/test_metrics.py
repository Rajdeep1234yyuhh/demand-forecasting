import math

import polars as pl
import pytest

from forecasting.evaluation.metrics import bias, wape


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
