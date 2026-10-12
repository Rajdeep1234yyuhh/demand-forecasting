from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from scipy import stats

from forecasting.models.base import NotFittedError
from forecasting.models.uncertainty import NegativeBinomialUncertainty, quantile_column

START = date(2016, 1, 1)
LEVELS = (0.5, 0.9, 0.95)


def _history(series: dict[str, list[int]]) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "id": [sid for sid, values in series.items() for _ in values],
            "date": [
                START + timedelta(days=i) for values in series.values() for i in range(len(values))
            ],
            "sales": [value for values in series.values() for value in values],
        },
        schema_overrides={"sales": pl.Int16},
    ).lazy()


def _forecasts(means: dict[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"id": list(means), "date": [START] * len(means), "forecast": list(means.values())}
    )


def test_quantile_column() -> None:
    assert [quantile_column(level) for level in (0.5, 0.9, 0.975)] == ["q_0.5", "q_0.9", "q_0.975"]


def test_method_of_moments_size() -> None:
    bursty = [0, 0, 0, 8] * 5
    sizes = NegativeBinomialUncertainty(window_days=20).fit(_history({"bursty": bursty})).sizes
    mean, variance = np.mean(bursty), np.var(bursty, ddof=1)
    assert sizes.row(0, named=True)["size"] == pytest.approx(mean**2 / (variance - mean))


def test_not_overdispersed_series_are_poisson() -> None:
    history = _history({"steady": [3] * 20, "never_sold": [0] * 20})
    sizes = NegativeBinomialUncertainty(window_days=20).fit(history).sizes
    assert sizes.get_column("size").null_count() == 2


def test_only_the_recent_window_counts() -> None:
    # Bursty long ago, steady over the last 10 days: the window sees only the steady part.
    history = _history({"s": [0, 20] * 10 + [3] * 10})
    sizes = NegativeBinomialUncertainty(window_days=10).fit(history).sizes
    assert sizes.get_column("size").to_list() == [None]


def test_quantiles_match_scipy() -> None:
    uncertainty = NegativeBinomialUncertainty(window_days=20)
    uncertainty.fit(_history({"bursty": [0, 0, 0, 8] * 5, "steady": [3] * 20}))
    size = uncertainty.sizes.filter(pl.col("id") == "bursty").item(0, "size")
    result = uncertainty.quantiles(_forecasts({"bursty": 2.5, "steady": 3.0}), LEVELS)
    rows = {row["id"]: row for row in result.iter_rows(named=True)}
    for level in LEVELS:
        column = quantile_column(level)
        expected_nb = stats.nbinom.ppf(level, size, size / (size + 2.5))
        assert rows["bursty"][column] == expected_nb
        assert rows["steady"][column] == stats.poisson.ppf(level, 3.0)


def test_quantiles_are_ordered_and_zero_means_zero() -> None:
    uncertainty = NegativeBinomialUncertainty(window_days=20)
    uncertainty.fit(_history({"bursty": [0, 0, 0, 8] * 5, "zero": [0] * 20}))
    result = uncertainty.quantiles(_forecasts({"bursty": 2.5, "zero": 0.0}), LEVELS)
    columns = [quantile_column(level) for level in LEVELS]
    for row in result.select(columns).rows():
        assert list(row) == sorted(row)
    zero = result.filter(pl.col("id") == "zero").select(columns).row(0)
    assert zero == (0.0, 0.0, 0.0)


def test_series_without_history_fall_back_to_poisson() -> None:
    uncertainty = NegativeBinomialUncertainty(window_days=20).fit(_history({"a": [1] * 20}))
    result = uncertainty.quantiles(_forecasts({"new_item": 4.0}), (0.9,))
    assert result.item(0, "q_0.9") == stats.poisson.ppf(0.9, 4.0)


def test_calibrated_on_negative_binomial_data() -> None:
    """With the true mean, estimated sizes give quantiles that cover about their level."""
    rng = np.random.default_rng(42)
    n_series, n_days, true_size, true_mean = 400, 112, 0.8, 2.0
    p = true_size / (true_size + true_mean)
    train = rng.negative_binomial(true_size, p, size=(n_series, n_days))
    ids = [f"s{i}" for i in range(n_series)]
    history = _history(dict(zip(ids, train.tolist(), strict=True)))
    uncertainty = NegativeBinomialUncertainty(window_days=n_days).fit(history)

    future = rng.negative_binomial(true_size, p, size=(n_series, 200))
    result = uncertainty.quantiles(_forecasts(dict.fromkeys(ids, true_mean)), LEVELS)
    for level in LEVELS:
        quantiles = result.sort("id").get_column(quantile_column(level)).to_numpy()
        order = np.argsort(ids)
        covered = (future[order] <= quantiles[:, None]).mean()
        # Counts are discrete, so coverage sits at or a little above the level.
        assert level - 0.03 <= covered <= level + 0.08, (level, covered)


def test_rejects_invalid_use() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        NegativeBinomialUncertainty(window_days=1)
    uncertainty = NegativeBinomialUncertainty(window_days=5)
    with pytest.raises(NotFittedError):
        uncertainty.quantiles(_forecasts({"a": 1.0}), LEVELS)
    with pytest.raises(ValueError, match="empty"):
        uncertainty.fit(_history({"a": []}))
    uncertainty.fit(_history({"a": [1, 2, 3]}))
    with pytest.raises(ValueError, match="at least one"):
        uncertainty.quantiles(_forecasts({"a": 1.0}), ())
    with pytest.raises(ValueError, match="between 0 and 1"):
        uncertainty.quantiles(_forecasts({"a": 1.0}), (1.0,))
