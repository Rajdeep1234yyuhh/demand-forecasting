import polars as pl
import pytest

from forecasting.features.price import PRICE_FEATURES, price_features


def _prices(rows: list[tuple[str, str, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "store_id": pl.String,
            "item_id": pl.String,
            "wm_yr_wk": pl.Int32,
            "sell_price": pl.Float64,
        },
        orient="row",
    )


def _series(features: pl.DataFrame, store: str, item: str, column: str) -> list[float | None]:
    selected = features.filter((pl.col("store_id") == store) & (pl.col("item_id") == item))
    return selected.sort("wm_yr_wk").get_column(column).to_list()


PRICES = _prices(
    [
        ("CA_1", "FOODS_1_001", 11101, 2.0),
        ("CA_1", "FOODS_1_001", 11102, 2.5),
        ("CA_1", "FOODS_1_001", 11103, 2.0),
        ("CA_1", "FOODS_1_001", 11104, 3.0),
        # Same item in another store: must not mix with CA_1.
        ("TX_1", "FOODS_1_001", 11101, 10.0),
        ("TX_1", "FOODS_1_001", 11102, 5.0),
        # Another item in the same store.
        ("CA_1", "FOODS_1_002", 11103, 4.0),
    ]
)


def test_columns() -> None:
    assert price_features(PRICES).columns == ["store_id", "item_id", "wm_yr_wk", *PRICE_FEATURES]


def test_relative_price_against_highest_price_so_far() -> None:
    features = price_features(PRICES)
    assert _series(features, "CA_1", "FOODS_1_001", "price_relative") == pytest.approx(
        [1.0, 1.0, 0.8, 1.0]
    )
    assert _series(features, "TX_1", "FOODS_1_001", "price_relative") == pytest.approx([1.0, 0.5])


def test_week_over_week_change() -> None:
    changes = _series(price_features(PRICES), "CA_1", "FOODS_1_001", "price_change")
    assert changes[0] is None
    assert changes[1:] == pytest.approx([0.25, -0.2, 0.5])
    assert _series(price_features(PRICES), "CA_1", "FOODS_1_002", "price_change") == [None]


def test_independent_of_input_order() -> None:
    shuffled = PRICES.sample(fraction=1.0, shuffle=True, seed=7)
    assert price_features(shuffled).equals(price_features(PRICES))


def test_accepts_lazy_input() -> None:
    assert price_features(PRICES.lazy()).equals(price_features(PRICES))
