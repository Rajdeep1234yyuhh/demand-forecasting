"""Weekly price features per store and item.

The retailer sets prices in advance, so they are known for the forecast window too. Each
feature uses only the current and earlier weeks, so a value never depends on later prices.
"""

from __future__ import annotations

import polars as pl

PRICE_FEATURES = ("sell_price", "price_relative", "price_change")


def price_features(prices: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Add relative price and week-over-week change to each store-item-week price.

    - ``price_relative``: price divided by the highest price the item has had in that store
      so far. 1.0 means full price; below 1.0 means a discount.
    - ``price_change``: fractional change from the item's previous priced week; null for
      its first week.
    """
    keys = ["store_id", "item_id"]
    price = pl.col("sell_price")
    return (
        prices.lazy()
        .select(*keys, "wm_yr_wk", price)
        .with_columns(
            (price / price.cum_max().over(keys, order_by="wm_yr_wk")).alias("price_relative"),
            (price / price.shift(1).over(keys, order_by="wm_yr_wk") - 1).alias("price_change"),
        )
        .sort(*keys, "wm_yr_wk")
        .collect()
    )
