"""Pandera schemas for raw and processed M5 data.

Raw schemas validate files as read from disk, with dtypes fixed at read time, so a dtype
mismatch is a data error rather than something coercion silently hides.
"""

from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaError, SchemaErrors
from pandera.polars import PolarsData

STATES = ("CA", "TX", "WI")
STORES = ("CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3")
CATEGORIES = ("FOODS", "HOBBIES", "HOUSEHOLD")
DEPARTMENTS = (
    "FOODS_1",
    "FOODS_2",
    "FOODS_3",
    "HOBBIES_1",
    "HOBBIES_2",
    "HOUSEHOLD_1",
    "HOUSEHOLD_2",
)
EVENT_TYPES = ("Cultural", "National", "Religious", "Sporting")
ID_COLUMNS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")

_ITEM_PATTERN = r"^(FOODS|HOBBIES|HOUSEHOLD)_\d_\d{3}$"
_ID_PATTERN = r"^(FOODS|HOBBIES|HOUSEHOLD)_\d_\d{3}_(CA|TX|WI)_\d_(evaluation|validation)$"
_DAY_PATTERN = r"^d_\d+$"


class DataValidationError(Exception):
    """Raised when input data violates a schema or a cross-table rule."""


class CalendarSchema(pa.DataFrameModel):
    date: pl.Date = pa.Field(unique=True)
    wm_yr_wk: pl.Int32 = pa.Field(gt=0)
    weekday: pl.String
    wday: pl.Int8 = pa.Field(ge=1, le=7)
    month: pl.Int8 = pa.Field(ge=1, le=12)
    year: pl.Int16 = pa.Field(ge=2000)
    d: pl.String = pa.Field(unique=True, str_matches=_DAY_PATTERN)
    event_name_1: pl.String = pa.Field(nullable=True)
    event_type_1: pl.String = pa.Field(nullable=True, isin=EVENT_TYPES)
    event_name_2: pl.String = pa.Field(nullable=True)
    event_type_2: pl.String = pa.Field(nullable=True, isin=EVENT_TYPES)
    snap_CA: pl.Int8 = pa.Field(isin=(0, 1))  # noqa: N815 - source column name
    snap_TX: pl.Int8 = pa.Field(isin=(0, 1))  # noqa: N815
    snap_WI: pl.Int8 = pa.Field(isin=(0, 1))  # noqa: N815

    class Config:
        strict = True

    @pa.dataframe_check
    @classmethod
    def dates_are_contiguous(cls, data: PolarsData) -> pl.LazyFrame:
        """Rows must be consecutive days, in order, with no gaps."""
        one_day = pl.duration(days=1)
        return data.lazyframe.select(pl.col("date").diff().fill_null(one_day) == one_day)

    @pa.dataframe_check
    @classmethod
    def day_index_matches_row_order(cls, data: PolarsData) -> pl.LazyFrame:
        """``d_N`` must be the N-th row, so day numbers map one-to-one onto dates."""
        day_number = pl.col("d").str.strip_prefix("d_").cast(pl.Int32)
        return data.lazyframe.select(day_number == pl.int_range(1, pl.len() + 1, dtype=pl.Int32))


class PricesSchema(pa.DataFrameModel):
    store_id: pl.String = pa.Field(isin=STORES)
    item_id: pl.String = pa.Field(str_matches=_ITEM_PATTERN)
    wm_yr_wk: pl.Int32 = pa.Field(gt=0)
    sell_price: pl.Float64 = pa.Field(gt=0)

    class Config:
        strict = True
        unique = ["store_id", "item_id", "wm_yr_wk"]  # noqa: RUF012 - pandera config


class SalesIdsSchema(pa.DataFrameModel):
    """Identifier columns of the wide sales file. Day columns are checked after reshaping."""

    id: pl.String = pa.Field(unique=True, str_matches=_ID_PATTERN)
    item_id: pl.String = pa.Field(str_matches=_ITEM_PATTERN)
    dept_id: pl.String = pa.Field(isin=DEPARTMENTS)
    cat_id: pl.String = pa.Field(isin=CATEGORIES)
    store_id: pl.String = pa.Field(isin=STORES)
    state_id: pl.String = pa.Field(isin=STATES)

    class Config:
        strict = False

    @pa.dataframe_check
    @classmethod
    def hierarchy_is_consistent(cls, data: PolarsData) -> pl.LazyFrame:
        """Each level must nest inside its parent, e.g. FOODS_1_001 belongs to FOODS_1."""
        return data.lazyframe.select(
            pl.col("id").str.starts_with(pl.col("item_id") + "_" + pl.col("store_id") + "_")
            & pl.col("item_id").str.starts_with(pl.col("dept_id") + "_")
            & pl.col("dept_id").str.starts_with(pl.col("cat_id") + "_")
            & pl.col("store_id").str.starts_with(pl.col("state_id") + "_")
        )


class SalesLongSchema(pa.DataFrameModel):
    """Processed sales: one row per series per day.

    ``(id, day)`` uniqueness is guaranteed by construction (unique ids x unique day columns),
    so it is not re-checked here; doing so on ~59M rows would dominate ingestion time.
    """

    id: pl.String
    item_id: pl.String
    dept_id: pl.String
    cat_id: pl.String
    store_id: pl.String
    state_id: pl.String
    date: pl.Date
    day: pl.Int16 = pa.Field(ge=1)
    sales: pl.Int16 = pa.Field(ge=0)

    class Config:
        strict = True


def validate(frame: pl.DataFrame, schema: type[pa.DataFrameModel], name: str) -> pl.DataFrame:
    """Validate ``frame``, collecting every failure into one ``DataValidationError``."""
    try:
        return cast(pl.DataFrame, schema.validate(frame, lazy=True))
    except (SchemaError, SchemaErrors) as exc:
        raise DataValidationError(f"{name} failed validation:\n{exc}") from exc
