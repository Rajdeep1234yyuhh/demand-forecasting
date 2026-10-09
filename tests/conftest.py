"""Shared fixtures: a tiny, M5-shaped dataset that mirrors the real files' structure."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from forecasting.utils.config import DataConfig

N_DAYS = 14
START = date(2011, 1, 29)

SERIES: dict[str, list[int]] = {
    # Steady seller.
    "FOODS_1_001_CA_1_evaluation": [3, 2, 4, 3, 5, 2, 3, 4, 3, 2, 4, 3, 5, 3],
    # Launches on day 5, then sells intermittently.
    "HOBBIES_1_001_CA_1_evaluation": [0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0],
    "FOODS_1_001_TX_1_evaluation": [1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0],
    # Never sells, so it has no prices, like unlaunched items in M5.
    "HOBBIES_1_001_TX_1_evaluation": [0] * N_DAYS,
}


@dataclass
class M5Fixture:
    calendar: pl.DataFrame
    prices: pl.DataFrame
    sales: pl.DataFrame

    def write(self, raw_dir: Path) -> None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.calendar.write_csv(raw_dir / "calendar.csv")
        self.prices.write_csv(raw_dir / "sell_prices.csv")
        self.sales.write_csv(raw_dir / "sales_train_evaluation.csv")

    @property
    def day_columns(self) -> list[str]:
        return [column for column in self.sales.columns if column.startswith("d_")]


def _calendar() -> pl.DataFrame:
    dates = [START + timedelta(days=offset) for offset in range(N_DAYS)]
    return pl.DataFrame(
        {
            "date": dates,
            "wm_yr_wk": [11101 if offset < 7 else 11102 for offset in range(N_DAYS)],
            "weekday": [d.strftime("%A") for d in dates],
            "wday": [(d.weekday() - 5) % 7 + 1 for d in dates],  # M5: 1 = Saturday
            "month": [d.month for d in dates],
            "year": [d.year for d in dates],
            "d": [f"d_{offset + 1}" for offset in range(N_DAYS)],
            "event_name_1": ["SuperBowl" if offset == 8 else None for offset in range(N_DAYS)],
            "event_type_1": ["Sporting" if offset == 8 else None for offset in range(N_DAYS)],
            "event_name_2": [None] * N_DAYS,
            "event_type_2": [None] * N_DAYS,
            "snap_CA": [1 if 2 <= offset <= 4 else 0 for offset in range(N_DAYS)],
            "snap_TX": [1 if 3 <= offset <= 5 else 0 for offset in range(N_DAYS)],
            "snap_WI": [1 if 1 <= offset <= 3 else 0 for offset in range(N_DAYS)],
        },
        schema_overrides={
            "wm_yr_wk": pl.Int32,
            "wday": pl.Int8,
            "month": pl.Int8,
            "year": pl.Int16,
            "event_name_1": pl.String,
            "event_type_1": pl.String,
            "event_name_2": pl.String,
            "event_type_2": pl.String,
            "snap_CA": pl.Int8,
            "snap_TX": pl.Int8,
            "snap_WI": pl.Int8,
        },
    )


def _sales() -> pl.DataFrame:
    rows: dict[str, list[object]] = {
        "id": [],
        "item_id": [],
        "dept_id": [],
        "cat_id": [],
        "store_id": [],
        "state_id": [],
    }
    for series_id in SERIES:
        parts = series_id.split("_")
        cat, dept_no, item_no, state, store_no = parts[0], parts[1], parts[2], parts[3], parts[4]
        rows["id"].append(series_id)
        rows["item_id"].append(f"{cat}_{dept_no}_{item_no}")
        rows["dept_id"].append(f"{cat}_{dept_no}")
        rows["cat_id"].append(cat)
        rows["store_id"].append(f"{state}_{store_no}")
        rows["state_id"].append(state)
    frame = pl.DataFrame(rows)
    days = pl.DataFrame(
        {f"d_{day + 1}": [values[day] for values in SERIES.values()] for day in range(N_DAYS)},
        schema={f"d_{day + 1}": pl.Int16 for day in range(N_DAYS)},
    )
    return pl.concat([frame, days], how="horizontal")


def _prices() -> pl.DataFrame:
    pairs = [("CA_1", "FOODS_1_001"), ("CA_1", "HOBBIES_1_001"), ("TX_1", "FOODS_1_001")]
    return pl.DataFrame(
        {
            "store_id": [store for store, _ in pairs for _ in (0, 1)],
            "item_id": [item for _, item in pairs for _ in (0, 1)],
            "wm_yr_wk": [11101, 11102] * len(pairs),
            "sell_price": [2.5, 2.5, 9.99, 8.99, 2.75, 2.75],
        },
        schema_overrides={"wm_yr_wk": pl.Int32},
    )


@pytest.fixture
def m5() -> M5Fixture:
    return M5Fixture(calendar=_calendar(), prices=_prices(), sales=_sales())


@pytest.fixture
def data_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        root=tmp_path,
        raw_prefix="raw/m5",
        processed_prefix="processed/m5",
        max_missing_price_ratio=0.0,
    )
