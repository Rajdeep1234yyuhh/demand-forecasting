"""Processed data, loaded lazily, as models and evaluation consume it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from forecasting.data.ingest import scan_sales
from forecasting.data.manifest import Manifest
from forecasting.data.schemas import STORES


@dataclass(frozen=True)
class Dataset:
    """Sales, calendar, and prices from one ingestion run, optionally limited to one state."""

    sales: pl.LazyFrame
    calendar: pl.DataFrame
    prices: pl.LazyFrame
    manifest: Manifest
    state: str | None = None

    @classmethod
    def load(cls, processed_dir: Path, state: str | None = None) -> Dataset:
        manifest_path = processed_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{manifest_path} not found. Run `make ingest`.")
        sales = scan_sales(processed_dir)
        prices = pl.scan_parquet(processed_dir / "prices.parquet")
        if state is not None:
            # Filtering on the partition column lets Polars skip other stores' files.
            stores = [store for store in STORES if store.startswith(f"{state}_")]
            sales = sales.filter(pl.col("store_id").is_in(stores))
            prices = prices.filter(pl.col("store_id").is_in(stores))
        return cls(
            sales=sales,
            calendar=pl.read_parquet(processed_dir / "calendar.parquet"),
            prices=prices,
            manifest=Manifest.model_validate_json(manifest_path.read_text("utf-8")),
            state=state,
        )
