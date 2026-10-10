"""Ingest raw M5 CSVs into validated, versioned Parquet.

Steps: hash sources -> read with fixed dtypes -> validate -> reshape sales to long format
one store at a time -> cross-table checks -> write to a staging directory -> swap it into
place atomically. Per-store processing bounds peak memory to roughly one store's data, and a
failed run never leaves a half-written dataset behind.
"""

from __future__ import annotations

import argparse
import shutil
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from forecasting import __version__
from forecasting.data.manifest import (
    Manifest,
    OutputTable,
    compute_data_version,
    describe_file,
)
from forecasting.data.schemas import (
    ID_COLUMNS,
    STATES,
    CalendarSchema,
    DataValidationError,
    PricesSchema,
    SalesIdsSchema,
    SalesLongSchema,
    validate,
)
from forecasting.utils.config import DataConfig, load_config
from forecasting.utils.logging import configure_logging, get_logger

DATASET = "m5-forecasting-accuracy"
# Bump whenever the transformation changes, so the data version changes with it.
TRANSFORM_VERSION = "1"

CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"
SALES_FILE = "sales_train_evaluation.csv"
SOURCE_FILES = (CALENDAR_FILE, PRICES_FILE, SALES_FILE)

_CALENDAR_DTYPES: dict[str, type[pl.DataType]] = {
    "date": pl.Date,
    "wm_yr_wk": pl.Int32,
    "weekday": pl.String,
    "wday": pl.Int8,
    "month": pl.Int8,
    "year": pl.Int16,
    "d": pl.String,
    "event_name_1": pl.String,
    "event_type_1": pl.String,
    "event_name_2": pl.String,
    "event_type_2": pl.String,
    "snap_CA": pl.Int8,
    "snap_TX": pl.Int8,
    "snap_WI": pl.Int8,
}
_PRICES_DTYPES: dict[str, type[pl.DataType]] = {
    "store_id": pl.String,
    "item_id": pl.String,
    "wm_yr_wk": pl.Int32,
    "sell_price": pl.Float64,
}

log = get_logger(__name__)


def _write_parquet(frame: pl.DataFrame, path: Path) -> None:
    frame.write_parquet(path, compression="zstd", statistics=True)


def read_calendar(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(path, schema_overrides=_CALENDAR_DTYPES)
    return validate(frame, CalendarSchema, "calendar")


def read_prices(path: Path, state: str | None = None) -> pl.DataFrame:
    lazy = pl.scan_csv(path, schema_overrides=_PRICES_DTYPES)
    if state is not None:
        lazy = lazy.filter(pl.col("store_id").str.starts_with(f"{state}_"))
    return validate(lazy.collect(), PricesSchema, "prices")


def read_sales_wide(path: Path, state: str | None = None) -> tuple[pl.DataFrame, list[str]]:
    """Read the wide sales file. Day columns are parsed as Int16, so bad values fail loudly."""
    header = pl.read_csv(path, n_rows=0).columns
    missing = [column for column in ID_COLUMNS if column not in header]
    if missing:
        raise DataValidationError(f"sales file is missing identifier columns: {missing}")
    day_columns = [column for column in header if column.startswith("d_")]
    if not day_columns:
        raise DataValidationError("sales file has no day columns (d_1, d_2, ...)")

    dtypes: dict[str, type[pl.DataType]] = dict.fromkeys(ID_COLUMNS, pl.String)
    dtypes.update(dict.fromkeys(day_columns, pl.Int16))
    lazy = pl.scan_csv(path, schema_overrides=dtypes)
    if state is not None:
        lazy = lazy.filter(pl.col("state_id") == state)
    frame = lazy.collect()
    if frame.is_empty():
        raise DataValidationError(f"no sales rows found for state={state}")
    return validate(frame, SalesIdsSchema, "sales"), day_columns


def check_days_covered(day_columns: Sequence[str], calendar: pl.DataFrame) -> None:
    """Every sales day must exist in the calendar, or the reshape would silently drop it."""
    known = set(calendar.get_column("d").to_list())
    missing = [day for day in day_columns if day not in known]
    if missing:
        raise DataValidationError(
            f"{len(missing)} sales day(s) missing from calendar, e.g. {missing[:3]}"
        )


def to_long(
    sales_wide: pl.DataFrame, day_columns: Sequence[str], calendar: pl.DataFrame
) -> pl.DataFrame:
    """Reshape wide sales (one column per day) into one row per series per day."""
    day_map = calendar.lazy().select(
        "d", "date", pl.col("d").str.strip_prefix("d_").cast(pl.Int16).alias("day")
    )
    long = (
        sales_wide.lazy()
        .unpivot(
            index=list(ID_COLUMNS), on=list(day_columns), variable_name="d", value_name="sales"
        )
        .join(day_map, on="d", how="inner")
        .select(*ID_COLUMNS, "date", "day", "sales")
        .collect()
    )
    return validate(long, SalesLongSchema, "sales (long)")


def price_coverage(
    sales_long: pl.DataFrame, calendar: pl.DataFrame, prices: pl.DataFrame
) -> tuple[int, int]:
    """Count positive-sales rows, and those with no price for that store, item, and week."""
    keys = ["store_id", "item_id", "wm_yr_wk"]
    positive = (
        sales_long.lazy()
        .filter(pl.col("sales") > 0)
        .join(calendar.lazy().select("date", "wm_yr_wk"), on="date")
        .select(keys)
    )
    total = positive.select(pl.len()).collect().item()
    unpriced = (
        positive.join(prices.lazy().select(keys), on=keys, how="anti")
        .select(pl.len())
        .collect()
        .item()
    )
    return int(total), int(unpriced)


def _write_store_partition(sales_long: pl.DataFrame, root: Path, store: str) -> None:
    """Write one Hive-style partition (store_id=CA_1/part-0.parquet), sorted for fast scans."""
    directory = root / f"store_id={store}"
    directory.mkdir(parents=True)
    _write_parquet(sales_long.drop("store_id").sort("id", "date"), directory / "part-0.parquet")


def _publish(staging: Path, target: Path) -> None:
    """Swap ``staging`` into ``target``; restore the previous dataset if the swap fails."""
    backup = target.with_name(f".{target.name}.old-{uuid.uuid4().hex}")
    if target.exists():
        target.rename(backup)
    try:
        staging.rename(target)
    except OSError:
        if backup.exists():
            backup.rename(target)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def run_ingest(config: DataConfig, state: str | None = None) -> Manifest:
    """Run ingestion end to end and return the manifest of the published dataset."""
    if state is not None and state not in STATES:
        raise ValueError(f"state must be one of {STATES}, got {state!r}")
    started = time.perf_counter()
    raw_dir, target = config.raw_dir, config.processed_dir
    paths = {name: raw_dir / name for name in SOURCE_FILES}
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"raw files not found: {absent}. Run `make download-data`.")

    log.info("ingest.start", raw_dir=str(raw_dir), state=state or "ALL")
    sources = tuple(describe_file(path) for path in paths.values())

    calendar = read_calendar(paths[CALENDAR_FILE])
    prices = read_prices(paths[PRICES_FILE], state)
    sales_wide, day_columns = read_sales_wide(paths[SALES_FILE], state)
    check_days_covered(day_columns, calendar)
    log.info("ingest.read", series=sales_wide.height, days=len(day_columns), prices=prices.height)

    calendar = calendar.with_columns(pl.col("d").str.strip_prefix("d_").cast(pl.Int16).alias("day"))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.staging-{uuid.uuid4().hex}")
    try:
        staging.mkdir()
        sales_rows = positive = unpriced = 0
        stores = sales_wide.partition_by("store_id", as_dict=True)
        del sales_wide
        for (store,), wide_part in sorted(stores.items()):
            long = to_long(wide_part, day_columns, calendar)
            store_positive, store_unpriced = price_coverage(
                long, calendar, prices.filter(pl.col("store_id") == store)
            )
            _write_store_partition(long, staging / "sales", str(store))
            sales_rows += long.height
            positive += store_positive
            unpriced += store_unpriced
            log.info("ingest.store", store=store, rows=long.height)

        ratio = unpriced / positive if positive else 0.0
        if ratio > config.max_missing_price_ratio:
            raise DataValidationError(
                f"{ratio:.2%} of positive-sales rows have no price "
                f"(limit {config.max_missing_price_ratio:.2%})"
            )

        _write_parquet(calendar, staging / "calendar.parquet")
        _write_parquet(prices, staging / "prices.parquet")
        manifest = Manifest(
            dataset=DATASET,
            data_version=compute_data_version(
                sources, transform_version=TRANSFORM_VERSION, ingest_state=state
            ),
            transform_version=TRANSFORM_VERSION,
            code_version=__version__,
            created_at=datetime.now(UTC),
            ingest_state=state,
            missing_price_ratio=ratio,
            sources=sources,
            outputs=(
                OutputTable(name="calendar", rows=calendar.height),
                OutputTable(name="prices", rows=prices.height),
                OutputTable(name="sales", rows=sales_rows),
            ),
        )
        (staging / "manifest.json").write_text(manifest.model_dump_json(indent=2), "utf-8")
        _publish(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    log.info(
        "ingest.done",
        data_version=manifest.data_version,
        rows=sales_rows,
        missing_price_ratio=round(ratio, 6),
        seconds=round(time.perf_counter() - started, 2),
    )
    return manifest


def scan_sales(processed_dir: Path) -> pl.LazyFrame:
    """Lazily read processed sales, restoring ``store_id`` from the partition path."""
    return pl.scan_parquet(processed_dir / "sales" / "**" / "*.parquet", hive_partitioning=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest raw M5 data into validated Parquet.")
    parser.add_argument(
        "--config",
        nargs="+",
        type=Path,
        default=[Path("configs/base.yaml"), Path("configs/inventory.yaml")],
        help="Config files, merged in order.",
    )
    parser.add_argument("--state", choices=STATES, help="Override data.ingest_state.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    config = load_config(*args.config)
    configure_logging(args.log_level, json_logs=config.env != "dev")
    state = args.state or config.data.ingest_state
    try:
        run_ingest(config.data, state)
    except (DataValidationError, FileNotFoundError) as exc:
        log.error("ingest.failed", error=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
