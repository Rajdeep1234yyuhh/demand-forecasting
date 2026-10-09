import json
from pathlib import Path

import polars as pl
import pytest

from forecasting.data.ingest import main, run_ingest, scan_sales, to_long
from forecasting.data.manifest import Manifest, describe_file
from forecasting.data.schemas import DataValidationError
from forecasting.utils.config import DataConfig
from tests.conftest import N_DAYS, SERIES, M5Fixture

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_publishes_validated_dataset(m5: M5Fixture, data_config: DataConfig) -> None:
    m5.write(data_config.raw_dir)
    manifest = run_ingest(data_config)

    out = data_config.processed_dir
    assert {p.name for p in out.iterdir()} == {
        "calendar.parquet",
        "prices.parquet",
        "sales",
        "manifest.json",
    }
    sales = scan_sales(out).collect()
    assert sales.height == len(SERIES) * N_DAYS
    assert sales.schema["sales"] == pl.Int16
    assert set(sales.get_column("store_id").unique()) == {"CA_1", "TX_1"}

    saved = Manifest.model_validate_json((out / "manifest.json").read_text("utf-8"))
    assert saved == manifest
    assert {o.name: o.rows for o in manifest.outputs}["sales"] == sales.height
    sources = {s.name: s for s in manifest.sources}
    assert sources["calendar.csv"] == describe_file(data_config.raw_dir / "calendar.csv")


def test_long_values_match_wide(m5: M5Fixture) -> None:
    long = to_long(m5.sales, m5.day_columns, m5.calendar)
    for series_id, values in SERIES.items():
        got = long.filter(pl.col("id") == series_id).sort("day").get_column("sales").to_list()
        assert got == values
    assert long.get_column("date").min() == m5.calendar.get_column("date").min()


def test_state_filter(m5: M5Fixture, data_config: DataConfig) -> None:
    m5.write(data_config.raw_dir)
    full = run_ingest(data_config)
    ca = run_ingest(data_config, "CA")

    sales = scan_sales(data_config.processed_dir).collect()
    assert set(sales.get_column("state_id").unique()) == {"CA"}
    prices = pl.read_parquet(data_config.processed_dir / "prices.parquet")
    assert prices.get_column("store_id").str.starts_with("CA_").all()
    assert ca.data_version != full.data_version


def test_data_version_is_deterministic(m5: M5Fixture, data_config: DataConfig) -> None:
    m5.write(data_config.raw_dir)
    assert run_ingest(data_config).data_version == run_ingest(data_config).data_version


def test_failed_run_keeps_previous_dataset(m5: M5Fixture, data_config: DataConfig) -> None:
    m5.write(data_config.raw_dir)
    good = run_ingest(data_config)

    m5.sales = m5.sales.with_columns(pl.lit(-1, dtype=pl.Int16).alias("d_3"))
    m5.write(data_config.raw_dir)
    with pytest.raises(DataValidationError, match=r"sales \(long\)"):
        run_ingest(data_config)

    manifest = json.loads((data_config.processed_dir / "manifest.json").read_text("utf-8"))
    assert manifest["data_version"] == good.data_version
    leftovers = [p.name for p in data_config.processed_dir.parent.iterdir() if p.name != "m5"]
    assert leftovers == []


def test_rejects_calendar_gap(m5: M5Fixture, data_config: DataConfig) -> None:
    m5.calendar = m5.calendar.with_columns(
        pl.when(pl.col("d") == "d_5")
        .then(pl.col("date") + pl.duration(days=1))
        .otherwise(pl.col("date"))
        .alias("date")
    )
    m5.write(data_config.raw_dir)
    with pytest.raises(DataValidationError, match="calendar"):
        run_ingest(data_config)


def test_rejects_days_missing_from_calendar(m5: M5Fixture, data_config: DataConfig) -> None:
    m5.calendar = m5.calendar.head(N_DAYS - 1)
    m5.write(data_config.raw_dir)
    with pytest.raises(DataValidationError, match="missing from calendar"):
        run_ingest(data_config)


def test_rejects_inconsistent_hierarchy(m5: M5Fixture, data_config: DataConfig) -> None:
    m5.sales = m5.sales.with_columns(
        pl.when(pl.col("id") == "FOODS_1_001_CA_1_evaluation")
        .then(pl.lit("FOODS_2"))
        .otherwise(pl.col("dept_id"))
        .alias("dept_id")
    )
    m5.write(data_config.raw_dir)
    with pytest.raises(DataValidationError, match="hierarchy_is_consistent"):
        run_ingest(data_config)


def test_rejects_unpriced_sales(m5: M5Fixture, data_config: DataConfig) -> None:
    m5.prices = m5.prices.filter(~((pl.col("store_id") == "TX_1") & (pl.col("wm_yr_wk") == 11102)))
    m5.write(data_config.raw_dir)
    with pytest.raises(DataValidationError, match="no price"):
        run_ingest(data_config)

    lenient = data_config.model_copy(update={"max_missing_price_ratio": 1.0})
    assert run_ingest(lenient).missing_price_ratio > 0


def test_rejects_sales_without_id_columns(m5: M5Fixture, data_config: DataConfig) -> None:
    m5.sales = m5.sales.drop("dept_id")
    m5.write(data_config.raw_dir)
    with pytest.raises(DataValidationError, match="identifier columns"):
        run_ingest(data_config)


def test_rejects_sales_without_day_columns(m5: M5Fixture, data_config: DataConfig) -> None:
    m5.sales = m5.sales.drop(m5.day_columns)
    m5.write(data_config.raw_dir)
    with pytest.raises(DataValidationError, match="no day columns"):
        run_ingest(data_config)


def test_rejects_state_with_no_rows(m5: M5Fixture, data_config: DataConfig) -> None:
    m5.write(data_config.raw_dir)
    with pytest.raises(DataValidationError, match="no sales rows"):
        run_ingest(data_config, "WI")


def test_rejects_unknown_state(data_config: DataConfig) -> None:
    with pytest.raises(ValueError, match="state must be one of"):
        run_ingest(data_config, "NY")


def test_missing_raw_files(data_config: DataConfig) -> None:
    with pytest.raises(FileNotFoundError, match="make download-data"):
        run_ingest(data_config)


def _cli_config(tmp_path: Path) -> list[str]:
    override = tmp_path / "override.yaml"
    override.write_text(f"data:\n  root: {tmp_path}\n  max_missing_price_ratio: 0.0\n", "utf-8")
    return [str(CONFIGS / "base.yaml"), str(CONFIGS / "inventory.yaml"), str(override)]


def test_cli_success(m5: M5Fixture, data_config: DataConfig, tmp_path: Path) -> None:
    m5.write(data_config.raw_dir)
    assert main(["--config", *_cli_config(tmp_path), "--state", "TX"]) == 0
    assert (data_config.processed_dir / "manifest.json").is_file()


def test_cli_failure_returns_nonzero(tmp_path: Path) -> None:
    assert main(["--config", *_cli_config(tmp_path)]) == 1
