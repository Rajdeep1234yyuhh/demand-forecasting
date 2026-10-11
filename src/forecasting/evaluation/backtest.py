"""Rolling-origin backtesting.

Each fold trains on history up to a cutoff and scores the following ``horizon`` days. Test
windows are non-overlapping and end at the last available day, mimicking periodic
retraining in production. Every report carries the data version, so each number is
traceable to the exact data that produced it.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from forecasting.data.ingest import scan_sales
from forecasting.data.manifest import Manifest
from forecasting.data.schemas import STATES, STORES, DataValidationError
from forecasting.evaluation.hierarchy import HIERARCHY_COLUMNS, LEVELS, HierarchicalEvaluator
from forecasting.models.base import Forecaster
from forecasting.models.baselines import BASELINES, make_baseline
from forecasting.utils.config import load_config
from forecasting.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

_SUMMARY_METRICS = ("wrmsse", "wape", "mase", "bias", "seconds")


@dataclass(frozen=True)
class Fold:
    index: int
    cutoff: date  # last training day
    end: date  # last test day

    @property
    def start(self) -> date:
        return self.cutoff + timedelta(days=1)


def make_folds(last_date: date, horizon: int, n_folds: int) -> list[Fold]:
    """Non-overlapping test windows of ``horizon`` days ending at ``last_date``, oldest first."""
    if horizon <= 0 or n_folds <= 0:
        raise ValueError("horizon and n_folds must be positive")
    folds = []
    for index in range(1, n_folds + 1):
        end = last_date - timedelta(days=(n_folds - index) * horizon)
        folds.append(Fold(index=index, cutoff=end - timedelta(days=horizon), end=end))
    return folds


@dataclass(frozen=True)
class Dataset:
    """Processed data for one backtest, optionally restricted to a single state."""

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


def _align(actuals: pl.DataFrame, forecasts: pl.DataFrame, model: str) -> pl.DataFrame:
    """Join forecasts to actuals, failing if any series-day is missing or unexpected."""
    keys = ["id", "date"]
    missing = actuals.join(forecasts, on=keys, how="anti").height
    unexpected = forecasts.join(actuals, on=keys, how="anti").height
    if missing or unexpected:
        raise DataValidationError(
            f"{model} forecasts do not match the test window: "
            f"{missing} series-days missing, {unexpected} unexpected"
        )
    return actuals.join(forecasts, on=keys, how="inner")


def run_backtest(
    dataset: Dataset, models: Sequence[Forecaster], *, horizon: int, n_folds: int
) -> pl.DataFrame:
    """Return one row per fold and model, with every metric and fit-and-predict time."""
    names = [model.name for model in models]
    if not names:
        raise ValueError("at least one model is required")
    if len(set(names)) != len(names):
        raise ValueError(f"model names must be unique, got {names}")

    first, last = (
        dataset.sales.select(
            pl.col("date").min().alias("first"), pl.col("date").max().alias("last")
        )
        .collect(engine="streaming")
        .row(0)
    )
    if last is None:
        raise DataValidationError("no sales data to backtest")
    folds = make_folds(last, horizon, n_folds)
    if folds[0].cutoff < first:
        raise ValueError(
            f"{n_folds} folds of {horizon} days need more history than {first} to {last}"
        )

    rows: list[dict[str, object]] = []
    for fold in folds:
        started = time.perf_counter()
        train = dataset.sales.filter(pl.col("date") <= fold.cutoff)
        actuals = (
            dataset.sales.filter(pl.col("date").is_between(fold.start, fold.end))
            .select(
                "id",
                *HIERARCHY_COLUMNS,
                "date",
                pl.col("sales").cast(pl.Float64).alias("actual"),
            )
            .collect(engine="streaming")
        )
        evaluator = HierarchicalEvaluator.from_history(
            train, dataset.calendar, dataset.prices, weight_days=horizon
        )
        log.info(
            "backtest.fold",
            fold=fold.index,
            cutoff=str(fold.cutoff),
            test_rows=actuals.height,
            setup_seconds=round(time.perf_counter() - started, 2),
        )
        for model in models:
            model_started = time.perf_counter()
            forecasts = model.fit(train).predict(horizon)
            seconds = time.perf_counter() - model_started
            evaluation = evaluator.score(_align(actuals, forecasts, model.name))
            rows.append(
                {
                    "fold": fold.index,
                    "cutoff": fold.cutoff,
                    "model": model.name,
                    "seconds": seconds,
                    **evaluation.as_row(),
                }
            )
            log.info(
                "backtest.model",
                fold=fold.index,
                model=model.name,
                wrmsse=round(evaluation.wrmsse, 4),
                wape=round(evaluation.wape, 4),
                seconds=round(seconds, 2),
            )
    return pl.DataFrame(rows)


def summarize(results: pl.DataFrame) -> pl.DataFrame:
    """Mean of each metric across folds, best model first."""
    return results.group_by("model").agg(pl.col(_SUMMARY_METRICS).mean()).sort("wrmsse")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    align = ["---", *["---:"] * (len(headers) - 1)]
    return [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(align)} |",
        *(f"| {' | '.join(row)} |" for row in rows),
    ]


def render_summary(results: pl.DataFrame, *, data_version: str, scope: str, horizon: int) -> str:
    """Markdown report: headline metrics per model and WRMSSE by hierarchy level."""
    summary = summarize(results)
    models = summary.get_column("model").to_list()
    cutoffs = sorted(set(results.get_column("cutoff").to_list()))
    by_level = results.group_by("model").agg(
        pl.col(f"wrmsse_{level.name}").mean() for level in LEVELS
    )
    level_means = {row["model"]: row for row in by_level.iter_rows(named=True)}
    lines = [
        "# Backtest report",
        "",
        f"- Data version: `{data_version}`",
        f"- Scope: {scope}",
        f"- Folds: {len(cutoffs)} x {horizon} days; training cutoffs "
        f"{', '.join(str(cutoff) for cutoff in cutoffs)}",
        "",
        "## Mean across folds",
        "",
        *_markdown_table(
            ["Model", "WRMSSE", "WAPE", "MASE", "Bias", "Seconds"],
            [
                [
                    row["model"],
                    f"{row['wrmsse']:.4f}",
                    f"{row['wape']:.2%}",
                    f"{row['mase']:.3f}",
                    f"{row['bias']:+.2%}",
                    f"{row['seconds']:.1f}",
                ]
                for row in summary.iter_rows(named=True)
            ],
        ),
        "",
        "## WRMSSE by level, mean across folds",
        "",
        *_markdown_table(
            ["Level", *models],
            [
                [level.name, *(f"{level_means[m][f'wrmsse_{level.name}']:.4f}" for m in models)]
                for level in LEVELS
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def write_report(
    results: pl.DataFrame, reports_dir: Path, *, data_version: str, scope: str, horizon: int
) -> Path:
    """Write results and a summary to a new, timestamped run directory."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = reports_dir / "backtest" / data_version / stamp
    run_dir.mkdir(parents=True)
    results.write_parquet(run_dir / "results.parquet")
    summary = render_summary(results, data_version=data_version, scope=scope, horizon=horizon)
    (run_dir / "summary.md").write_text(summary, "utf-8")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest models with rolling-origin folds.")
    parser.add_argument(
        "--config",
        nargs="+",
        type=Path,
        default=[Path("configs/base.yaml"), Path("configs/inventory.yaml")],
        help="Config files, merged in order.",
    )
    parser.add_argument("--models", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument(
        "--state",
        choices=[*STATES, "ALL"],
        help="Override data.train_subset_state. ALL uses every ingested state.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    config = load_config(*args.config)
    configure_logging(args.log_level, json_logs=config.env != "dev")
    if args.state is None:
        state = config.data.train_subset_state
    else:
        state = None if args.state == "ALL" else args.state
    try:
        dataset = Dataset.load(config.data.processed_dir, state)
        models = [make_baseline(name, config.forecast) for name in args.models]
        results = run_backtest(
            dataset,
            models,
            horizon=config.forecast.horizon_days,
            n_folds=config.forecast.backtest_folds,
        )
    except (DataValidationError, FileNotFoundError, ValueError) as exc:
        log.error("backtest.failed", error=str(exc))
        return 1

    run_dir = write_report(
        results,
        config.reports_dir,
        data_version=dataset.manifest.data_version,
        scope=state or "all states",
        horizon=config.forecast.horizon_days,
    )
    log.info("backtest.done", report=str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
