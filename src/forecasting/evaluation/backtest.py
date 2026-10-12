"""Rolling-origin backtesting.

Each fold trains on history up to a cutoff and scores the following ``horizon`` days. Test
windows are non-overlapping and end at the last available day, mimicking periodic
retraining in production. Every report carries the data version, so each number is
traceable to the exact data that produced it.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
from mlflow.exceptions import MlflowException

from forecasting.data.dataset import Dataset
from forecasting.data.schemas import STATES, DataValidationError
from forecasting.evaluation.hierarchy import HIERARCHY_COLUMNS, LEVELS, HierarchicalEvaluator
from forecasting.evaluation.metrics import coverage, pinball_loss
from forecasting.models.base import ExplainsFeatures, Forecaster
from forecasting.models.registry import MODEL_NAMES, make_model
from forecasting.models.uncertainty import NegativeBinomialUncertainty, quantile_column
from forecasting.tracking import (
    MlflowTracker,
    RunRecord,
    code_version,
    flatten_params,
    log_nested,
)
from forecasting.utils.config import AppConfig, get_secrets, load_config
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


def _quantile_metrics(
    aligned: pl.DataFrame, uncertainty: NegativeBinomialUncertainty, levels: Sequence[float]
) -> dict[str, float]:
    """Coverage and pinball loss of each quantile level, at the bottom level."""
    with_quantiles = uncertainty.quantiles(aligned, levels)
    metrics: dict[str, float] = {}
    for level in levels:
        column = quantile_column(level)
        metrics[f"coverage_{column}"] = coverage(with_quantiles, column)
        metrics[f"pinball_{column}"] = pinball_loss(with_quantiles, column, level)
    return metrics


def run_backtest(
    dataset: Dataset,
    models: Sequence[Forecaster],
    *,
    horizon: int,
    n_folds: int,
    uncertainty: NegativeBinomialUncertainty | None = None,
    levels: Sequence[float] = (),
) -> pl.DataFrame:
    """Return one row per fold and model, with every metric and fit-and-predict time.

    With ``uncertainty``, each model's point forecasts also become quantiles at ``levels``,
    scored for coverage and pinball loss. Uncertainty is refitted on each fold's training
    data, like the models.
    """
    names = [model.name for model in models]
    if not names:
        raise ValueError("at least one model is required")
    if len(set(names)) != len(names):
        raise ValueError(f"model names must be unique, got {names}")
    if uncertainty is not None and not levels:
        raise ValueError("quantile levels are required when uncertainty is given")

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
        if uncertainty is not None:
            uncertainty.fit(train)
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
            aligned = _align(actuals, forecasts, model.name)
            evaluation = evaluator.score(aligned)
            row: dict[str, object] = {
                "fold": fold.index,
                "cutoff": fold.cutoff,
                "model": model.name,
                "seconds": seconds,
                **evaluation.as_row(),
            }
            if uncertainty is not None:
                row.update(_quantile_metrics(aligned, uncertainty, levels))
            rows.append(row)
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


def _calibration_section(results: pl.DataFrame, models: Sequence[str]) -> list[str]:
    """Coverage and pinball loss per quantile, if the backtest scored quantiles."""
    columns = sorted(
        c.removeprefix("coverage_") for c in results.columns if c.startswith("coverage_")
    )
    if not columns:
        return []
    means = results.group_by("model").agg(
        pl.col(f"{metric}_{column}").mean()
        for metric in ("coverage", "pinball")
        for column in columns
    )
    by_model = {row["model"]: row for row in means.iter_rows(named=True)}
    return [
        "",
        "## Quantile calibration, bottom level, mean across folds",
        "",
        "A calibrated quantile covers about its level of actuals. Lower pinball loss is better.",
        "",
        *_markdown_table(
            ["Model", *(f"{c} coverage" for c in columns), *(f"{c} pinball" for c in columns)],
            [
                [
                    model,
                    *(f"{by_model[model][f'coverage_{c}']:.1%}" for c in columns),
                    *(f"{by_model[model][f'pinball_{c}']:.4f}" for c in columns),
                ]
                for model in models
            ],
        ),
    ]


def _importance_section(importance: Mapping[str, pl.DataFrame], top: int = 10) -> list[str]:
    """Top features per model by share of gain, averaged over stores (last fold)."""
    lines: list[str] = []
    for model, frame in importance.items():
        ranked = (
            frame.group_by("feature")
            .agg(pl.col("share").mean())
            .sort("share", "feature", descending=[True, False])
            .head(top)
        )
        lines += [
            "",
            f"## {model}: top features, last fold",
            "",
            "Share of total gain, averaged over stores.",
            "",
            *_markdown_table(
                ["Feature", "Share"],
                [[row["feature"], f"{row['share']:.1%}"] for row in ranked.iter_rows(named=True)],
            ),
        ]
    return lines


def render_summary(
    results: pl.DataFrame,
    *,
    data_version: str,
    scope: str,
    horizon: int,
    importance: Mapping[str, pl.DataFrame] | None = None,
) -> str:
    """Markdown report: headline metrics, WRMSSE by level, calibration, feature importance."""
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
        *_calibration_section(results, models),
        *_importance_section(importance or {}),
    ]
    return "\n".join(lines) + "\n"


def write_report(
    results: pl.DataFrame,
    reports_dir: Path,
    *,
    data_version: str,
    scope: str,
    horizon: int,
    importance: Mapping[str, pl.DataFrame] | None = None,
) -> Path:
    """Write results, feature importance, and a summary to a new, timestamped run directory."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = reports_dir / "backtest" / data_version / stamp
    run_dir.mkdir(parents=True)
    results.write_parquet(run_dir / "results.parquet")
    for model, frame in (importance or {}).items():
        frame.write_parquet(run_dir / f"feature_importance_{model}.parquet")
    summary = render_summary(
        results, data_version=data_version, scope=scope, horizon=horizon, importance=importance
    )
    (run_dir / "summary.md").write_text(summary, "utf-8")
    return run_dir


def backtest_runs(
    results: pl.DataFrame, *, config: AppConfig, data_version: str, scope: str, run_dir: Path
) -> tuple[RunRecord, list[RunRecord]]:
    """Tracking runs for one backtest: a parent with the report, one child per model.

    Every run carries the full config, data version, and code version, so any number in
    MLflow can be traced back and reproduced. Child metrics are means across folds, plus
    each fold's value under ``fold/<metric>`` with the fold number as the step.
    """
    params = flatten_params(config.model_dump(mode="json"))
    tags = {"kind": "backtest", "data_version": data_version, "scope": scope}
    tags["code_version"] = code_version()
    parent = RunRecord(
        name=f"backtest {scope}",
        params=params,
        tags=tags,
        artifacts=sorted(path for path in run_dir.iterdir() if path.is_file()),
    )
    metric_columns = [c for c in results.columns if c not in ("fold", "cutoff", "model")]
    children = []
    for model in sorted(set(results.get_column("model").to_list())):
        folds = results.filter(pl.col("model") == model).sort("fold")
        means = folds.select(pl.col(metric_columns).mean()).row(0, named=True)
        children.append(
            RunRecord(
                name=model,
                params={**params, "model": model},
                metrics={key: float(value) for key, value in means.items()},
                step_metrics=[
                    (f"fold/{column}", float(row[column]), int(row["fold"]))
                    for row in folds.iter_rows(named=True)
                    for column in metric_columns
                ],
                tags={**tags, "model": model},
            )
        )
    return parent, children


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest models with rolling-origin folds.")
    parser.add_argument(
        "--config",
        nargs="+",
        type=Path,
        default=[Path("configs/base.yaml"), Path("configs/inventory.yaml")],
        help="Config files, merged in order.",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
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
        models = [make_model(name, config, dataset) for name in args.models]
        results = run_backtest(
            dataset,
            models,
            horizon=config.forecast.horizon_days,
            n_folds=config.forecast.backtest_folds,
            uncertainty=NegativeBinomialUncertainty(config.models.uncertainty.window_days),
            levels=config.forecast.quantiles,
        )
    except (DataValidationError, FileNotFoundError, ValueError) as exc:
        log.error("backtest.failed", error=str(exc))
        return 1

    importance = {
        model.name: model.feature_importance()
        for model in models
        if isinstance(model, ExplainsFeatures)
    }
    run_dir = write_report(
        results,
        config.reports_dir,
        data_version=dataset.manifest.data_version,
        scope=state or "all states",
        horizon=config.forecast.horizon_days,
        importance=importance,
    )
    log.info("backtest.done", report=str(run_dir))

    tracking_uri = get_secrets().mlflow_tracking_uri
    if tracking_uri is None:
        log.info("tracking.skipped", reason="DF_MLFLOW_TRACKING_URI is not set")
        return 0
    parent, children = backtest_runs(
        results,
        config=config,
        data_version=dataset.manifest.data_version,
        scope=state or "all states",
        run_dir=run_dir,
    )
    try:
        run_id = log_nested(
            MlflowTracker(tracking_uri, config.tracking.experiment), parent, children
        )
    except (MlflowException, OSError) as exc:
        # The report is already on disk, so nothing is lost; the run is marked failed.
        log.error("tracking.failed", error=str(exc), report=str(run_dir))
        return 1
    log.info("tracking.done", run_id=run_id, experiment=config.tracking.experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
