# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 · Exploratory data analysis: M5 Walmart sales
#
# Goal: understand the demand patterns that drive modeling choices in Phases 2-4.
# All computation lives in `forecasting.analysis.eda` (unit-tested); this notebook only
# calls those functions and plots the results.
#
# Run `make ingest` first, then `make eda` to execute this notebook end to end.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from forecasting.analysis import eda
from forecasting.data.ingest import scan_sales
from forecasting.data.schemas import CATEGORIES
from forecasting.utils.config import load_config

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
config = load_config(ROOT / "configs/base.yaml", ROOT / "configs/inventory.yaml")
processed = ROOT / config.data.processed_dir

manifest = json.loads((processed / "manifest.json").read_text("utf-8"))
sales = scan_sales(processed)
calendar = pl.read_parquet(processed / "calendar.parquet")
prices = pl.scan_parquet(processed / "prices.parquet")

plt.rcParams.update({"figure.figsize": (11, 4), "axes.grid": True, "grid.alpha": 0.3})
print(f"data_version={manifest['data_version']}  states={manifest['ingest_state'] or 'ALL'}")

# %% [markdown]
# ## 1. Dataset shape
#
# The hierarchy (state → store → category → department → item) defines the levels we can
# forecast and report at.

# %%
counts = eda.hierarchy_counts(sales)
totals = eda.daily_totals(sales)
print(counts)
print(f"days: {totals.height}  ({totals['date'].min()} → {totals['date'].max()})")
print(f"series-days: {sum(o['rows'] for o in manifest['outputs'] if o['name'] == 'sales'):,}")

# %% [markdown]
# ## 2. Zeros and intermittency
#
# Most retail series sell nothing on most days. That decides which metrics are valid
# (percentage errors break on zeros) and which loss functions fit.

# %%
print(f"zero share, all series-days: {eda.zero_share(sales):.1%}")
active_zero_share = eda.zero_share(eda.trim_leading_zeros(sales))
print(f"zero share, after removing pre-launch days: {active_zero_share:.1%}")
for category in CATEGORIES:
    share = eda.zero_share(sales.filter(pl.col("cat_id") == category))
    print(f"  {category:<10} {share:.1%}")

# %%
classes = eda.demand_classification(sales)
series_categories = sales.select("id", "cat_id").unique().collect(engine="streaming")
by_class = (
    classes.join(series_categories, on="id")
    .group_by("cat_id", "demand_class")
    .len()
    .pivot(on="demand_class", index="cat_id", values="len")
    .fill_null(0)
    .sort("cat_id")
)
print(by_class)

order = ["smooth", "erratic", "intermittent", "lumpy"]
shares = classes.group_by("demand_class").len().with_columns(pl.col("len") / pl.col("len").sum())
share_map = dict(zip(shares["demand_class"], shares["len"], strict=True))
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(order, [share_map.get(c, 0.0) for c in order])
ax.set_title("Demand classes (Syntetos-Boylan), share of series")
ax.set_ylabel("share of series")
plt.show()

# %% [markdown]
# ## 3. Trend and level by state

# %%
weekly = (
    eda.daily_totals(sales, by=["state_id"])
    .group_by_dynamic("date", every="1w", group_by="state_id")
    .agg(pl.col("sales").sum())
)
fig, ax = plt.subplots()
for (state,), part in weekly.partition_by("state_id", as_dict=True).items():
    ax.plot(part["date"], part["sales"], label=state)
ax.set_title("Weekly units sold by state")
ax.legend()
plt.show()

# %% [markdown]
# ## 4. Closures and anomalies
#
# Days with near-zero total sales are store closures, not demand. They must be flagged so
# models don't learn them and evaluation doesn't score them.

# %%
closures = totals.filter(pl.col("sales") < 0.01 * pl.col("sales").median())
print(closures)

# %% [markdown]
# ## 5. Weekly seasonality

# %%
profile = eda.weekday_profile(sales, by=["state_id"])
labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, ax = plt.subplots(figsize=(8, 3.5))
for (state,), part in profile.partition_by("state_id", as_dict=True).items():
    ax.plot(labels, part.sort("weekday")["index"], marker="o", label=state)
ax.axhline(1.0, color="grey", linewidth=1)
ax.set_title("Weekday index (1.0 = state average)")
ax.legend()
plt.show()

# %% [markdown]
# ## 6. SNAP effect on food sales
#
# SNAP (food assistance) days differ by state. This is a naive comparison: SNAP days sit in
# the first half of each month, so the lift is confounded with month position. It tells
# us SNAP is worth a feature, not how large its causal effect is.

# %%
lift = eda.snap_lift(sales, calendar, category="FOODS")
print(lift)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(lift["state_id"], lift["lift"])
ax.set_title("FOODS: mean daily units on SNAP vs non-SNAP days (lift)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
plt.show()

# %% [markdown]
# ## 7. Price variation

# %%
print(f"store-items with at least one price change: {eda.price_change_share(prices):.1%}")

# %% [markdown]
# ## 8. Implications for modeling
#
# Confirm each against the outputs above before relying on it:
#
# - **Intermittency** — if most series are intermittent or lumpy: no MAPE; evaluate with
#   WAPE, MASE, and WRMSSE; use a Tweedie or Poisson objective in LightGBM; produce
#   quantile forecasts, since inventory decisions depend on the upper tail.
# - **Pre-launch zeros** — trim them before training and evaluation.
# - **Closures** — flag closure days as features and exclude them from scoring.
# - **Weekly seasonality** — weekday features plus lags at 7 and 28 days.
# - **SNAP** — state-specific SNAP flags as features, especially for FOODS.
# - **Price changes** — if common: relative-price and price-change features in Phase 3.
