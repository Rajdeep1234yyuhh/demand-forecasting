.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check clean download-data ingest eda

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install dependencies and git hooks
	uv sync --frozen
	uv run pre-commit install

lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint issues and format
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Static type checking
	uv run mypy src tests

test: ## Run tests with coverage
	uv run pytest

check: lint typecheck test ## Run all quality gates

clean: ## Remove caches and build artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

download-data: ## Download M5 from Kaggle (needs Kaggle API credentials and accepted rules)
	mkdir -p data/raw/m5
	uv run --group data kaggle competitions download -c m5-forecasting-accuracy -p data/raw/m5
	unzip -o data/raw/m5/m5-forecasting-accuracy.zip -d data/raw/m5
	rm data/raw/m5/m5-forecasting-accuracy.zip

ingest: ## Validate raw data and publish processed Parquet
	uv run df-ingest

eda: ## Execute the EDA notebook (writes notebooks/01_eda.ipynb)
	uv run --group notebook jupytext --to ipynb --execute notebooks/01_eda.py
