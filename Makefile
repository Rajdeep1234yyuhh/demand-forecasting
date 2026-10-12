.DEFAULT_GOAL := help
MLFLOW_VERSION ?= 3.17.0
KAGGLE_DIR := dist/kaggle
.PHONY: help install lint format typecheck test check clean download-data ingest eda backtest results \
        kaggle-notebook kaggle-bundle mlflow

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

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

backtest: ## Backtest every model with rolling-origin folds (writes reports/backtest/...)
	uv run df-backtest

results: ## Execute the results notebook (writes notebooks/02_results.ipynb; needs a backtest)
	uv run --group notebook jupytext --to ipynb --execute notebooks/02_results.py

kaggle-notebook: ## Rebuild notebooks/kaggle_run.ipynb, the notebook to import into Kaggle
	uv run --group notebook jupytext --update --to ipynb notebooks/kaggle_run.py

kaggle-bundle: ## Zip the last commit for Kaggle (upload it as a private dataset)
	@git diff --quiet HEAD -- || { echo "Commit your changes first: the bundle is built from HEAD."; exit 1; }
	@ignored="$$(git ls-files --others --ignored --exclude-standard -- 'src/*.py' 'tests/*.py' 'notebooks/*.py' 'configs/*')"; \
		test -z "$$ignored" || { echo "Ignored by git, so missing from the bundle:"; echo "$$ignored"; exit 1; }
	mkdir -p $(KAGGLE_DIR)
	git log -1 --format='{"commit": "%H", "committed_at": "%cI"}' > $(KAGGLE_DIR)/BUILD_INFO.json
	git archive --format=zip --prefix=demand-forecasting/ --add-file=$(KAGGLE_DIR)/BUILD_INFO.json \
		--output=$(KAGGLE_DIR)/demand-forecasting-code.zip HEAD
	@echo "Upload $(KAGGLE_DIR)/demand-forecasting-code.zip to Kaggle as a private dataset."

mlflow: ## Run a local MLflow server on http://127.0.0.1:5000 (data in mlruns/)
	mkdir -p mlruns
	MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false \
	uvx --from "mlflow==$(MLFLOW_VERSION)" mlflow server \
		--backend-store-uri sqlite:///mlruns/mlflow.db \
		--artifacts-destination mlruns/artifacts \
		--host 127.0.0.1 --port 5000 --workers 1
