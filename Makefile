# ---------------------------------------------------------------------------
# Common tasks. Run `make help` for the list.
# ---------------------------------------------------------------------------
PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help install setup ingest ingest-fast refresh train train-fast eda \
        dashboard api test test-fast lint clean reset docker

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

setup: install  ## Install, create the schema and copy .env
	@test -f .env || cp .env.example .env
	$(PYTHON) scripts/init_db.py

ingest:  ## Collect all data (first run takes ~20 minutes)
	$(PYTHON) scripts/ingest.py

ingest-fast:  ## Collect without ball-by-ball data (~3 minutes)
	$(PYTHON) scripts/ingest.py --no-deliveries

refresh:  ## Incrementally refresh the current season
	$(PYTHON) scripts/ingest.py --seasons $$(date +%Y)

train:  ## Train and compare every model
	$(PYTHON) scripts/train_models.py

train-fast:  ## Train without the chase model (much quicker)
	$(PYTHON) scripts/train_models.py --skip-chase

eda:  ## Generate the EDA report into reports/
	$(PYTHON) scripts/run_eda.py

dashboard:  ## Launch the Streamlit dashboard
	$(PYTHON) -m streamlit run streamlit_app.py

api:  ## Launch the FastAPI service
	$(PYTHON) scripts/run_api.py

test:  ## Run the full test suite
	$(PYTHON) -m pytest

test-fast:  ## Run tests, skipping the one that needs the network
	$(PYTHON) -m pytest -m "not network"

lint:  ## Lint with ruff (if installed)
	@ruff check src tests scripts || echo "ruff not installed: pip install ruff"

clean:  ## Remove caches and generated reports
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .coverage htmlcov reports/figures reports/*.html
	@echo "Cleaned."

reset:  ## DESTRUCTIVE - drop the database and delete trained models
	$(PYTHON) scripts/init_db.py --drop
	@rm -rf models/artifacts/*
	@echo "Database and models reset."

docker:  ## Build and start the full stack
	docker compose up --build
