.PHONY: install data features train_logreg train_torch api test lint fmt all

RSCRIPT ?= Rscript

all: data features train_logreg train_torch

install:
pip install -e ".[dev]"

data:
"$(RSCRIPT)" scripts/export_uslapseagent.R --out data/raw/uslapseagent.csv

features:
python -m lapse_poc.data.build_features --raw data/raw/uslapseagent.csv --out data/processed/policy_quarter.parquet --horizon 1

train_logreg:
python -m lapse_poc.models.train_logreg --data data/processed/policy_quarter.parquet --out artifacts/logreg

train_torch:
python -m lapse_poc.models.train_torch --data data/processed/policy_quarter.parquet --out artifacts/torch

api:
uvicorn lapse_poc.api.app:app --reload --port 8000

test:
pytest -q

lint:
ruff check .

fmt:
black .