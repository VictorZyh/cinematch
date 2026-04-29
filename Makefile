.PHONY: install test download run recommend clean

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install ".[dev]"

test:
	PYTHONPATH=src:. LOKY_MAX_CPU_COUNT=1 $(VENV_PYTHON) -m pytest -q

download:
	PYTHONPATH=src $(VENV_PYTHON) scripts/download_movielens.py --dataset latest-small

run:
	PYTHONPATH=src LOKY_MAX_CPU_COUNT=1 $(VENV_PYTHON) scripts/run_pipeline.py --config configs/default.json

recommend:
	PYTHONPATH=src LOKY_MAX_CPU_COUNT=1 $(VENV_PYTHON) scripts/batch_recommend.py --artifact-dir artifacts --user-file artifacts/users.txt --output-path artifacts/batch_recommendations.csv --top-k 10

clean:
	rm -rf artifacts data/processed .pytest_cache .coverage htmlcov build dist *.egg-info
