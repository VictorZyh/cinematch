.PHONY: install test download run clean

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install ".[dev]"

test:
	PYTHONPATH=src:. $(VENV_PYTHON) -m pytest -q

download:
	PYTHONPATH=src $(VENV_PYTHON) scripts/download_movielens.py --dataset latest-small

run:
	PYTHONPATH=src $(VENV_PYTHON) scripts/run_pipeline.py --config configs/default.json

clean:
	rm -rf artifacts data/processed .pytest_cache .coverage htmlcov build dist *.egg-info
