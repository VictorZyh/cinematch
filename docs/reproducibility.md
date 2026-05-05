# Reproducibility Guide

This guide describes how to reproduce the CineMatch package, tests, data download, training pipeline, evaluation metrics, and batch inference artifacts from a clean clone.

## 1. Clone the Repository

```bash
git clone https://github.com/VictorZyh/cinematch.git
cd cinematch
```

## 2. Create the Environment and Install the Package

The project is configured as a Python package through `pyproject.toml`.

```bash
make install
```

Equivalent manual commands:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install ".[dev]"
```

## 3. Download the Dataset

The default dataset is MovieLens Latest Small.

```bash
make download
```

Equivalent command:

```bash
PYTHONPATH=src .venv/bin/python scripts/download_movielens.py --dataset latest-small
```

Expected files:

```text
data/raw/ml-latest-small/
├── ratings.csv
├── movies.csv
├── tags.csv
└── links.csv
```

The pipeline currently uses `ratings.csv` and `movies.csv`.

## 4. Run Unit Tests

```bash
make test
```

Current local validation:

```text
53 passed
Total coverage: 94.51%
```

The pytest configuration enforces coverage above 80%, matching the course rubric requirement.

## 5. Run the End-to-End Pipeline

```bash
make run
```

Equivalent command:

```bash
PYTHONPATH=src LOKY_MAX_CPU_COUNT=1 .venv/bin/python scripts/run_pipeline.py --config configs/default.json
```

The pipeline runs:

```text
load data
preprocess
time-based train/test split
candidate generation
ranking feature construction
ranker training
offline evaluation
artifact persistence
```

Expected output:

```text
CineMatch pipeline status: completed
Metrics saved to: artifacts/metrics.json
```

## 6. Inspect Artifacts

The run writes:

```text
artifacts/
├── candidate_generator.pkl
├── feature_builder.pkl
├── metrics.json
├── ranker.pkl
├── recommendations.csv
├── run_metadata.json
└── train_interactions.pkl
```

`artifacts/` is intentionally ignored by Git because it is generated output.

## 7. Run Batch Inference

Create a simple user file:

```bash
printf "1\n2\n3\n" > artifacts/users.txt
```

Run:

```bash
make recommend
```

Equivalent command:

```bash
PYTHONPATH=src LOKY_MAX_CPU_COUNT=1 .venv/bin/python scripts/batch_recommend.py \
  --artifact-dir artifacts \
  --user-file artifacts/users.txt \
  --output-path artifacts/batch_recommendations.csv \
  --top-k 10
```

Expected output:

```text
artifacts/batch_recommendations.csv
```

## Determinism Notes

- The default random seed is configured in `configs/default.json`.
- Candidate generation and model training use deterministic seeds where applicable.
- BPR uses seeded NumPy sampling.
- Results can vary slightly across platforms due to numerical libraries, but the pipeline and tests are reproducible.
