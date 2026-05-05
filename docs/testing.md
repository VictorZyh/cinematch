# Testing Documentation

CineMatch includes unit tests for all core modules. The goal is to verify correctness, leakage control, reproducibility, and end-to-end behavior.

## Running Tests

```bash
make test
```

Equivalent command:

```bash
PYTHONPATH=src:. LOKY_MAX_CPU_COUNT=1 .venv/bin/python -m pytest -q
```

Current validation:

```text
53 passed
Total coverage: 94.51%
```

The project enforces:

```text
--cov-fail-under=80
```

in `pyproject.toml`.

## Test Files

### `tests/test_config.py`

Validates loading of `configs/default.json` into typed configuration objects.

### `tests/test_data_loader.py`

Covers:

- required schema validation
- missing column errors
- missing file errors
- successful MovieLens CSV loading

### `tests/test_preprocessing.py`

Covers:

- genre parsing
- rating type conversion
- invalid rating filtering
- movie de-duplication
- filtering ratings to known movie ids

### `tests/test_split.py`

Covers:

- timestamp-based train/test split
- filtering users with insufficient history
- stable behavior with unsorted input
- explicit temporal leakage checks

### `tests/test_candidate.py`

Covers:

- seen-item filtering
- popularity retrieval
- item-item collaborative filtering
- SVD matrix-factorization retrieval
- BPR pairwise matrix-factorization retrieval
- hybrid candidate merging
- pre-fit error handling

### `tests/test_features.py`

Covers:

- user/item aggregate features
- genre overlap
- genre Jaccard
- genre affinity
- fallback values for unknown users/items
- pre-fit error handling

### `tests/test_ranking.py`

Covers:

- positive label construction
- negative sampling
- logistic regression ranker
- histogram gradient boosting ranker option
- unsupported model type errors

### `tests/test_evaluation.py`

Covers:

- Precision@K
- Recall@K
- nDCG@K
- HitRate@K through aggregate evaluation
- invalid K handling

### `tests/test_pipeline.py`

Covers a tiny end-to-end run using synthetic MovieLens-shaped data. It verifies that the pipeline creates:

- metrics
- metadata
- recommendations

### `tests/test_artifacts.py`

Covers pickle persistence helpers.

### `tests/test_inference.py`

Covers:

- loading user ids from text and CSV
- CSV schema validation
- batch recommendation from saved artifacts

## Why This Test Strategy Matters

The tests focus on the highest-risk ML engineering failure modes:

- data schema mismatch
- temporal leakage
- recommending already-seen items
- incorrect label construction
- invalid evaluation metrics
- broken end-to-end orchestration
- inability to load saved artifacts for inference
