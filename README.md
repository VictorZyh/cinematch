# CineMatch

CineMatch is a production-style MovieLens recommendation system implemented as a clean Python package.
It is designed for reproducible end-to-end training, candidate generation, ranking, and offline evaluation using only `pandas`, `numpy`, and `scikit-learn`.

## Project Goals

- Build a modular recommendation system with clear production boundaries.
- Avoid data leakage through timestamp-aware train/test splitting.
- Support testable candidate generation, ranking, evaluation, and pipeline components.
- Provide a reproducible command-line entry point for end-to-end runs.
- Maintain high unit-test coverage for core ML logic.

## Dataset

The project targets the MovieLens dataset family from GroupLens.
The default development path is intended for MovieLens Latest Small for fast local reproduction, while the same code structure can be pointed at larger MovieLens releases.

Expected raw files:

- `ratings.csv`
- `movies.csv`

## Current Status

Step 1 completed:

- Python package skeleton
- Project metadata
- Default JSON config
- Constants and typed config objects
- Utility helpers

Implementation modules will be added incrementally.

## Planned End-to-End Command

```bash
python scripts/run_pipeline.py --config configs/default.json
```

The command will load data, create a leakage-safe split, generate candidates, train a ranking model, evaluate top-K metrics, and save reproducible artifacts.
