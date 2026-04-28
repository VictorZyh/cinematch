# CineMatch

CineMatch is a production-style, end-to-end movie recommendation system built on MovieLens.
The project focuses on clean machine-learning engineering: modular code, leakage-safe evaluation, reproducible training, unit tests, and a runnable pipeline.

The current model is intentionally a reliable baseline rather than an overfit research model. The architecture makes it easy to replace or improve candidate generation, ranking features, or the ranker later.

## What It Does

CineMatch trains and evaluates a two-stage recommender:

1. **Candidate generation**
   - Popularity-based retrieval
   - Item-item collaborative filtering with cosine nearest neighbors
   - Weighted hybrid candidate merging

2. **Ranking**
   - Leakage-safe user, item, genre, and candidate-source features
   - Supervised baseline ranker using `sklearn.linear_model.LogisticRegression`

3. **Evaluation**
   - Timestamp-based train/test split
   - Precision@K
   - Recall@K
   - nDCG@K
   - HitRate@K
   - Catalog coverage

## Why This Project Is Structured This Way

Industrial recommenders usually do not score every item for every user in one step. They first retrieve a smaller candidate set, then rank those candidates with richer features. CineMatch follows that pattern while keeping the implementation small enough to understand, test, and extend.

The split is time-based per user: each user's latest interactions are held out as future test data, while features and models are fit only on earlier interactions. This avoids the most common recommender-system data leakage mistake.

## Repository Structure

```text
cinematch/
├── configs/
│   └── default.json
├── docs/
│   ├── architecture.md
│   └── data.md
├── scripts/
│   ├── download_movielens.py
│   └── run_pipeline.py
├── src/
│   └── cinematch/
│       ├── candidate.py
│       ├── config.py
│       ├── constants.py
│       ├── data_loader.py
│       ├── evaluation.py
│       ├── features.py
│       ├── pipeline.py
│       ├── preprocessing.py
│       ├── ranking.py
│       ├── split.py
│       └── utils.py
├── tests/
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## Dataset

The default dataset is MovieLens Latest Small from GroupLens.

Expected files after download:

```text
data/raw/ml-latest-small/
├── ratings.csv
└── movies.csv
```

Download it with:

```bash
python scripts/download_movielens.py --dataset latest-small
```

The larger MovieLens latest dataset is also supported:

```bash
python scripts/download_movielens.py --dataset latest
```

Raw data is intentionally excluded from Git. The dataset is downloaded reproducibly from the official GroupLens file server.

## Quickstart

Create the environment and install dependencies:

```bash
make install
```

Download the default dataset:

```bash
make download
```

Run the full pipeline:

```bash
make run
```

Run tests:

```bash
make test
```

## Pipeline Output

The pipeline writes artifacts to:

```text
artifacts/
├── metrics.json
├── recommendations.csv
└── run_metadata.json
```

Example command:

```bash
PYTHONPATH=src python scripts/run_pipeline.py --config configs/default.json
```

## Current Test Status

Local validation:

```text
42 passed
Total coverage: 93.83%
```

The GitHub Actions workflow also runs the test suite on push and pull request.

## Design Principles

- No notebooks in the production path
- Only `pandas`, `numpy`, and `scikit-learn` for ML/data logic
- Modular components with clear ownership
- Explicit type hints and docstrings
- Leakage-safe feature computation
- Testable pure functions where possible
- Reproducible command-line pipeline

## Next Improvements

- Add model serialization and loading
- Add batch inference for selected users
- Add feature importance and diagnostics
- Add stronger ranking models and better negative sampling
- Add experiment tracking with metrics history
- Add API serving once the offline pipeline stabilizes
