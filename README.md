# CineMatch

ORIE 5270 Big Data Technology Course Project  
Team members: Yunhe Zhang, Evelyn Feng

CineMatch is a production-style, end-to-end movie recommendation system built on the MovieLens dataset. The project is packaged as a clean Python package and focuses on reproducibility, modular code quality, leakage-safe evaluation, unit testing, and useful runnable scripts.

This README is the grading-facing overview: it explains what the project does, what data it uses, how to install it, how to run it, and how to import the package. The `docs/` folder provides deeper engineering documentation for architecture, data contracts, configuration, experiments, reproducibility, and testing.

## Project Purpose

The goal of CineMatch is to recommend movies to users from historical MovieLens ratings. The system follows a two-stage recommendation architecture:

1. **Candidate generation** retrieves a broad set of movies a user may like.
2. **Ranking** scores and orders those candidates for each user.

The pipeline can download data, preprocess it, create a time-based train/test split, train recommendation components, evaluate top-K ranking quality, save artifacts, and generate batch recommendations from saved artifacts.

## Rubric Coverage

| ORIE 5270 Requirement | How This Project Satisfies It |
|---|---|
| Set up as a Python package | `pyproject.toml` defines the package; source code lives under `src/cinematch/`. |
| README explains project purpose | See Project Purpose and Model Summary in this README. |
| README explains dataset used | See Dataset section and `docs/data.md`. |
| README gives install commands | See Quickstart section. |
| README gives import and script instructions | See Useful Scripts and Package Import Example sections; more examples are in `docs/package_usage.md`. |
| High unit-test coverage | Current validation: 53 tests passing, 94.51% coverage. See `docs/testing.md`. |
| Clean file structure | See Repository Structure section. |
| Detailed documentation | See the `docs/` folder for deeper technical documentation beyond the README. |

## Model Summary

CineMatch uses lightweight mainstream recommendation methods while staying within a `pandas`, `numpy`, and `scikit-learn` stack.

### Candidate Generation

The hybrid candidate generator combines four retrieval sources:

- **Popularity retrieval**: recommends globally popular and highly rated movies.
- **Item-item collaborative filtering**: uses cosine nearest neighbors between movies.
- **SVD matrix factorization**: uses `sklearn.decomposition.TruncatedSVD` to learn latent user/item factors.
- **BPR matrix factorization**: uses a NumPy implementation of pairwise Bayesian Personalized Ranking to learn from positive-vs-negative item preferences.

The retrieval outputs are normalized and merged with configurable weights.

### Ranking

The ranker is a supervised sklearn model. The default ranker is logistic regression because it performed best among the current lightweight experiments. A histogram gradient boosting ranker is also available as a configuration option.

Ranking features include:

- candidate retrieval score
- user rating count and average rating
- item rating count and average rating
- item popularity score
- user-item genre overlap
- user-item genre Jaccard similarity
- user-item genre affinity

All aggregate features are computed from training data only to avoid leakage.

## Dataset

The project uses MovieLens Latest Small from GroupLens.

The pipeline uses:

```text
ratings.csv
movies.csv
```

After downloading, the expected layout is:

```text
data/raw/ml-latest-small/
├── ratings.csv
├── movies.csv
├── tags.csv
└── links.csv
```

Raw data is intentionally excluded from Git. The dataset can be downloaded reproducibly with the provided script.

## Quickstart

Create the environment and install the package:

```bash
make install
```

Equivalent manual installation:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install ".[dev]"
```

Download the default MovieLens dataset:

```bash
make download
```

Run the full training and evaluation pipeline:

```bash
make run
```

The package also exposes a console entry point:

```bash
.venv/bin/cinematch --config configs/default.json
```

Run the test suite:

```bash
make test
```

Generate batch recommendations after running the pipeline:

```bash
printf "1\n2\n3\n" > artifacts/users.txt
make recommend
```

## Useful Scripts

Download MovieLens data:

```bash
.venv/bin/python scripts/download_movielens.py --dataset latest-small
```

Run the full pipeline:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py --config configs/default.json
```

Generate recommendations from saved artifacts:

```bash
PYTHONPATH=src .venv/bin/python scripts/batch_recommend.py \
  --artifact-dir artifacts \
  --user-file artifacts/users.txt \
  --output-path artifacts/batch_recommendations.csv \
  --top-k 10
```

## Package Import Example

After installation, the package can be imported directly:

```python
from cinematch.config import load_config
from cinematch.pipeline import run_pipeline

config = load_config("configs/default.json")
results = run_pipeline(config)
print(results["status"])
```

More module-level examples are available in `docs/package_usage.md`.

## Pipeline Output

The pipeline writes reproducible artifacts to `artifacts/`:

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

These outputs are generated files and are ignored by Git.

## Evaluation Metrics

A held-out movie is treated as relevant when its rating is at least `4.0`.

- **Precision@K**: fraction of top-K recommendations that are relevant.
- **Recall@K**: fraction of relevant held-out movies recovered in the top-K recommendations.
- **nDCG@K**: ranking metric that gives more credit when relevant movies appear near the top.
- **HitRate@K**: fraction of users who receive at least one relevant movie in their top-K recommendations.
- **Catalog Coverage**: number of unique movies recommended across users.

Latest default MovieLens small run:

```text
Recall@10: 0.1157
Recall@20: 0.1625
nDCG@10:   0.0477
Coverage:  2506
```

## Testing and Code Quality

Local validation:

```text
53 passed
Total coverage: 94.51%
```

The test suite covers data loading, preprocessing, leakage-safe splitting, candidate generation, ranking, evaluation metrics, artifact persistence, batch inference, and the end-to-end pipeline. GitHub Actions also runs the tests on push and pull request.

## Repository Structure

```text
cinematch/
├── configs/
│   └── default.json
├── docs/
│   ├── architecture.md
│   ├── configuration.md
│   ├── data.md
│   ├── experiment_report.md
│   ├── model_card.md
│   ├── package_usage.md
│   ├── reproducibility.md
│   ├── sample_metrics.json
│   └── testing.md
├── scripts/
│   ├── batch_recommend.py
│   ├── download_movielens.py
│   └── run_pipeline.py
├── src/
│   └── cinematch/
│       ├── artifacts.py
│       ├── candidate.py
│       ├── config.py
│       ├── constants.py
│       ├── data_loader.py
│       ├── evaluation.py
│       ├── features.py
│       ├── inference.py
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

## Detailed Documentation

- [Architecture](docs/architecture.md): system design and module boundaries.
- [Configuration reference](docs/configuration.md): all fields in `configs/default.json`.
- [Data documentation](docs/data.md): dataset schema, label definition, and split strategy.
- [Model card](docs/model_card.md): model purpose, features, limitations, and future improvements.
- [Experiment report](docs/experiment_report.md): offline experiment setup and sample results.
- [Package usage](docs/package_usage.md): import examples for package modules.
- [Reproducibility guide](docs/reproducibility.md): clean-clone reproduction steps.
- [Sample metrics](docs/sample_metrics.json): metrics from a default run.
- [Testing documentation](docs/testing.md): test coverage and test intent.

