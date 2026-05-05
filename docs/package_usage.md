# Package Usage

CineMatch is organized as an importable Python package under `src/cinematch`.

## Importing the Package

After installation:

```bash
make install
```

You can import the package:

```python
from cinematch.config import load_config
from cinematch.pipeline import run_pipeline

config = load_config("configs/default.json")
results = run_pipeline(config)
print(results["status"])
```

## Main Modules

### `config.py`

Loads typed configuration objects from JSON.

```python
from cinematch.config import load_config

config = load_config("configs/default.json")
```

### `data_loader.py`

Reads MovieLens CSV files and validates required columns.

```python
from cinematch.data_loader import load_movielens_data

raw_data = load_movielens_data(config.data)
```

### `preprocessing.py`

Cleans ratings, parses movie genres, and filters ratings to known movies.

```python
from cinematch.preprocessing import preprocess_movielens

ratings, movies = preprocess_movielens(
    raw_data.ratings,
    raw_data.movies,
    min_rating=config.data.min_rating,
)
```

### `split.py`

Creates leakage-safe time-based train/test splits.

```python
from cinematch.split import time_based_train_test_split

split = time_based_train_test_split(
    ratings,
    test_interactions_per_user=config.split.test_interactions_per_user,
    min_train_interactions_per_user=config.split.min_train_interactions_per_user,
)
```

### `candidate.py`

Generates recommendation candidates using multiple retrieval models:

- popularity
- item-item collaborative filtering
- SVD matrix factorization
- BPR pairwise matrix factorization

```python
from cinematch.candidate import build_seen_items, create_default_candidate_generator

seen_items = build_seen_items(split.train)
generator = create_default_candidate_generator(
    num_similar_items=config.candidate.num_similar_items,
    num_factors=config.candidate.num_factors,
    bpr_factors=config.candidate.bpr_factors,
    bpr_epochs=config.candidate.bpr_epochs,
    bpr_samples_per_epoch=config.candidate.bpr_samples_per_epoch,
    popularity_weight=config.candidate.popularity_weight,
    similarity_weight=config.candidate.similarity_weight,
    matrix_factorization_weight=config.candidate.matrix_factorization_weight,
    bpr_weight=config.candidate.bpr_weight,
    positive_threshold=config.data.positive_rating_threshold,
    random_seed=config.random_seed,
).fit(split.train)
```

### `features.py`

Builds leakage-safe ranking features from training data.

```python
from cinematch.features import FeatureBuilder

feature_builder = FeatureBuilder().fit(split.train, movies)
feature_frame = feature_builder.transform(candidates)
```

### `ranking.py`

Builds labeled training candidates and trains a configurable sklearn ranker.

```python
from cinematch.ranking import train_ranker

feature_builder, ranker, training_frame = train_ranker(
    candidate_rows=train_candidates,
    train_interactions=split.train,
    movies=movies,
    positive_rating_threshold=config.data.positive_rating_threshold,
    negatives_per_positive=config.ranking.negative_samples_per_positive,
    random_seed=config.random_seed,
    max_iter=config.ranking.max_iter,
    model_type=config.ranking.model_type,
)
```

### `evaluation.py`

Computes top-K recommendation metrics.

```python
from cinematch.evaluation import evaluate_recommendations

metrics = evaluate_recommendations(
    scored_candidates=scored_candidates,
    test_interactions=split.test,
    positive_rating_threshold=config.data.positive_rating_threshold,
    k_values=config.evaluation.k_values,
)
```

### `artifacts.py`

Saves and loads trained components.

```python
from cinematch.artifacts import default_artifact_paths, load_pickle

paths = default_artifact_paths("artifacts")
ranker = load_pickle(paths.ranker)
```

### `inference.py`

Generates batch recommendations from saved artifacts.

```python
from cinematch.inference import generate_recommendations

recommendations = generate_recommendations(
    artifact_dir="artifacts",
    user_ids=[1, 2, 3],
    num_candidates=500,
    top_k=10,
)
```

## Scripts

Download data:

```bash
python scripts/download_movielens.py --dataset latest-small
```

Run training/evaluation:

```bash
python scripts/run_pipeline.py --config configs/default.json
```

Run batch inference:

```bash
python scripts/batch_recommend.py --artifact-dir artifacts --user-file artifacts/users.txt
```
