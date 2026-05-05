# Configuration Reference

The default pipeline configuration lives in:

```text
configs/default.json
```

The config is loaded into typed dataclasses in `src/cinematch/config.py`.

## Top-Level Fields

### `project_name`

Human-readable project identifier.

### `random_seed`

Seed used for reproducible candidate generation, BPR sampling, negative sampling, and model training where applicable.

## `data`

### `raw_dir`

Directory containing MovieLens raw files.

Default:

```text
data/raw/ml-latest-small
```

### `processed_dir`

Reserved location for processed data outputs.

### `ratings_filename`

Ratings file name.

Default:

```text
ratings.csv
```

### `movies_filename`

Movie metadata file name.

Default:

```text
movies.csv
```

### `min_rating`

Minimum rating allowed during preprocessing.

### `positive_rating_threshold`

Threshold used to define positive interactions for ranking and evaluation.

Default:

```text
4.0
```

## `split`

### `test_interactions_per_user`

Number of latest interactions per user held out for test.

### `min_train_interactions_per_user`

Minimum number of training interactions a user must have after the holdout.

## `candidate`

### `num_candidates`

Number of candidates kept per user after hybrid retrieval.

### `num_similar_items`

Number of nearest neighbors used by item-item collaborative filtering.

### `num_factors`

Latent dimension for SVD matrix-factorization retrieval.

### `bpr_factors`

Latent dimension for BPR matrix factorization.

### `bpr_epochs`

Number of BPR training epochs.

### `bpr_samples_per_epoch`

Number of sampled pairwise updates per BPR epoch.

### `popularity_weight`

Hybrid weight for popularity retrieval.

### `similarity_weight`

Hybrid weight for item-item collaborative filtering retrieval.

### `matrix_factorization_weight`

Hybrid weight for SVD matrix-factorization retrieval.

### `bpr_weight`

Hybrid weight for BPR retrieval.

## `ranking`

### `negative_samples_per_positive`

Number of negative candidate pairs sampled per positive pair during supervised ranker training.

### `model_type`

Sklearn ranker type.

Supported values:

```text
logistic_regression
hist_gradient_boosting
```

The default is `logistic_regression` because it performed better on the current offline experiment.

### `max_iter`

Maximum iterations for the sklearn ranker.

## `evaluation`

### `k_values`

K values used for top-K metrics.

Default:

```text
[5, 10, 20]
```

## `artifacts`

### `output_dir`

Directory where pipeline outputs are written.

Default:

```text
artifacts
```
