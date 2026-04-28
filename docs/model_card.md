# CineMatch Model Card

## Model Overview

CineMatch is a two-stage movie recommendation system:

1. Candidate generation retrieves a small set of plausible movies for each user.
2. A supervised ranker scores and orders those candidates.

The current ranking model is a baseline logistic regression model from scikit-learn. It is intentionally simple so the project can emphasize correct ML system design, leakage-safe evaluation, modularity, and reproducibility.

## Intended Use

This model is intended for offline movie recommendation experiments and MLE portfolio demonstration. It can produce top-K recommendations for users that appear in the MovieLens interaction history.

It is not intended for commercial deployment without additional work on fairness, privacy, monitoring, calibration, A/B testing, and broader data governance.

## Dataset

Default dataset:

```text
MovieLens Latest Small
```

Pipeline inputs:

- `ratings.csv`
- `movies.csv`

Default local run statistics:

- ratings rows: 100,836
- movies rows: 9,742
- train rows: 100,226
- test rows: 610

## Label Definition

A positive interaction is defined as:

```text
rating >= 4.0
```

This threshold is configured in `configs/default.json`.

## Features

The current ranker uses:

- candidate retrieval score
- user historical rating count
- user average rating
- item historical rating count
- item average rating
- item popularity score
- user-item genre overlap

All aggregate features are fit on training data only.

## Training Procedure

1. Load MovieLens data.
2. Clean ratings and movie metadata.
3. Hold out each user's latest interaction for test.
4. Fit candidate generators on training data.
5. Generate training candidates.
6. Add positive training pairs from high-rating training interactions.
7. Sample negative user-item candidates.
8. Build ranking features.
9. Train logistic regression.
10. Score held-out test candidates.
11. Evaluate top-K ranking metrics.

## Evaluation

Metrics:

- Precision@K
- Recall@K
- nDCG@K
- HitRate@K
- catalog coverage

Sample metrics from a local MovieLens Latest Small run are stored in:

```text
docs/sample_metrics.json
```

## Known Limitations

- The ranker is a baseline model, not an optimized production ranker.
- Negative sampling is simple and can be improved.
- The model does not use tags, temporal decay, sequence features, or text metadata.
- No demographic features are used because MovieLens Latest Small does not provide them.
- Metrics are offline estimates and do not imply online user satisfaction.
- Cold-start recommendations are limited to popularity and available metadata.

## Recommended Improvements

- Add stronger feature engineering around recency and genre preferences.
- Add hard-negative sampling from high-ranking false positives.
- Compare multiple sklearn rankers.
- Add artifact versioning and model registry metadata.
- Add an API serving layer after the offline and batch paths are stable.
