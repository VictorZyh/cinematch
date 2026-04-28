# Data Documentation

## Dataset

CineMatch uses MovieLens datasets from GroupLens.

The default development dataset is `ml-latest-small`, which contains:

- `ratings.csv`
- `movies.csv`
- `tags.csv`
- `links.csv`

The current pipeline uses:

- `ratings.csv`
- `movies.csv`

## Required Schema

### `ratings.csv`

```text
userId,movieId,rating,timestamp
```

### `movies.csv`

```text
movieId,title,genres
```

## Download

```bash
python scripts/download_movielens.py --dataset latest-small
```

This extracts:

```text
data/raw/ml-latest-small/
├── ratings.csv
├── movies.csv
├── tags.csv
└── links.csv
```

## Why Raw Data Is Not Committed

Raw datasets are excluded from Git through `.gitignore`.
This keeps the repository lightweight and makes data acquisition explicit and reproducible.

## Label Definition

The pipeline treats a rating as positive when:

```text
rating >= positive_rating_threshold
```

The default threshold is configured in `configs/default.json`.

## Train/Test Split

The split is timestamp-based per user.
For each eligible user:

- latest interactions are held out for test
- earlier interactions are used for training

This simulates recommending future movies from past behavior.
