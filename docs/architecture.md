# CineMatch Architecture

CineMatch is organized as a two-stage offline recommendation pipeline.

## Data Flow

```text
MovieLens CSV files
        |
        v
data_loader.py
        |
        v
preprocessing.py
        |
        v
split.py
        |
        v
candidate.py
        |
        v
features.py
        |
        v
ranking.py
        |
        v
evaluation.py
        |
        v
artifacts/
        |
        v
batch_recommend.py
```

## Module Boundaries

### `data_loader.py`

Owns CSV reading and schema validation. It does not clean or transform data.

### `preprocessing.py`

Owns type normalization, duplicate handling, invalid rating filtering, genre parsing, and filtering ratings to known movies.

### `split.py`

Owns leakage-safe time-based train/test splitting. The default strategy holds out each user's latest interaction.

### `candidate.py`

Owns retrieval. It generates a manageable set of user-item candidates from:

- global popularity
- item-item collaborative filtering
- matrix factorization with truncated SVD
- weighted hybrid merging

Candidate generation excludes items already seen in the training history.

### `features.py`

Owns ranking feature construction. It fits user/item/genre statistics on training data only, then transforms candidate pairs.

### `ranking.py`

Owns supervised ranking. It builds positive and negative training examples, trains a configurable sklearn ranker, and scores candidates. The default ranker is histogram-based gradient boosting.

### `evaluation.py`

Owns top-K recommender metrics using held-out future interactions.

### `pipeline.py`

Owns orchestration and artifact writing. It is intentionally thin and delegates ML logic to the modules above.

### `artifacts.py`

Owns serialization and loading of trained components.

### `inference.py`

Owns batch recommendation generation from saved artifacts.

## Leakage Controls

- Test interactions are later than training interactions for the same user.
- Aggregate statistics are fit only on training data.
- Candidate generation is fit only on training data.
- Seen training items are excluded from recommendation candidates.
- Evaluation uses held-out positive test interactions.

## Extension Points

The safest future improvements are:

- Replace `LogisticRegressionRanker` with another sklearn model.
- Add additional feature columns in `FeatureBuilder`.
- Add a new candidate generator implementing the same `fit` and `generate` interface.
- Add API serving after batch inference is stable.
