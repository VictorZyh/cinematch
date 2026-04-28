# Experiment Report

## Objective

Build a reproducible baseline recommender that demonstrates an end-to-end MLE workflow:

```text
data loading -> preprocessing -> leakage-safe split -> candidate generation -> ranking -> evaluation -> artifacts -> batch inference
```

## Configuration

Default config:

```text
configs/default.json
```

Important settings:

- dataset path: `data/raw/ml-latest-small`
- positive rating threshold: `4.0`
- test interactions per user: `1`
- candidate count per user: `100`
- item similarity neighbors: `50`
- matrix factorization factors: `32`
- ranker: logistic regression
- negative samples per positive: `4`
- K values: `5, 10, 20`

## Data Split

The experiment uses a per-user timestamp split:

- each eligible user's latest interaction is held out for test
- earlier interactions are used for training

This simulates recommending future movies from historical user behavior and avoids using future interactions in feature construction.

Observed split from a local run:

```text
ratings rows: 100,836
movies rows: 9,742
train rows: 100,226
test rows: 610
train candidate rows: 61,000
test candidate rows: 61,000
training examples: 109,217
```

## Results

Sample metrics are available in:

```text
docs/sample_metrics.json
```

Latest default summary:

```text
Recall@5:  0.0523
Recall@10: 0.0909
Recall@20: 0.1433
nDCG@5:    0.0291
nDCG@10:   0.0412
nDCG@20:   0.0544
HitRate@5:  0.0523
HitRate@10: 0.0909
HitRate@20: 0.1433
```

These numbers are reasonable for a first baseline with simple features and limited candidate generation. The most important outcome at this stage is that the system evaluates the right prediction problem without temporal leakage.

## Interpretation

The recall and hit-rate values show that the pipeline can recover some held-out positive interactions, but there is significant room to improve retrieval and ranking quality.

The current catalog coverage is limited because the hybrid candidate generator retrieves only the top candidates from popularity and item similarity. Increasing recall will likely require:

- broader candidate generation
- better user preference features
- stronger negative sampling
- sequence or recency signals

## Model Comparison Notes

During the model upgrade, several sklearn-only configurations were compared on the same MovieLens Latest Small split:

```text
configuration       Recall@10  Recall@20  nDCG@10  Coverage
svd_light_logreg      0.0909     0.1433    0.0412       921
svd_light_hgb         0.0248     0.0854    0.0118       921
svd_tiny_logreg       0.0882     0.1350    0.0413       827
svd_none_hgb          0.0193     0.0882    0.0072       807
```

The selected default is `svd_light_logreg`: a hybrid retriever with a small SVD component and a logistic regression ranker. Histogram gradient boosting remains available through config, but it is not the default because it underperformed on this feature set and sampling strategy.

## Reproduction

```bash
make install
make download
make run
make test
```

Batch inference after training:

```bash
printf "1\n2\n3\n" > artifacts/users.txt
make recommend
```

## Next Experiment Ideas

1. Add a popularity-only baseline comparison.
2. Add feature ablation to measure the value of genre overlap and item statistics.
3. Tune candidate generator weights.
4. Add a tree-based sklearn ranker and compare Recall@K/nDCG@K.
5. Improve negative sampling with hard negatives.
