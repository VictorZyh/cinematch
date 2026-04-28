"""Tests for supervised ranking utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from cinematch.constants import ITEM_ID, LABEL, SCORE
from cinematch.features import FeatureBuilder
from cinematch.ranking import (
    LogisticRegressionRanker,
    build_positive_pairs,
    build_training_candidates,
    sample_negative_pairs,
    train_ranker,
)


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 11, 10, 12],
            "rating": [5.0, 2.0, 4.5, 1.0],
            "timestamp": [1, 2, 1, 2],
        }
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [10, 11, 12, 10, 11, 12],
            "score": [0.9, 0.4, 0.3, 0.8, 0.5, 0.2],
        }
    )


def _movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [10, 11, 12],
            "title": ["A", "B", "C"],
            "genres": ["Action", "Drama", "Comedy"],
            "genre_list": [["Action"], ["Drama"], ["Comedy"]],
        }
    )


def test_build_positive_pairs_uses_rating_threshold() -> None:
    positives = build_positive_pairs(_interactions(), positive_rating_threshold=4.0)

    assert set(positives[ITEM_ID]) == {10}
    assert positives[LABEL].tolist() == [1, 1]


def test_sample_negative_pairs_never_samples_positive_items() -> None:
    positives = build_positive_pairs(_interactions(), positive_rating_threshold=4.0)
    negatives = sample_negative_pairs(_candidates(), positives, 2, random_seed=42)

    positive_keys = set(zip(positives["userId"], positives["movieId"]))
    negative_keys = set(zip(negatives["userId"], negatives["movieId"]))

    assert positive_keys.isdisjoint(negative_keys)
    assert set(negatives[LABEL]) == {0}


def test_build_training_candidates_contains_positive_and_negative_labels() -> None:
    training = build_training_candidates(
        candidate_rows=_candidates(),
        interactions=_interactions(),
        positive_rating_threshold=4.0,
        negatives_per_positive=2,
        random_seed=42,
    )

    assert set(training[LABEL]) == {0, 1}


def test_logistic_regression_ranker_fit_and_predict_scores() -> None:
    feature_builder = FeatureBuilder().fit(_interactions(), _movies())
    training = build_training_candidates(_candidates(), _interactions(), 4.0, 2, 42)
    feature_frame = feature_builder.transform(training)
    feature_frame[LABEL] = training[LABEL].values

    ranker = LogisticRegressionRanker(max_iter=200).fit(feature_frame)
    scores = ranker.predict_scores(feature_builder.transform(_candidates()))

    assert not scores.empty
    assert scores[SCORE].between(0.0, 1.0).all()


def test_logistic_regression_ranker_requires_fit_before_predict() -> None:
    with pytest.raises(RuntimeError, match="must be fit"):
        LogisticRegressionRanker().predict_scores(pd.DataFrame())


def test_train_ranker_returns_feature_builder_ranker_and_training_frame() -> None:
    feature_builder, ranker, feature_frame = train_ranker(
        candidate_rows=_candidates(),
        train_interactions=_interactions(),
        movies=_movies(),
        positive_rating_threshold=4.0,
        negatives_per_positive=2,
        random_seed=42,
        max_iter=200,
    )

    assert isinstance(feature_builder, FeatureBuilder)
    assert ranker.model_ is not None
    assert set(feature_frame[LABEL]) == {0, 1}
