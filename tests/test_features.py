"""Tests for leakage-safe feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from cinematch.constants import ITEM_ID, SCORE, USER_ID
from cinematch.features import FEATURE_COLUMNS, FeatureBuilder


def _train_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [10, 11, 10],
            "rating": [5.0, 3.0, 4.0],
            "timestamp": [1, 2, 1],
        }
    )


def _movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [10, 11, 12],
            "title": ["A", "B", "C"],
            "genres": ["Action", "Drama", "Action|Comedy"],
            "genre_list": [["Action"], ["Drama"], ["Action", "Comedy"]],
        }
    )


def test_feature_builder_adds_expected_feature_columns() -> None:
    candidates = pd.DataFrame({"userId": [1], "movieId": [12], "score": [0.8]})

    features = FeatureBuilder().fit(_train_interactions(), _movies()).transform(candidates)

    assert features[USER_ID].tolist() == [1]
    assert features[ITEM_ID].tolist() == [12]
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert features["candidate_score"].tolist() == [0.8]
    assert features["user_item_genre_overlap"].tolist() == [0.5]
    assert features["user_item_genre_jaccard"].tolist() == [1 / 3]
    assert features["user_item_genre_affinity"].tolist() == [5.0]


def test_feature_builder_uses_fallbacks_for_unknown_user_or_item() -> None:
    candidates = pd.DataFrame({"userId": [99], "movieId": [999], "score": [0.1]})

    features = FeatureBuilder().fit(_train_interactions(), _movies()).transform(candidates)

    assert features["user_rating_count"].tolist() == [0.0]
    assert features["item_popularity_score"].tolist() == [0.0]
    assert features[SCORE.replace("score", "candidate_score")].tolist() == [0.1]
    assert features["user_item_genre_overlap"].tolist() == [0.0]


def test_feature_builder_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="must be fit"):
        FeatureBuilder().transform(pd.DataFrame({"userId": [1], "movieId": [10], "score": [1.0]}))
