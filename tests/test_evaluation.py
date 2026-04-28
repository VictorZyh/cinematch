"""Tests for top-K recommendation metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from cinematch.evaluation import (
    build_relevance_by_user,
    evaluate_recommendations,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    top_k_recommendations,
)


def test_top_k_recommendations_limits_each_user() -> None:
    scored = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2],
            "movieId": [10, 11, 12, 20, 21],
            "score": [0.2, 0.9, 0.1, 0.5, 0.6],
        }
    )

    top_k = top_k_recommendations(scored, k=1)

    assert top_k["movieId"].tolist() == [11, 21]


def test_pointwise_metrics_match_hand_computed_values() -> None:
    recommended = [10, 11, 12]
    relevant = {10, 12}

    assert precision_at_k(recommended, relevant, 2) == 0.5
    assert recall_at_k(recommended, relevant, 2) == 0.5
    assert ndcg_at_k(recommended, relevant, 3) > 0.0


def test_metrics_reject_invalid_k() -> None:
    with pytest.raises(ValueError, match="positive"):
        precision_at_k([1], {1}, 0)
    with pytest.raises(ValueError, match="positive"):
        recall_at_k([1], {1}, 0)
    with pytest.raises(ValueError, match="positive"):
        ndcg_at_k([1], {1}, 0)
    with pytest.raises(ValueError, match="positive"):
        top_k_recommendations(pd.DataFrame(), 0)


def test_build_relevance_by_user_uses_positive_threshold() -> None:
    interactions = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [10, 11, 20],
            "rating": [4.5, 3.0, 5.0],
        }
    )

    relevance = build_relevance_by_user(interactions, positive_rating_threshold=4.0)

    assert relevance == {1: {10}, 2: {20}}


def test_evaluate_recommendations_returns_expected_keys() -> None:
    scored = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 11, 20, 21],
            "score": [0.9, 0.1, 0.8, 0.2],
        }
    )
    test = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [10, 21],
            "rating": [5.0, 5.0],
        }
    )

    metrics = evaluate_recommendations(scored, test, positive_rating_threshold=4.0, k_values=[1, 2])

    assert metrics["evaluated_users"] == 2.0
    assert metrics["precision_at_1"] == 0.5
    assert metrics["recall_at_2"] == 1.0
    assert metrics["hit_rate_at_2"] == 1.0
