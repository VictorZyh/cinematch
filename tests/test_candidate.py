"""Tests for candidate generation."""

from __future__ import annotations

import pandas as pd
import pytest

from cinematch.candidate import (
    HybridCandidateGenerator,
    ItemSimilarityCandidateGenerator,
    PopularityCandidateGenerator,
    build_seen_items,
    create_default_candidate_generator,
)
from cinematch.constants import ITEM_ID, SCORE, USER_ID


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3, 4],
            "movieId": [10, 11, 10, 12, 11, 12, 13],
            "rating": [5.0, 4.0, 4.0, 5.0, 5.0, 4.0, 3.0],
            "timestamp": [1, 2, 1, 2, 1, 2, 1],
        }
    )


def test_build_seen_items_returns_user_item_sets() -> None:
    seen = build_seen_items(_interactions())

    assert seen[1] == {10, 11}
    assert seen[2] == {10, 12}


def test_popularity_generator_excludes_seen_items() -> None:
    interactions = _interactions()
    generator = PopularityCandidateGenerator().fit(interactions)
    candidates = generator.generate(
        user_ids=[1],
        seen_items_by_user=build_seen_items(interactions),
        num_candidates=2,
    )

    assert set(candidates[ITEM_ID]) == {12, 13}
    assert candidates[USER_ID].tolist() == [1, 1]


def test_popularity_generator_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="must be fit"):
        PopularityCandidateGenerator().generate([1], {}, 10)


def test_item_similarity_generator_scores_unseen_similar_items() -> None:
    interactions = _interactions()
    generator = ItemSimilarityCandidateGenerator(num_similar_items=2).fit(interactions)
    candidates = generator.generate(
        user_ids=[1],
        seen_items_by_user=build_seen_items(interactions),
        num_candidates=2,
    )

    assert 12 in set(candidates[ITEM_ID])
    assert all(candidates[SCORE] > 0.0)
    assert not set(candidates[ITEM_ID]).intersection({10, 11})


def test_item_similarity_generator_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="must be fit"):
        ItemSimilarityCandidateGenerator().generate([1], {}, 10)


def test_hybrid_candidate_generator_combines_and_limits_candidates() -> None:
    interactions = _interactions()
    generator = create_default_candidate_generator(
        num_similar_items=2,
        popularity_weight=0.4,
        similarity_weight=0.6,
    ).fit(interactions)

    candidates = generator.generate(
        user_ids=[1, 2],
        seen_items_by_user=build_seen_items(interactions),
        num_candidates=2,
    )

    assert set(candidates[USER_ID]) == {1, 2}
    assert candidates.groupby(USER_ID).size().max() <= 2
    assert all(candidates[SCORE] >= 0.0)


def test_hybrid_candidate_generator_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="same length"):
        HybridCandidateGenerator(generators=[PopularityCandidateGenerator()], weights=[0.5, 0.5])

    with pytest.raises(ValueError, match="non-negative"):
        HybridCandidateGenerator(generators=[PopularityCandidateGenerator()], weights=[-1.0])
