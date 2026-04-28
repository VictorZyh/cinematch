"""Top-K recommendation evaluation metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Set

import numpy as np
import pandas as pd

from cinematch.constants import ITEM_ID, RATING, SCORE, USER_ID


def build_relevance_by_user(
    interactions: pd.DataFrame,
    positive_rating_threshold: float,
) -> Dict[int, Set[int]]:
    """Build a mapping of users to relevant held-out items."""

    positives = interactions.loc[interactions[RATING] >= positive_rating_threshold]
    grouped = positives.groupby(USER_ID)[ITEM_ID].apply(set)
    return {int(user_id): {int(item_id) for item_id in item_ids} for user_id, item_ids in grouped.items()}


def top_k_recommendations(scored_candidates: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return the top-k scored candidates for each user."""

    if k <= 0:
        raise ValueError("k must be positive.")
    ranked = scored_candidates.sort_values(
        [USER_ID, SCORE, ITEM_ID],
        ascending=[True, False, True],
    )
    return ranked.groupby(USER_ID).head(k).reset_index(drop=True)


def precision_at_k(recommended_items: Iterable[int], relevant_items: Set[int], k: int) -> float:
    """Compute precision@k for one user."""

    if k <= 0:
        raise ValueError("k must be positive.")
    recommendations = list(recommended_items)[:k]
    if not recommendations:
        return 0.0
    hits = len(set(recommendations).intersection(relevant_items))
    return float(hits / k)


def recall_at_k(recommended_items: Iterable[int], relevant_items: Set[int], k: int) -> float:
    """Compute recall@k for one user."""

    if k <= 0:
        raise ValueError("k must be positive.")
    if not relevant_items:
        return 0.0
    recommendations = set(list(recommended_items)[:k])
    hits = len(recommendations.intersection(relevant_items))
    return float(hits / len(relevant_items))


def ndcg_at_k(recommended_items: Iterable[int], relevant_items: Set[int], k: int) -> float:
    """Compute binary nDCG@k for one user."""

    if k <= 0:
        raise ValueError("k must be positive.")
    gains = [
        1.0 if item_id in relevant_items else 0.0
        for item_id in list(recommended_items)[:k]
    ]
    if not gains:
        return 0.0
    discounts = [1.0 / np.log2(index + 2) for index in range(len(gains))]
    dcg = float(np.sum(np.array(gains) * np.array(discounts)))
    ideal_hits = min(len(relevant_items), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = float(np.sum([1.0 / np.log2(index + 2) for index in range(ideal_hits)]))
    return dcg / ideal_dcg


def evaluate_recommendations(
    scored_candidates: pd.DataFrame,
    test_interactions: pd.DataFrame,
    positive_rating_threshold: float,
    k_values: Iterable[int],
) -> dict[str, float]:
    """Evaluate scored candidates with standard top-K recommendation metrics."""

    relevance_by_user = build_relevance_by_user(test_interactions, positive_rating_threshold)
    all_candidate_items = set(scored_candidates[ITEM_ID].unique())
    metrics: dict[str, float] = {
        "evaluated_users": float(len(relevance_by_user)),
        "catalog_coverage": 0.0,
    }
    if scored_candidates.empty:
        return metrics

    metrics["catalog_coverage"] = float(len(all_candidate_items))

    for k in k_values:
        top_k = top_k_recommendations(scored_candidates, k)
        precision_values = []
        recall_values = []
        ndcg_values = []
        hit_values = []

        for user_id, relevant_items in relevance_by_user.items():
            user_recommendations = top_k.loc[top_k[USER_ID] == user_id, ITEM_ID].tolist()
            precision_values.append(precision_at_k(user_recommendations, relevant_items, k))
            recall = recall_at_k(user_recommendations, relevant_items, k)
            recall_values.append(recall)
            ndcg_values.append(ndcg_at_k(user_recommendations, relevant_items, k))
            hit_values.append(1.0 if recall > 0.0 else 0.0)

        metrics[f"precision_at_{k}"] = float(np.mean(precision_values)) if precision_values else 0.0
        metrics[f"recall_at_{k}"] = float(np.mean(recall_values)) if recall_values else 0.0
        metrics[f"ndcg_at_{k}"] = float(np.mean(ndcg_values)) if ndcg_values else 0.0
        metrics[f"hit_rate_at_{k}"] = float(np.mean(hit_values)) if hit_values else 0.0

    return metrics
