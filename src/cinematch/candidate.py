"""Candidate generation for two-stage movie recommendation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Protocol, Set

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import numpy as np

from cinematch.constants import ITEM_ID, RATING, SCORE, USER_ID


class CandidateGenerator(Protocol):
    """Protocol implemented by candidate generation strategies."""

    def fit(self, interactions: pd.DataFrame) -> "CandidateGenerator":
        """Fit the generator from historical interactions."""

    def generate(
        self,
        user_ids: Iterable[int],
        seen_items_by_user: Dict[int, Set[int]],
        num_candidates: int,
    ) -> pd.DataFrame:
        """Generate candidate item rows for each requested user."""


def build_seen_items(interactions: pd.DataFrame) -> Dict[int, Set[int]]:
    """Build a mapping from user id to items observed in historical data."""

    grouped = interactions.groupby(USER_ID)[ITEM_ID].apply(set)
    return {int(user_id): {int(item_id) for item_id in item_ids} for user_id, item_ids in grouped.items()}


def _candidate_rows_to_frame(rows: List[dict[str, float | int]]) -> pd.DataFrame:
    """Convert candidate rows into a stable candidate DataFrame."""

    if not rows:
        return pd.DataFrame(columns=[USER_ID, ITEM_ID, SCORE])
    frame = pd.DataFrame(rows)
    return frame.sort_values([USER_ID, SCORE, ITEM_ID], ascending=[True, False, True]).reset_index(
        drop=True
    )


@dataclass
class PopularityCandidateGenerator:
    """Generate candidates from globally popular and highly rated movies.

    The popularity score combines interaction volume and average rating. This is
    a strong production baseline because it is deterministic, robust for cold
    users, and easy to monitor.
    """

    item_scores_: pd.Series | None = None

    def fit(self, interactions: pd.DataFrame) -> "PopularityCandidateGenerator":
        """Fit item popularity scores from training interactions only."""

        item_stats = interactions.groupby(ITEM_ID).agg(
            rating_count=(RATING, "size"),
            mean_rating=(RATING, "mean"),
        )
        count_score = np.log1p(item_stats["rating_count"])
        rating_score = item_stats["mean_rating"] / 5.0
        raw_score = count_score * rating_score
        self.item_scores_ = raw_score.sort_values(ascending=False)
        return self

    def generate(
        self,
        user_ids: Iterable[int],
        seen_items_by_user: Dict[int, Set[int]],
        num_candidates: int,
    ) -> pd.DataFrame:
        """Generate top popularity candidates while excluding seen items."""

        if self.item_scores_ is None:
            raise RuntimeError("PopularityCandidateGenerator must be fit before generate.")

        rows: List[dict[str, float | int]] = []
        ranked_items = [(int(item_id), float(score)) for item_id, score in self.item_scores_.items()]
        for user_id in user_ids:
            user_id_int = int(user_id)
            seen_items = seen_items_by_user.get(user_id_int, set())
            user_count = 0
            for item_id, score in ranked_items:
                if item_id in seen_items:
                    continue
                rows.append({USER_ID: user_id_int, ITEM_ID: item_id, SCORE: score})
                user_count += 1
                if user_count >= num_candidates:
                    break
        return _candidate_rows_to_frame(rows)


@dataclass
class ItemSimilarityCandidateGenerator:
    """Generate candidates from item-item collaborative filtering.

    Item similarity is computed from a user-item rating matrix using cosine
    similarity. Candidate scores are weighted sums of a user's historical item
    ratings and the similarity between seen items and unseen candidate items.
    """

    num_similar_items: int = 50
    item_to_index_: Dict[int, int] | None = None
    item_ids_: np.ndarray | None = None
    similar_items_: Dict[int, List[tuple[int, float]]] | None = None
    user_history_: Dict[int, List[tuple[int, float]]] | None = None

    def fit(self, interactions: pd.DataFrame) -> "ItemSimilarityCandidateGenerator":
        """Fit item-item similarities from training interactions only."""

        user_item = interactions.pivot_table(
            index=USER_ID,
            columns=ITEM_ID,
            values=RATING,
            aggfunc="mean",
            fill_value=0.0,
        )
        item_ids = user_item.columns.to_numpy(dtype=np.int64)
        item_user_matrix = user_item.to_numpy(dtype=np.float64).T
        num_neighbors = min(self.num_similar_items + 1, len(item_ids))
        nearest_neighbors = NearestNeighbors(
            n_neighbors=num_neighbors,
            metric="cosine",
            algorithm="brute",
        )
        nearest_neighbors.fit(item_user_matrix)
        distances, neighbor_indices = nearest_neighbors.kneighbors(item_user_matrix)

        similar_items: Dict[int, List[tuple[int, float]]] = {}
        for index, item_id in enumerate(item_ids):
            item_neighbors: List[tuple[int, float]] = []
            for distance, neighbor_index in zip(distances[index], neighbor_indices[index]):
                neighbor_item_id = int(item_ids[neighbor_index])
                if neighbor_item_id == int(item_id):
                    continue
                similarity = 1.0 - float(distance)
                if similarity > 0.0:
                    item_neighbors.append((neighbor_item_id, similarity))
            similar_items[int(item_id)] = item_neighbors[: self.num_similar_items]

        history_frame = interactions.sort_values([USER_ID, ITEM_ID])
        user_history = {
            int(user_id): [
                (int(row[ITEM_ID]), float(row[RATING]))
                for _, row in user_rows.iterrows()
            ]
            for user_id, user_rows in history_frame.groupby(USER_ID)
        }

        self.item_to_index_ = {int(item_id): index for index, item_id in enumerate(item_ids)}
        self.item_ids_ = item_ids
        self.similar_items_ = similar_items
        self.user_history_ = user_history
        return self

    def generate(
        self,
        user_ids: Iterable[int],
        seen_items_by_user: Dict[int, Set[int]],
        num_candidates: int,
    ) -> pd.DataFrame:
        """Generate item-similarity candidates while excluding seen items."""

        if self.similar_items_ is None or self.user_history_ is None:
            raise RuntimeError("ItemSimilarityCandidateGenerator must be fit before generate.")

        rows: List[dict[str, float | int]] = []
        for user_id in user_ids:
            user_id_int = int(user_id)
            seen_items = seen_items_by_user.get(user_id_int, set())
            item_scores: Dict[int, float] = {}
            for seen_item_id, rating in self.user_history_.get(user_id_int, []):
                for candidate_item_id, similarity in self.similar_items_.get(seen_item_id, []):
                    if candidate_item_id in seen_items:
                        continue
                    item_scores[candidate_item_id] = item_scores.get(candidate_item_id, 0.0) + (
                        similarity * rating
                    )

            ranked_candidates = sorted(item_scores.items(), key=lambda item: (-item[1], item[0]))
            for item_id, score in ranked_candidates[:num_candidates]:
                rows.append({USER_ID: user_id_int, ITEM_ID: int(item_id), SCORE: float(score)})

        return _candidate_rows_to_frame(rows)


@dataclass
class MatrixFactorizationCandidateGenerator:
    """Generate candidates with latent factors from truncated SVD.

    This is a lightweight matrix-factorization retrieval model. It factorizes
    the user-item rating matrix into dense user and item embeddings, then scores
    candidate items with the dot product between user and item factors.
    """

    num_factors: int = 32
    random_seed: int = 42
    user_ids_: np.ndarray | None = None
    item_ids_: np.ndarray | None = None
    user_factors_: np.ndarray | None = None
    item_factors_: np.ndarray | None = None
    user_to_index_: Dict[int, int] | None = None

    def fit(self, interactions: pd.DataFrame) -> "MatrixFactorizationCandidateGenerator":
        """Fit latent user and item factors from training interactions only."""

        user_item = interactions.pivot_table(
            index=USER_ID,
            columns=ITEM_ID,
            values=RATING,
            aggfunc="mean",
            fill_value=0.0,
        )
        user_ids = user_item.index.to_numpy(dtype=np.int64)
        item_ids = user_item.columns.to_numpy(dtype=np.int64)
        matrix = user_item.to_numpy(dtype=np.float64)

        max_components = max(1, min(matrix.shape) - 1)
        n_components = min(self.num_factors, max_components)
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_seed)
        user_factors = svd.fit_transform(matrix)
        item_factors = svd.components_.T

        self.user_ids_ = user_ids
        self.item_ids_ = item_ids
        self.user_factors_ = user_factors
        self.item_factors_ = item_factors
        self.user_to_index_ = {int(user_id): index for index, user_id in enumerate(user_ids)}
        return self

    def generate(
        self,
        user_ids: Iterable[int],
        seen_items_by_user: Dict[int, Set[int]],
        num_candidates: int,
    ) -> pd.DataFrame:
        """Generate latent-factor candidates while excluding seen items."""

        if (
            self.user_to_index_ is None
            or self.item_ids_ is None
            or self.user_factors_ is None
            or self.item_factors_ is None
        ):
            raise RuntimeError("MatrixFactorizationCandidateGenerator must be fit before generate.")

        rows: List[dict[str, float | int]] = []
        for user_id in user_ids:
            user_id_int = int(user_id)
            user_index = self.user_to_index_.get(user_id_int)
            if user_index is None:
                continue
            scores = self.item_factors_ @ self.user_factors_[user_index]
            seen_items = seen_items_by_user.get(user_id_int, set())
            ranked_indices = np.argsort(-scores)
            user_count = 0
            for item_index in ranked_indices:
                item_id = int(self.item_ids_[item_index])
                if item_id in seen_items:
                    continue
                rows.append(
                    {
                        USER_ID: user_id_int,
                        ITEM_ID: item_id,
                        SCORE: float(scores[item_index]),
                    }
                )
                user_count += 1
                if user_count >= num_candidates:
                    break

        return _candidate_rows_to_frame(rows)


@dataclass
class HybridCandidateGenerator:
    """Combine multiple candidate generators into a single scored candidate set."""

    generators: List[CandidateGenerator]
    weights: List[float]

    def __post_init__(self) -> None:
        """Validate generator weights."""

        if len(self.generators) != len(self.weights):
            raise ValueError("generators and weights must have the same length.")
        if not self.generators:
            raise ValueError("At least one candidate generator is required.")
        if any(weight < 0.0 for weight in self.weights):
            raise ValueError("Candidate generator weights must be non-negative.")

    def fit(self, interactions: pd.DataFrame) -> "HybridCandidateGenerator":
        """Fit all candidate generators on training interactions."""

        for generator in self.generators:
            generator.fit(interactions)
        return self

    def generate(
        self,
        user_ids: Iterable[int],
        seen_items_by_user: Dict[int, Set[int]],
        num_candidates: int,
    ) -> pd.DataFrame:
        """Generate weighted hybrid candidates.

        Scores from each generator are min-max normalized per generator before
        weighting. This keeps a high-magnitude similarity score from dominating
        a lower-magnitude popularity score by accident.
        """

        weighted_frames: List[pd.DataFrame] = []
        for generator, weight in zip(self.generators, self.weights):
            frame = generator.generate(
                user_ids=user_ids,
                seen_items_by_user=seen_items_by_user,
                num_candidates=num_candidates,
            )
            if frame.empty or weight == 0.0:
                continue
            normalized = frame.copy()
            score_min = float(normalized[SCORE].min())
            score_max = float(normalized[SCORE].max())
            if score_max > score_min:
                normalized[SCORE] = (normalized[SCORE] - score_min) / (score_max - score_min)
            else:
                normalized[SCORE] = 1.0
            normalized[SCORE] = normalized[SCORE] * weight
            weighted_frames.append(normalized)

        if not weighted_frames:
            return pd.DataFrame(columns=[USER_ID, ITEM_ID, SCORE])

        combined = pd.concat(weighted_frames, ignore_index=True)
        combined = (
            combined.groupby([USER_ID, ITEM_ID], as_index=False)[SCORE]
            .sum()
            .sort_values([USER_ID, SCORE, ITEM_ID], ascending=[True, False, True])
        )
        return combined.groupby(USER_ID).head(num_candidates).reset_index(drop=True)


def create_default_candidate_generator(
    num_similar_items: int,
    num_factors: int,
    popularity_weight: float,
    similarity_weight: float,
    matrix_factorization_weight: float,
    random_seed: int = 42,
) -> HybridCandidateGenerator:
    """Create the default production-style hybrid candidate generator."""

    return HybridCandidateGenerator(
        generators=[
            PopularityCandidateGenerator(),
            ItemSimilarityCandidateGenerator(num_similar_items=num_similar_items),
            MatrixFactorizationCandidateGenerator(
                num_factors=num_factors,
                random_seed=random_seed,
            ),
        ],
        weights=[popularity_weight, similarity_weight, matrix_factorization_weight],
    )
