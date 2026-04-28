"""Feature engineering for candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

import numpy as np
import pandas as pd

from cinematch.constants import GENRES, ITEM_ID, RATING, SCORE, USER_ID


FEATURE_COLUMNS = [
    "candidate_score",
    "user_rating_count",
    "user_avg_rating",
    "item_rating_count",
    "item_avg_rating",
    "item_popularity_score",
    "user_item_genre_overlap",
]


@dataclass
class FeatureBuilder:
    """Build leakage-safe ranking features from training interactions only."""

    global_avg_rating_: float = 0.0
    global_item_count_: float = 0.0
    user_stats_: pd.DataFrame | None = None
    item_stats_: pd.DataFrame | None = None
    item_genres_: Dict[int, Set[str]] | None = None
    user_genres_: Dict[int, Set[str]] | None = None

    def fit(self, train_interactions: pd.DataFrame, movies: pd.DataFrame) -> "FeatureBuilder":
        """Fit aggregate feature state using only training interactions."""

        self.global_avg_rating_ = float(train_interactions[RATING].mean())

        user_stats = train_interactions.groupby(USER_ID).agg(
            user_rating_count=(RATING, "size"),
            user_avg_rating=(RATING, "mean"),
        )

        item_stats = train_interactions.groupby(ITEM_ID).agg(
            item_rating_count=(RATING, "size"),
            item_avg_rating=(RATING, "mean"),
        )
        item_stats["item_popularity_score"] = np.log1p(item_stats["item_rating_count"])
        self.global_item_count_ = float(item_stats["item_rating_count"].mean())

        item_genres = {
            int(row[ITEM_ID]): set(row["genre_list"])
            for _, row in movies[[ITEM_ID, "genre_list"]].iterrows()
        }

        train_with_genres = train_interactions[[USER_ID, ITEM_ID]].copy()
        train_with_genres["genre_list"] = train_with_genres[ITEM_ID].map(item_genres)
        user_genres: Dict[int, Set[str]] = {}
        for user_id, user_rows in train_with_genres.groupby(USER_ID):
            genres: Set[str] = set()
            for genre_list in user_rows["genre_list"]:
                if isinstance(genre_list, set):
                    genres.update(genre_list)
            user_genres[int(user_id)] = genres

        self.user_stats_ = user_stats
        self.item_stats_ = item_stats
        self.item_genres_ = item_genres
        self.user_genres_ = user_genres
        return self

    def transform(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Add ranking features to candidate rows.

        Unknown users or items receive conservative global fallback values so
        the ranker can still score cold-start rows.
        """

        if self.user_stats_ is None or self.item_stats_ is None:
            raise RuntimeError("FeatureBuilder must be fit before transform.")

        features = candidates[[USER_ID, ITEM_ID, SCORE]].copy()
        features = features.rename(columns={SCORE: "candidate_score"})
        features = features.merge(self.user_stats_, left_on=USER_ID, right_index=True, how="left")
        features = features.merge(self.item_stats_, left_on=ITEM_ID, right_index=True, how="left")

        features["user_rating_count"] = features["user_rating_count"].fillna(0.0)
        features["user_avg_rating"] = features["user_avg_rating"].fillna(self.global_avg_rating_)
        features["item_rating_count"] = features["item_rating_count"].fillna(self.global_item_count_)
        features["item_avg_rating"] = features["item_avg_rating"].fillna(self.global_avg_rating_)
        features["item_popularity_score"] = features["item_popularity_score"].fillna(0.0)
        features["user_item_genre_overlap"] = [
            self._genre_overlap(int(user_id), int(item_id))
            for user_id, item_id in zip(features[USER_ID], features[ITEM_ID])
        ]
        return features[[USER_ID, ITEM_ID, *FEATURE_COLUMNS]]

    def _genre_overlap(self, user_id: int, item_id: int) -> float:
        """Compute the fraction of item genres present in the user's history."""

        if self.user_genres_ is None or self.item_genres_ is None:
            raise RuntimeError("FeatureBuilder must be fit before computing genre overlap.")

        user_genres = self.user_genres_.get(user_id, set())
        item_genres = self.item_genres_.get(item_id, set())
        if not item_genres:
            return 0.0
        return float(len(user_genres.intersection(item_genres)) / len(item_genres))
