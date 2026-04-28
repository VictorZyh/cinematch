"""Supervised ranking model utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from cinematch.constants import ITEM_ID, LABEL, SCORE, USER_ID
from cinematch.features import FEATURE_COLUMNS, FeatureBuilder


def build_positive_pairs(
    interactions: pd.DataFrame,
    positive_rating_threshold: float,
) -> pd.DataFrame:
    """Create positive user-item labels from high-rating interactions."""

    positives = interactions.loc[
        interactions["rating"] >= positive_rating_threshold,
        [USER_ID, ITEM_ID],
    ].drop_duplicates()
    positives[LABEL] = 1
    return positives.reset_index(drop=True)


def sample_negative_pairs(
    candidates: pd.DataFrame,
    positive_pairs: pd.DataFrame,
    negatives_per_positive: int,
    random_seed: int,
) -> pd.DataFrame:
    """Sample candidate rows that are not positive interactions as negatives."""

    if negatives_per_positive <= 0:
        raise ValueError("negatives_per_positive must be positive.")

    positive_key_set: Set[tuple[int, int]] = {
        (int(row[USER_ID]), int(row[ITEM_ID])) for _, row in positive_pairs.iterrows()
    }
    negative_pool = candidates.loc[
        [
            (int(row[USER_ID]), int(row[ITEM_ID])) not in positive_key_set
            for _, row in candidates.iterrows()
        ],
        [USER_ID, ITEM_ID],
    ].drop_duplicates()

    max_negatives = len(positive_pairs) * negatives_per_positive
    if negative_pool.empty or max_negatives == 0:
        negatives = negative_pool
    else:
        sample_size = min(len(negative_pool), max_negatives)
        negatives = negative_pool.sample(n=sample_size, random_state=random_seed)

    negatives = negatives.copy()
    negatives[LABEL] = 0
    return negatives.reset_index(drop=True)


def build_training_candidates(
    candidate_rows: pd.DataFrame,
    interactions: pd.DataFrame,
    positive_rating_threshold: float,
    negatives_per_positive: int,
    random_seed: int,
) -> pd.DataFrame:
    """Build labeled user-item candidates for supervised ranker training."""

    positives = build_positive_pairs(interactions, positive_rating_threshold)
    positive_candidates = positives.merge(
        candidate_rows[[USER_ID, ITEM_ID, SCORE]],
        on=[USER_ID, ITEM_ID],
        how="left",
    )
    positive_candidates[SCORE] = positive_candidates[SCORE].fillna(1.0)

    negatives = sample_negative_pairs(
        candidates=candidate_rows,
        positive_pairs=positives,
        negatives_per_positive=negatives_per_positive,
        random_seed=random_seed,
    )
    negative_candidates = negatives.merge(
        candidate_rows[[USER_ID, ITEM_ID, SCORE]],
        on=[USER_ID, ITEM_ID],
        how="left",
    )
    labeled_candidates = pd.concat(
        [positive_candidates, negative_candidates],
        ignore_index=True,
    )
    return labeled_candidates.drop_duplicates([USER_ID, ITEM_ID, LABEL]).reset_index(drop=True)


@dataclass
class SklearnRanker:
    """Configurable sklearn-based ranker for candidate scoring."""

    model_type: str = "logistic_regression"
    max_iter: int = 1000
    model_: LogisticRegression | HistGradientBoostingClassifier | None = None

    def _build_model(self) -> LogisticRegression | HistGradientBoostingClassifier:
        """Instantiate the configured sklearn ranking model."""

        if self.model_type == "logistic_regression":
            return LogisticRegression(max_iter=self.max_iter, random_state=0)
        if self.model_type == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(
                max_iter=self.max_iter,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=0,
            )
        raise ValueError(f"Unsupported ranker model_type: {self.model_type}")

    def fit(self, feature_frame: pd.DataFrame) -> "SklearnRanker":
        """Fit the ranking model from labeled feature rows."""

        if LABEL not in feature_frame.columns:
            raise ValueError(f"feature_frame must contain '{LABEL}'.")
        labels = feature_frame[LABEL].astype(int)
        if labels.nunique() < 2:
            raise ValueError("Ranker training requires both positive and negative labels.")

        model = self._build_model()
        model.fit(feature_frame[FEATURE_COLUMNS], labels)
        self.model_ = model
        return self

    def predict_scores(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Predict ranking scores for candidate feature rows."""

        if self.model_ is None:
            raise RuntimeError("SklearnRanker must be fit before predict_scores.")

        scored = feature_frame[[USER_ID, ITEM_ID]].copy()
        scored[SCORE] = self.model_.predict_proba(feature_frame[FEATURE_COLUMNS])[:, 1]
        return scored.sort_values([USER_ID, SCORE, ITEM_ID], ascending=[True, False, True]).reset_index(
            drop=True
        )


def train_ranker(
    candidate_rows: pd.DataFrame,
    train_interactions: pd.DataFrame,
    movies: pd.DataFrame,
    positive_rating_threshold: float,
    negatives_per_positive: int,
    random_seed: int,
    max_iter: int,
    model_type: str = "logistic_regression",
) -> tuple[FeatureBuilder, SklearnRanker, pd.DataFrame]:
    """Build ranking features, train a ranker, and return training diagnostics."""

    feature_builder = FeatureBuilder().fit(train_interactions, movies)
    labeled_candidates = build_training_candidates(
        candidate_rows=candidate_rows,
        interactions=train_interactions,
        positive_rating_threshold=positive_rating_threshold,
        negatives_per_positive=negatives_per_positive,
        random_seed=random_seed,
    )
    feature_frame = feature_builder.transform(labeled_candidates)
    feature_frame[LABEL] = labeled_candidates[LABEL].values

    ranker = SklearnRanker(model_type=model_type, max_iter=max_iter).fit(feature_frame)
    return feature_builder, ranker, feature_frame


LogisticRegressionRanker = SklearnRanker
