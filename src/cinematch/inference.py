"""Batch inference utilities for trained CineMatch artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from cinematch.artifacts import default_artifact_paths, load_pickle
from cinematch.candidate import build_seen_items
from cinematch.constants import USER_ID
from cinematch.evaluation import top_k_recommendations


def load_user_ids(path: str | Path) -> list[int]:
    """Load user ids from a text file or a CSV file with a ``userId`` column."""

    input_path = Path(path)
    if input_path.suffix.lower() == ".csv":
        frame = pd.read_csv(input_path)
        if USER_ID not in frame.columns:
            raise ValueError(f"CSV user file must contain a '{USER_ID}' column.")
        return [int(user_id) for user_id in frame[USER_ID].dropna().unique()]

    user_ids = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            user_ids.append(int(stripped))
    return user_ids


def generate_recommendations(
    artifact_dir: str | Path,
    user_ids: Iterable[int],
    num_candidates: int,
    top_k: int,
) -> pd.DataFrame:
    """Generate top-k movie recommendations from persisted artifacts."""

    paths = default_artifact_paths(artifact_dir)
    candidate_generator = load_pickle(paths.candidate_generator)
    feature_builder = load_pickle(paths.feature_builder)
    ranker = load_pickle(paths.ranker)
    train_interactions = load_pickle(paths.train_interactions)

    requested_user_ids = [int(user_id) for user_id in user_ids]
    seen_items_by_user = build_seen_items(train_interactions)
    candidates = candidate_generator.generate(
        user_ids=requested_user_ids,
        seen_items_by_user=seen_items_by_user,
        num_candidates=num_candidates,
    )
    if candidates.empty:
        return candidates

    features = feature_builder.transform(candidates)
    scored = ranker.predict_scores(features)
    return top_k_recommendations(scored, k=top_k)
