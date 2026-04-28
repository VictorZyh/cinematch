"""Tests for batch inference utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cinematch.config import (
    ArtifactConfig,
    CandidateConfig,
    DataConfig,
    EvaluationConfig,
    ProjectConfig,
    RankingConfig,
    SplitConfig,
)
from cinematch.inference import generate_recommendations, load_user_ids
from cinematch.pipeline import run_pipeline


def _write_dataset(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "movieId": [10, 11, 12, 13, 10, 12, 13, 14, 11, 12, 14, 15],
            "rating": [5.0, 4.5, 2.0, 4.0, 4.0, 5.0, 2.0, 4.5, 5.0, 3.0, 4.5, 2.0],
            "timestamp": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        }
    ).to_csv(raw_dir / "ratings.csv", index=False)
    pd.DataFrame(
        {
            "movieId": [10, 11, 12, 13, 14, 15],
            "title": ["A", "B", "C", "D", "E", "F"],
            "genres": ["Action", "Drama", "Comedy", "Action|Drama", "Comedy|Drama", "Action|Comedy"],
        }
    ).to_csv(raw_dir / "movies.csv", index=False)


def _config(tmp_path: Path) -> ProjectConfig:
    raw_dir = tmp_path / "raw"
    _write_dataset(raw_dir)
    return ProjectConfig(
        project_name="cinematch-inference-test",
        random_seed=42,
        data=DataConfig(
            raw_dir=raw_dir,
            processed_dir=tmp_path / "processed",
            ratings_filename="ratings.csv",
            movies_filename="movies.csv",
            min_rating=0.5,
            positive_rating_threshold=4.0,
        ),
        split=SplitConfig(test_interactions_per_user=1, min_train_interactions_per_user=2),
        candidate=CandidateConfig(
            num_candidates=3,
            num_similar_items=2,
            num_factors=2,
            popularity_weight=0.5,
            similarity_weight=0.3,
            matrix_factorization_weight=0.2,
        ),
        ranking=RankingConfig(
            negative_samples_per_positive=2,
            model_type="logistic_regression",
            max_iter=200,
        ),
        evaluation=EvaluationConfig(k_values=[1, 3]),
        artifacts=ArtifactConfig(output_dir=tmp_path / "artifacts"),
    )


def test_load_user_ids_from_text_and_csv(tmp_path: Path) -> None:
    text_path = tmp_path / "users.txt"
    text_path.write_text("1\n2\n\n3\n", encoding="utf-8")
    csv_path = tmp_path / "users.csv"
    pd.DataFrame({"userId": [1, 2, 2]}).to_csv(csv_path, index=False)

    assert load_user_ids(text_path) == [1, 2, 3]
    assert load_user_ids(csv_path) == [1, 2]


def test_load_user_ids_rejects_csv_without_user_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"id": [1]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="userId"):
        load_user_ids(csv_path)


def test_generate_recommendations_from_saved_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_pipeline(config)

    recommendations = generate_recommendations(
        artifact_dir=config.artifacts.output_dir,
        user_ids=[1, 2],
        num_candidates=3,
        top_k=2,
    )

    assert not recommendations.empty
    assert recommendations.groupby("userId").size().max() <= 2
