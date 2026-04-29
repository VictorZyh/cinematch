"""Tests for the end-to-end pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cinematch.config import (
    ArtifactConfig,
    CandidateConfig,
    DataConfig,
    EvaluationConfig,
    ProjectConfig,
    RankingConfig,
    SplitConfig,
)
from cinematch.pipeline import run_pipeline


def _write_tiny_movielens_dataset(raw_dir: Path) -> None:
    """Write a tiny MovieLens-shaped dataset for pipeline tests."""

    raw_dir.mkdir(parents=True)
    ratings = pd.DataFrame(
        {
            "userId": [
                1, 1, 1, 1,
                2, 2, 2, 2,
                3, 3, 3, 3,
                4, 4, 4, 4,
            ],
            "movieId": [
                10, 11, 12, 13,
                10, 12, 13, 14,
                11, 12, 14, 15,
                10, 11, 15, 16,
            ],
            "rating": [
                5.0, 4.5, 2.0, 4.0,
                4.0, 5.0, 2.0, 4.5,
                5.0, 3.0, 4.5, 2.0,
                4.5, 4.0, 2.0, 5.0,
            ],
            "timestamp": [
                1, 2, 3, 4,
                1, 2, 3, 4,
                1, 2, 3, 4,
                1, 2, 3, 4,
            ],
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": [10, 11, 12, 13, 14, 15, 16],
            "title": ["A", "B", "C", "D", "E", "F", "G"],
            "genres": [
                "Action",
                "Drama",
                "Comedy",
                "Action|Drama",
                "Comedy|Drama",
                "Action|Comedy",
                "Drama",
            ],
        }
    )
    ratings.to_csv(raw_dir / "ratings.csv", index=False)
    movies.to_csv(raw_dir / "movies.csv", index=False)


def _config(tmp_path: Path) -> ProjectConfig:
    """Create a minimal ProjectConfig for end-to-end tests."""

    raw_dir = tmp_path / "raw"
    _write_tiny_movielens_dataset(raw_dir)
    return ProjectConfig(
        project_name="cinematch-test",
        random_seed=42,
        data=DataConfig(
            raw_dir=raw_dir,
            processed_dir=tmp_path / "processed",
            ratings_filename="ratings.csv",
            movies_filename="movies.csv",
            min_rating=0.5,
            positive_rating_threshold=4.0,
        ),
        split=SplitConfig(
            test_interactions_per_user=1,
            min_train_interactions_per_user=2,
        ),
        candidate=CandidateConfig(
            num_candidates=3,
            num_similar_items=2,
            num_factors=2,
            bpr_factors=2,
            bpr_epochs=1,
            bpr_samples_per_epoch=10,
            popularity_weight=0.5,
            similarity_weight=0.3,
            matrix_factorization_weight=0.1,
            bpr_weight=0.1,
        ),
        ranking=RankingConfig(
            negative_samples_per_positive=2,
            model_type="logistic_regression",
            max_iter=200,
        ),
        evaluation=EvaluationConfig(k_values=[1, 3]),
        artifacts=ArtifactConfig(output_dir=tmp_path / "artifacts"),
    )


def test_run_pipeline_creates_metrics_and_recommendations(tmp_path: Path) -> None:
    """A tiny dataset should run through the full pipeline and save artifacts."""

    config = _config(tmp_path)

    results = run_pipeline(config)

    assert results["status"] == "completed"
    assert results["train_rows"] == 12.0
    assert results["test_rows"] == 4.0
    assert (config.artifacts.output_dir / "metrics.json").exists()
    assert (config.artifacts.output_dir / "run_metadata.json").exists()
    assert (config.artifacts.output_dir / "recommendations.csv").exists()
    assert "precision_at_1" in results
    assert "recall_at_3" in results
