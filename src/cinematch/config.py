"""Typed configuration objects for the CineMatch recommendation pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class DataConfig:
    """Configuration for raw and processed MovieLens data."""

    raw_dir: Path
    processed_dir: Path
    ratings_filename: str
    movies_filename: str
    min_rating: float
    positive_rating_threshold: float


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for leakage-safe time-based splitting."""

    test_interactions_per_user: int
    min_train_interactions_per_user: int


@dataclass(frozen=True)
class CandidateConfig:
    """Configuration for candidate generation."""

    num_candidates: int
    num_similar_items: int
    num_factors: int
    popularity_weight: float
    similarity_weight: float
    matrix_factorization_weight: float


@dataclass(frozen=True)
class RankingConfig:
    """Configuration for supervised ranking model training."""

    negative_samples_per_positive: int
    model_type: str
    max_iter: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for top-K offline evaluation."""

    k_values: List[int]


@dataclass(frozen=True)
class ArtifactConfig:
    """Configuration for generated artifacts."""

    output_dir: Path


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level immutable configuration for an end-to-end run."""

    project_name: str
    random_seed: int
    data: DataConfig
    split: SplitConfig
    candidate: CandidateConfig
    ranking: RankingConfig
    evaluation: EvaluationConfig
    artifacts: ArtifactConfig


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load and validate a project configuration from a JSON file.

    Parameters
    ----------
    config_path:
        Path to a JSON configuration file.

    Returns
    -------
    ProjectConfig
        Typed configuration object used by the pipeline.
    """

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        raw_config: Dict[str, Any] = json.load(file)

    return ProjectConfig(
        project_name=str(raw_config["project_name"]),
        random_seed=int(raw_config["random_seed"]),
        data=DataConfig(
            raw_dir=Path(raw_config["data"]["raw_dir"]),
            processed_dir=Path(raw_config["data"]["processed_dir"]),
            ratings_filename=str(raw_config["data"]["ratings_filename"]),
            movies_filename=str(raw_config["data"]["movies_filename"]),
            min_rating=float(raw_config["data"]["min_rating"]),
            positive_rating_threshold=float(raw_config["data"]["positive_rating_threshold"]),
        ),
        split=SplitConfig(
            test_interactions_per_user=int(raw_config["split"]["test_interactions_per_user"]),
            min_train_interactions_per_user=int(raw_config["split"]["min_train_interactions_per_user"]),
        ),
        candidate=CandidateConfig(
            num_candidates=int(raw_config["candidate"]["num_candidates"]),
            num_similar_items=int(raw_config["candidate"]["num_similar_items"]),
            num_factors=int(raw_config["candidate"]["num_factors"]),
            popularity_weight=float(raw_config["candidate"]["popularity_weight"]),
            similarity_weight=float(raw_config["candidate"]["similarity_weight"]),
            matrix_factorization_weight=float(
                raw_config["candidate"]["matrix_factorization_weight"]
            ),
        ),
        ranking=RankingConfig(
            negative_samples_per_positive=int(raw_config["ranking"]["negative_samples_per_positive"]),
            model_type=str(raw_config["ranking"]["model_type"]),
            max_iter=int(raw_config["ranking"]["max_iter"]),
        ),
        evaluation=EvaluationConfig(
            k_values=[int(value) for value in raw_config["evaluation"]["k_values"]],
        ),
        artifacts=ArtifactConfig(
            output_dir=Path(raw_config["artifacts"]["output_dir"]),
        ),
    )
