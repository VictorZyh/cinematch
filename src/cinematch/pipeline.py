"""End-to-end orchestration for CineMatch."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cinematch.artifacts import default_artifact_paths, save_pickle
from cinematch.candidate import build_seen_items, create_default_candidate_generator
from cinematch.config import ProjectConfig, load_config
from cinematch.data_loader import load_movielens_data
from cinematch.evaluation import evaluate_recommendations
from cinematch.preprocessing import preprocess_movielens
from cinematch.ranking import train_ranker
from cinematch.split import assert_no_temporal_leakage, time_based_train_test_split
from cinematch.utils import ensure_directory, save_json, set_random_seed


def _save_recommendations(scored_candidates: pd.DataFrame, output_path: Path) -> None:
    """Save scored recommendation candidates to CSV."""

    ensure_directory(output_path.parent)
    scored_candidates.to_csv(output_path, index=False)


def run_pipeline(config: ProjectConfig) -> dict[str, float | str]:
    """Run the end-to-end recommendation pipeline.

    Parameters
    ----------
    config:
        Typed project configuration.

    Returns
    -------
    dict[str, float | str]
        Run metadata and evaluation metrics.
    """

    set_random_seed(config.random_seed)
    output_dir = ensure_directory(config.artifacts.output_dir)

    raw_data = load_movielens_data(config.data)
    ratings, movies = preprocess_movielens(
        ratings=raw_data.ratings,
        movies=raw_data.movies,
        min_rating=config.data.min_rating,
    )
    split = time_based_train_test_split(
        ratings=ratings,
        test_interactions_per_user=config.split.test_interactions_per_user,
        min_train_interactions_per_user=config.split.min_train_interactions_per_user,
    )
    assert_no_temporal_leakage(split)

    if split.train.empty or split.test.empty:
        raise ValueError("Train/test split is empty. Check split configuration and input data.")

    train_user_ids = split.train["userId"].drop_duplicates().tolist()
    test_user_ids = split.test["userId"].drop_duplicates().tolist()
    seen_items_by_user = build_seen_items(split.train)

    candidate_generator = create_default_candidate_generator(
        num_similar_items=config.candidate.num_similar_items,
        popularity_weight=config.candidate.popularity_weight,
        similarity_weight=config.candidate.similarity_weight,
    ).fit(split.train)

    train_candidates = candidate_generator.generate(
        user_ids=train_user_ids,
        seen_items_by_user=seen_items_by_user,
        num_candidates=config.candidate.num_candidates,
    )
    feature_builder, ranker, training_frame = train_ranker(
        candidate_rows=train_candidates,
        train_interactions=split.train,
        movies=movies,
        positive_rating_threshold=config.data.positive_rating_threshold,
        negatives_per_positive=config.ranking.negative_samples_per_positive,
        random_seed=config.random_seed,
        max_iter=config.ranking.max_iter,
    )

    test_candidates = candidate_generator.generate(
        user_ids=test_user_ids,
        seen_items_by_user=seen_items_by_user,
        num_candidates=config.candidate.num_candidates,
    )
    test_features = feature_builder.transform(test_candidates)
    scored_candidates = ranker.predict_scores(test_features)

    metrics = evaluate_recommendations(
        scored_candidates=scored_candidates,
        test_interactions=split.test,
        positive_rating_threshold=config.data.positive_rating_threshold,
        k_values=config.evaluation.k_values,
    )

    recommendations_path = output_dir / "recommendations.csv"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "run_metadata.json"
    artifact_paths = default_artifact_paths(output_dir)
    _save_recommendations(scored_candidates, recommendations_path)
    save_json(metrics, metrics_path)
    save_pickle(candidate_generator, artifact_paths.candidate_generator)
    save_pickle(feature_builder, artifact_paths.feature_builder)
    save_pickle(ranker, artifact_paths.ranker)
    save_pickle(split.train, artifact_paths.train_interactions)

    metadata: dict[str, float | str] = {
        "project_name": config.project_name,
        "status": "completed",
        "output_dir": str(output_dir),
        "ratings_rows": float(len(ratings)),
        "movies_rows": float(len(movies)),
        "train_rows": float(len(split.train)),
        "test_rows": float(len(split.test)),
        "train_candidate_rows": float(len(train_candidates)),
        "test_candidate_rows": float(len(test_candidates)),
        "training_examples": float(len(training_frame)),
        "recommendations_path": str(recommendations_path),
        "metrics_path": str(metrics_path),
        "candidate_generator_path": str(artifact_paths.candidate_generator),
        "feature_builder_path": str(artifact_paths.feature_builder),
        "ranker_path": str(artifact_paths.ranker),
        "train_interactions_path": str(artifact_paths.train_interactions),
    }
    save_json(metadata, metadata_path)
    return {**metadata, **metrics}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run the CineMatch recommendation pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
        help="Path to the JSON configuration file.",
    )
    return parser


def main() -> None:
    """Console entry point for running the recommendation pipeline."""

    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    results = run_pipeline(config)
    print(f"CineMatch pipeline status: {results['status']}")
    print(f"Metrics saved to: {results['metrics_path']}")
