"""Tests for configuration loading."""

from __future__ import annotations

from cinematch.config import load_config


def test_load_default_config_has_expected_values() -> None:
    """The default config should load into typed objects with stable values."""

    config = load_config("configs/default.json")

    assert config.project_name == "cinematch"
    assert config.random_seed == 42
    assert config.data.ratings_filename == "ratings.csv"
    assert config.split.test_interactions_per_user == 1
    assert config.candidate.num_candidates == 500
    assert config.candidate.num_similar_items == 100
    assert config.candidate.num_factors == 64
    assert config.candidate.bpr_factors == 48
    assert config.candidate.bpr_epochs == 8
    assert config.candidate.bpr_weight == 0.10
    assert config.ranking.model_type == "logistic_regression"
    assert config.evaluation.k_values == [5, 10, 20]
