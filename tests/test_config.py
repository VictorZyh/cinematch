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
    assert config.evaluation.k_values == [5, 10, 20]
