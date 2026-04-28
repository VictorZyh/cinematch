"""Tests for artifact persistence helpers."""

from __future__ import annotations

from cinematch.artifacts import default_artifact_paths, load_pickle, save_pickle


def test_default_artifact_paths_are_under_output_dir(tmp_path) -> None:
    paths = default_artifact_paths(tmp_path)

    assert paths.candidate_generator.parent == tmp_path
    assert paths.feature_builder.name == "feature_builder.pkl"
    assert paths.ranker.name == "ranker.pkl"
    assert paths.train_interactions.name == "train_interactions.pkl"


def test_save_and_load_pickle_round_trip(tmp_path) -> None:
    path = tmp_path / "nested" / "object.pkl"

    save_pickle({"value": 7}, path)

    assert load_pickle(path) == {"value": 7}
