"""Persistence helpers for trained CineMatch artifacts."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cinematch.utils import ensure_directory


@dataclass(frozen=True)
class ArtifactPaths:
    """Filesystem paths for persisted recommendation artifacts."""

    candidate_generator: Path
    feature_builder: Path
    ranker: Path
    train_interactions: Path


def default_artifact_paths(output_dir: str | Path) -> ArtifactPaths:
    """Return default artifact paths under an output directory."""

    directory = Path(output_dir)
    return ArtifactPaths(
        candidate_generator=directory / "candidate_generator.pkl",
        feature_builder=directory / "feature_builder.pkl",
        ranker=directory / "ranker.pkl",
        train_interactions=directory / "train_interactions.pkl",
    )


def save_pickle(obj: Any, path: str | Path) -> None:
    """Serialize an object to a pickle file."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("wb") as file:
        pickle.dump(obj, file)


def load_pickle(path: str | Path) -> Any:
    """Load a pickle file from disk."""

    input_path = Path(path)
    with input_path.open("rb") as file:
        return pickle.load(file)
