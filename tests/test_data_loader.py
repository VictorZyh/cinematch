"""Tests for MovieLens data loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cinematch.config import DataConfig
from cinematch.data_loader import DataValidationError, load_movielens_data, validate_columns


def _data_config(raw_dir: Path) -> DataConfig:
    return DataConfig(
        raw_dir=raw_dir,
        processed_dir=raw_dir / "processed",
        ratings_filename="ratings.csv",
        movies_filename="movies.csv",
        min_rating=0.5,
        positive_rating_threshold=4.0,
    )


def test_validate_columns_raises_for_missing_column() -> None:
    frame = pd.DataFrame({"userId": [1], "movieId": [10]})

    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_columns(frame, ["userId", "movieId", "rating"], "ratings")


def test_load_movielens_data_reads_expected_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "ml"
    raw_dir.mkdir()
    pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
            "rating": [4.5],
            "timestamp": [1000],
        }
    ).to_csv(raw_dir / "ratings.csv", index=False)
    pd.DataFrame(
        {
            "movieId": [10],
            "title": ["Example Movie (2000)"],
            "genres": ["Drama"],
        }
    ).to_csv(raw_dir / "movies.csv", index=False)

    data = load_movielens_data(_data_config(raw_dir))

    assert data.ratings.shape == (1, 4)
    assert data.movies.shape == (1, 3)


def test_load_movielens_data_raises_for_missing_file(tmp_path: Path) -> None:
    raw_dir = tmp_path / "ml"
    raw_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="ratings.csv"):
        load_movielens_data(_data_config(raw_dir))
