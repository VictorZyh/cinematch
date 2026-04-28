"""Tests for the MovieLens download helper."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.download_movielens import (
    download_movielens_dataset,
)


def _write_fake_movielens_zip(path: Path) -> None:
    """Create a tiny zip archive with the same folder shape as MovieLens."""

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ml-latest-small/ratings.csv", "userId,movieId,rating,timestamp\n")
        archive.writestr("ml-latest-small/movies.csv", "movieId,title,genres\n")


def test_download_movielens_dataset_reuses_existing_valid_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ml-latest-small"
    dataset_dir.mkdir()
    (dataset_dir / "ratings.csv").write_text("userId,movieId,rating,timestamp\n", encoding="utf-8")
    (dataset_dir / "movies.csv").write_text("movieId,title,genres\n", encoding="utf-8")

    result = download_movielens_dataset(output_dir=tmp_path)

    assert result == dataset_dir


def test_download_movielens_dataset_rejects_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported dataset"):
        download_movielens_dataset(dataset_name="unknown", output_dir=tmp_path)


def test_download_movielens_dataset_extracts_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_archive = tmp_path / "fake.zip"
    _write_fake_movielens_zip(fake_archive)

    def fake_download_file(url: str, destination: Path) -> None:
        destination.write_bytes(fake_archive.read_bytes())

    monkeypatch.setattr("scripts.download_movielens._download_file", fake_download_file)

    result = download_movielens_dataset(output_dir=tmp_path / "raw")

    assert (result / "ratings.csv").exists()
    assert (result / "movies.csv").exists()
