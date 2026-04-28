"""Data loading and schema validation for MovieLens files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from cinematch.config import DataConfig
from cinematch.constants import REQUIRED_MOVIES_COLUMNS, REQUIRED_RATINGS_COLUMNS


class DataValidationError(ValueError):
    """Raised when an input data file does not match the expected schema."""


@dataclass(frozen=True)
class MovieLensData:
    """Container for raw MovieLens tables used by the recommendation pipeline."""

    ratings: pd.DataFrame
    movies: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file and raise a clear error if it is missing."""

    if not path.exists():
        raise FileNotFoundError(f"Expected data file does not exist: {path}")
    return pd.read_csv(path)


def validate_columns(frame: pd.DataFrame, required_columns: Iterable[str], table_name: str) -> None:
    """Validate that a DataFrame contains all required columns.

    Parameters
    ----------
    frame:
        DataFrame to validate.
    required_columns:
        Column names required by downstream modules.
    table_name:
        Human-readable table name used in error messages.

    Raises
    ------
    DataValidationError
        If one or more required columns are missing.
    """

    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise DataValidationError(
            f"{table_name} is missing required columns: {missing_columns}"
        )


def load_movielens_data(config: DataConfig) -> MovieLensData:
    """Load raw MovieLens ratings and movies data from disk.

    The function intentionally performs only I/O and schema validation. Cleaning,
    type conversion, filtering, and feature parsing are handled by preprocessing
    functions so each step remains independently testable.
    """

    ratings_path = config.raw_dir / config.ratings_filename
    movies_path = config.raw_dir / config.movies_filename

    ratings = _read_csv(ratings_path)
    movies = _read_csv(movies_path)

    validate_columns(ratings, REQUIRED_RATINGS_COLUMNS, "ratings")
    validate_columns(movies, REQUIRED_MOVIES_COLUMNS, "movies")

    return MovieLensData(ratings=ratings, movies=movies)
