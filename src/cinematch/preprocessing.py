"""Preprocessing utilities for MovieLens recommendation data."""

from __future__ import annotations

from typing import List

import pandas as pd

from cinematch.constants import (
    GENRES,
    ITEM_ID,
    RATING,
    TIMESTAMP,
    TITLE,
    UNKNOWN_GENRE_TOKEN,
    USER_ID,
)


def clean_ratings(ratings: pd.DataFrame, min_rating: float) -> pd.DataFrame:
    """Clean MovieLens ratings while preserving event timestamps.

    Parameters
    ----------
    ratings:
        Raw ratings table with user, item, rating, and timestamp columns.
    min_rating:
        Minimum allowed rating value. Rows below this threshold are removed.

    Returns
    -------
    pd.DataFrame
        Cleaned ratings sorted by user and timestamp.
    """

    cleaned = ratings.copy()
    cleaned[USER_ID] = cleaned[USER_ID].astype("int64")
    cleaned[ITEM_ID] = cleaned[ITEM_ID].astype("int64")
    cleaned[RATING] = cleaned[RATING].astype("float64")
    cleaned[TIMESTAMP] = cleaned[TIMESTAMP].astype("int64")

    cleaned = cleaned.loc[cleaned[RATING] >= min_rating]
    cleaned = cleaned.drop_duplicates(
        subset=[USER_ID, ITEM_ID, TIMESTAMP], keep="last"
    )
    cleaned = cleaned.sort_values([USER_ID, TIMESTAMP, ITEM_ID]).reset_index(drop=True)
    return cleaned


def parse_genres(value: object) -> List[str]:
    """Parse a MovieLens pipe-delimited genre string into a clean list.

    MovieLens uses ``(no genres listed)`` for missing genre metadata. This
    function normalizes that value to an empty list so downstream feature logic
    can treat missing genre metadata explicitly.
    """

    if value is None or pd.isna(value):
        return []

    genre_text = str(value).strip()
    if not genre_text or genre_text == UNKNOWN_GENRE_TOKEN:
        return []

    return [genre.strip() for genre in genre_text.split("|") if genre.strip()]


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Clean MovieLens movie metadata and add parsed genre lists."""

    cleaned = movies.copy()
    cleaned[ITEM_ID] = cleaned[ITEM_ID].astype("int64")
    cleaned[TITLE] = cleaned[TITLE].astype("string")
    cleaned[GENRES] = cleaned[GENRES].astype("string")
    cleaned["genre_list"] = cleaned[GENRES].apply(parse_genres)
    cleaned = cleaned.drop_duplicates(subset=[ITEM_ID], keep="last")
    cleaned = cleaned.sort_values(ITEM_ID).reset_index(drop=True)
    return cleaned


def filter_ratings_to_known_movies(
    ratings: pd.DataFrame, movies: pd.DataFrame
) -> pd.DataFrame:
    """Keep only rating rows whose item id exists in the movie metadata table."""

    known_item_ids = set(movies[ITEM_ID].unique())
    filtered = ratings.loc[ratings[ITEM_ID].isin(known_item_ids)].copy()
    return filtered.reset_index(drop=True)


def preprocess_movielens(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    min_rating: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all MovieLens preprocessing steps in the correct order."""

    cleaned_movies = clean_movies(movies)
    cleaned_ratings = clean_ratings(ratings, min_rating=min_rating)
    cleaned_ratings = filter_ratings_to_known_movies(cleaned_ratings, cleaned_movies)
    return cleaned_ratings, cleaned_movies
