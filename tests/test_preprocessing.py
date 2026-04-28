"""Tests for MovieLens preprocessing."""

from __future__ import annotations

import pandas as pd

from cinematch.constants import ITEM_ID, RATING, USER_ID
from cinematch.preprocessing import (
    clean_movies,
    clean_ratings,
    filter_ratings_to_known_movies,
    parse_genres,
    preprocess_movielens,
)


def test_parse_genres_handles_pipe_delimited_and_missing_values() -> None:
    assert parse_genres("Action|Adventure|Sci-Fi") == ["Action", "Adventure", "Sci-Fi"]
    assert parse_genres("(no genres listed)") == []
    assert parse_genres(None) == []


def test_clean_ratings_converts_types_filters_and_sorts() -> None:
    ratings = pd.DataFrame(
        {
            "userId": ["2", "1", "1"],
            "movieId": ["20", "10", "11"],
            "rating": ["0.0", "4.0", "3.5"],
            "timestamp": ["3", "2", "1"],
        }
    )

    cleaned = clean_ratings(ratings, min_rating=0.5)

    assert cleaned[USER_ID].tolist() == [1, 1]
    assert cleaned[ITEM_ID].tolist() == [11, 10]
    assert cleaned[RATING].tolist() == [3.5, 4.0]
    assert str(cleaned[USER_ID].dtype) == "int64"
    assert str(cleaned[RATING].dtype) == "float64"


def test_clean_movies_adds_genre_list_and_deduplicates_items() -> None:
    movies = pd.DataFrame(
        {
            "movieId": [10, 10, 11],
            "title": ["Old Title", "New Title", "Missing Genres"],
            "genres": ["Drama", "Comedy|Drama", "(no genres listed)"],
        }
    )

    cleaned = clean_movies(movies)

    assert cleaned[ITEM_ID].tolist() == [10, 11]
    assert cleaned.loc[cleaned[ITEM_ID] == 10, "genre_list"].iloc[0] == ["Comedy", "Drama"]
    assert cleaned.loc[cleaned[ITEM_ID] == 11, "genre_list"].iloc[0] == []


def test_filter_ratings_to_known_movies_removes_unknown_items() -> None:
    ratings = pd.DataFrame({"movieId": [10, 99], "userId": [1, 1], "rating": [4.0, 5.0]})
    movies = pd.DataFrame({"movieId": [10]})

    filtered = filter_ratings_to_known_movies(ratings, movies)

    assert filtered[ITEM_ID].tolist() == [10]


def test_preprocess_movielens_runs_all_steps() -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [10, 99, 11],
            "rating": [4.0, 5.0, 0.0],
            "timestamp": [1, 2, 3],
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": [10, 11],
            "title": ["Known", "Low Rating"],
            "genres": ["Drama", "Comedy"],
        }
    )

    cleaned_ratings, cleaned_movies = preprocess_movielens(
        ratings,
        movies,
        min_rating=0.5,
    )

    assert cleaned_ratings[ITEM_ID].tolist() == [10]
    assert cleaned_movies[ITEM_ID].tolist() == [10, 11]
