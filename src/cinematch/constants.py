"""Shared column names and file-level constants for CineMatch."""

from __future__ import annotations

USER_ID = "userId"
ITEM_ID = "movieId"
RATING = "rating"
TIMESTAMP = "timestamp"
TITLE = "title"
GENRES = "genres"

LABEL = "label"
SCORE = "score"
RANK = "rank"

REQUIRED_RATINGS_COLUMNS = (USER_ID, ITEM_ID, RATING, TIMESTAMP)
REQUIRED_MOVIES_COLUMNS = (ITEM_ID, TITLE, GENRES)

UNKNOWN_GENRE_TOKEN = "(no genres listed)"
