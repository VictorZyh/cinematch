"""Tests for leakage-safe train/test splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from cinematch.constants import ITEM_ID, TIMESTAMP, USER_ID
from cinematch.split import assert_no_temporal_leakage, time_based_train_test_split


def test_time_based_train_test_split_keeps_latest_interaction_for_test() -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [10, 11, 12, 20, 21, 22],
            "rating": [4.0, 3.5, 5.0, 4.5, 2.0, 5.0],
            "timestamp": [100, 200, 300, 50, 60, 70],
        }
    )

    split = time_based_train_test_split(
        ratings,
        test_interactions_per_user=1,
        min_train_interactions_per_user=2,
    )

    assert split.test[ITEM_ID].tolist() == [12, 22]
    assert split.train.groupby(USER_ID).size().to_dict() == {1: 2, 2: 2}
    assert_no_temporal_leakage(split)


def test_time_based_train_test_split_filters_users_with_insufficient_history() -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 2],
            "movieId": [10, 11, 20, 21, 22],
            "rating": [4.0, 3.5, 4.5, 2.0, 5.0],
            "timestamp": [100, 200, 50, 60, 70],
        }
    )

    split = time_based_train_test_split(
        ratings,
        test_interactions_per_user=1,
        min_train_interactions_per_user=2,
    )

    assert set(split.train[USER_ID].unique()) == {2}
    assert set(split.test[USER_ID].unique()) == {2}


def test_time_based_train_test_split_is_stable_for_unsorted_input() -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1],
            "movieId": [12, 10, 11],
            "rating": [5.0, 4.0, 3.5],
            "timestamp": [300, 100, 200],
        }
    )

    split = time_based_train_test_split(
        ratings,
        test_interactions_per_user=1,
        min_train_interactions_per_user=2,
    )

    assert split.train[TIMESTAMP].tolist() == [100, 200]
    assert split.test[TIMESTAMP].tolist() == [300]


def test_time_based_train_test_split_returns_empty_when_no_user_is_eligible() -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 1],
            "movieId": [10, 11],
            "rating": [4.0, 3.5],
            "timestamp": [100, 200],
        }
    )

    split = time_based_train_test_split(
        ratings,
        test_interactions_per_user=1,
        min_train_interactions_per_user=3,
    )

    assert split.train.empty
    assert split.test.empty


def test_time_based_train_test_split_rejects_invalid_parameters() -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1],
            "movieId": [10, 11, 12],
            "rating": [4.0, 3.5, 5.0],
            "timestamp": [100, 200, 300],
        }
    )

    with pytest.raises(ValueError, match="test_interactions_per_user"):
        time_based_train_test_split(ratings, 0, 2)

    with pytest.raises(ValueError, match="min_train_interactions_per_user"):
        time_based_train_test_split(ratings, 1, 0)


def test_assert_no_temporal_leakage_raises_when_test_precedes_train() -> None:
    split = time_based_train_test_split(
        pd.DataFrame(
            {
                "userId": [1, 1, 1],
                "movieId": [10, 11, 12],
                "rating": [4.0, 3.5, 5.0],
                "timestamp": [100, 200, 300],
            }
        ),
        test_interactions_per_user=1,
        min_train_interactions_per_user=2,
    )
    bad_split = type(split)(
        train=split.train.assign(timestamp=[100, 400]),
        test=split.test,
    )

    with pytest.raises(ValueError, match="Temporal leakage"):
        assert_no_temporal_leakage(bad_split)
