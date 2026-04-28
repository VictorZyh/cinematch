"""Leakage-safe train/test splitting for recommendation data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cinematch.constants import ITEM_ID, TIMESTAMP, USER_ID


@dataclass(frozen=True)
class TrainTestSplit:
    """Container for timestamp-aware recommendation splits."""

    train: pd.DataFrame
    test: pd.DataFrame


def _validate_split_parameters(
    test_interactions_per_user: int,
    min_train_interactions_per_user: int,
) -> None:
    """Validate split parameters before touching the input data."""

    if test_interactions_per_user <= 0:
        raise ValueError("test_interactions_per_user must be positive.")
    if min_train_interactions_per_user <= 0:
        raise ValueError("min_train_interactions_per_user must be positive.")


def time_based_train_test_split(
    ratings: pd.DataFrame,
    test_interactions_per_user: int,
    min_train_interactions_per_user: int,
) -> TrainTestSplit:
    """Split user interactions into train and test sets by timestamp.

    For each user, the latest ``test_interactions_per_user`` interactions are
    assigned to test and earlier interactions are assigned to train. Users that
    do not have enough history for both train and test are excluded. This
    mirrors offline recommender evaluation where the model must predict future
    interactions from past behavior only.

    Parameters
    ----------
    ratings:
        Cleaned ratings table containing at least user and timestamp columns.
    test_interactions_per_user:
        Number of most recent interactions per user to reserve for test.
    min_train_interactions_per_user:
        Minimum number of historical train interactions required for a user to
        remain in the split.

    Returns
    -------
    TrainTestSplit
        Train and test DataFrames sorted by user and timestamp.
    """

    _validate_split_parameters(
        test_interactions_per_user=test_interactions_per_user,
        min_train_interactions_per_user=min_train_interactions_per_user,
    )

    min_total_interactions = test_interactions_per_user + min_train_interactions_per_user
    interaction_counts = ratings.groupby(USER_ID).size()
    eligible_users = interaction_counts[
        interaction_counts >= min_total_interactions
    ].index

    eligible_ratings = ratings.loc[ratings[USER_ID].isin(eligible_users)].copy()
    if eligible_ratings.empty:
        empty = ratings.iloc[0:0].copy()
        return TrainTestSplit(train=empty, test=empty)

    sorted_ratings = eligible_ratings.sort_values(
        [USER_ID, TIMESTAMP, ITEM_ID]
    ).reset_index(drop=True)
    reverse_position = sorted_ratings.groupby(USER_ID).cumcount(ascending=False)
    is_test = reverse_position < test_interactions_per_user

    train = sorted_ratings.loc[~is_test].reset_index(drop=True)
    test = sorted_ratings.loc[is_test].reset_index(drop=True)
    return TrainTestSplit(train=train, test=test)


def assert_no_temporal_leakage(split: TrainTestSplit) -> None:
    """Raise an error if any user's test data occurs before train data."""

    max_train_timestamp = split.train.groupby(USER_ID)[TIMESTAMP].max()
    min_test_timestamp = split.test.groupby(USER_ID)[TIMESTAMP].min()
    common_users = max_train_timestamp.index.intersection(min_test_timestamp.index)

    leaking_users = [
        user_id
        for user_id in common_users
        if min_test_timestamp.loc[user_id] < max_train_timestamp.loc[user_id]
    ]
    if leaking_users:
        raise ValueError(f"Temporal leakage detected for users: {leaking_users}")
