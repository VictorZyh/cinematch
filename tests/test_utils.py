"""Tests for utility helpers."""

from __future__ import annotations

import json

import numpy as np

from cinematch.utils import ensure_directory, save_json, set_random_seed


def test_set_random_seed_makes_numpy_reproducible() -> None:
    """Setting the same seed twice should reproduce the same NumPy draws."""

    set_random_seed(7)
    first_draw = np.random.random(3)
    set_random_seed(7)
    second_draw = np.random.random(3)

    assert np.array_equal(first_draw, second_draw)


def test_save_json_creates_parent_directory(tmp_path) -> None:
    """JSON saving should create missing parent directories."""

    output_path = tmp_path / "nested" / "payload.json"
    save_json({"b": 2, "a": 1}, output_path)

    assert ensure_directory(tmp_path / "nested").exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
