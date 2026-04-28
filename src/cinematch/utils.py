"""Small reusable utilities for reproducible pipeline execution."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def set_random_seed(seed: int) -> None:
    """Set random seeds used by the standard library and NumPy."""

    random.seed(seed)
    np.random.seed(seed)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(payload: Mapping[str, Any], path: str | Path) -> None:
    """Persist a JSON-serializable mapping with deterministic formatting."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
