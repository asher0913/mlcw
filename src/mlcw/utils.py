"""Shared utilities used throughout the coursework code base."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: Path) -> Path:
    """Creates the directory if necessary and returns it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> None:
    """Persists the provided object as pretty-printed JSON."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2))


def set_global_seed(seed: int) -> None:
    """Initialises both Python's and NumPy's RNGs."""
    random.seed(seed)
    np.random.seed(seed)
