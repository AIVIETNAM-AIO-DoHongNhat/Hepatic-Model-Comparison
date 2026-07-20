"""Shared utilities for the Hepatic Model Comparison project.

Marks ``src`` as a Python package so notebooks can do::

    import sys; sys.path.append("..")   # point at the project root
    from src import data, metrics, stats

Importing the package eagerly loads the three submodules and re-exports the
constants most code needs, so callers can use e.g. ``src.LABELS`` directly.
"""

from . import data, metrics, stats

# Re-export the shared constants defined in data.py for convenience.
from .data import (
    RANDOM_STATE,
    N_FOLDS,
    TARGET_COL,
    LABELS,
    LABEL_MAP,
)

__all__ = [
    "data",
    "metrics",
    "stats",
    "RANDOM_STATE",
    "N_FOLDS",
    "TARGET_COL",
    "LABELS",
    "LABEL_MAP",
]
