"""Deterministic development/comparison split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import COMPARISON_FRACTION, RANDOM_SEED


@dataclass(frozen=True)
class DataSplit:
    """Row positions for the development and final comparison partitions."""

    development_indices: np.ndarray
    comparison_indices: np.ndarray
    seed: int
    comparison_fraction: float


def make_split(
    row_count: int,
    *,
    seed: int = RANDOM_SEED,
    comparison_fraction: float = COMPARISON_FRACTION,
) -> DataSplit:
    """Apply the PRD's ``default_rng(42).permutation`` split exactly."""

    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if not 0 < comparison_fraction < 1:
        raise ValueError("comparison_fraction must be between 0 and 1")
    permutation = np.random.default_rng(seed).permutation(row_count)
    comparison_count = int(np.ceil(comparison_fraction * row_count))
    comparison_indices = permutation[:comparison_count]
    development_indices = permutation[comparison_count:]
    return DataSplit(
        development_indices=development_indices,
        comparison_indices=comparison_indices,
        seed=seed,
        comparison_fraction=comparison_fraction,
    )

