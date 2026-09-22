"""General-purpose, source-column-grouped feature generation for model B."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


class FeatureTransformError(ValueError):
    """Raised when a feature transformer is used before fitting."""


@dataclass
class GeneralFeatureBuilder:
    """Build raw, unary, pair-product, and pair-sign features.

    The transformer learns only per-source-column medians during ``fit``.  All
    derived columns are rebuilt from the selected source columns on every
    transform, which makes grouped permutation importance faithful to a
    source-column group.
    """

    source_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        self.source_columns = tuple(self.source_columns)
        if not self.source_columns:
            raise ValueError("at least one source column is required")
        if len(set(self.source_columns)) != len(self.source_columns):
            raise ValueError("source columns must be unique")
        self.medians_: pd.Series | None = None
        self.feature_names_: tuple[str, ...] | None = None

    def _raw_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.source_columns if column not in frame.columns]
        if missing:
            raise FeatureTransformError(f"missing source columns: {', '.join(missing)}")
        numeric = pd.DataFrame(
            {
                column: pd.to_numeric(frame[column], errors="coerce")
                for column in self.source_columns
            },
            index=frame.index,
        )
        if self.medians_ is None:
            raise FeatureTransformError("feature builder must be fit before transform")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.fillna(self.medians_)
        numeric = numeric.fillna(0.0)
        return numeric.astype(np.float64)

    def _make_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for column in self.source_columns:
            names.extend((column, f"abs__{column}", f"square__{column}", f"sign__{column}"))
        for left, right in combinations(self.source_columns, 2):
            names.extend((f"product__{left}__{right}", f"sign_product__{left}__{right}"))
        return tuple(names)

    def fit(self, frame: pd.DataFrame) -> "GeneralFeatureBuilder":
        """Estimate training-only medians and freeze feature names."""

        raw = pd.DataFrame(
            {
                column: pd.to_numeric(frame[column], errors="coerce")
                if column in frame.columns
                else pd.Series(np.nan, index=frame.index)
                for column in self.source_columns
            },
            index=frame.index,
        ).replace([np.inf, -np.inf], np.nan)
        self.medians_ = raw.median(axis=0, skipna=True)
        self.feature_names_ = self._make_feature_names()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform a frame using the medians learned by ``fit``."""

        raw = self._raw_frame(frame)
        values: list[np.ndarray] = []
        for column in self.source_columns:
            array = raw[column].to_numpy(dtype=np.float64)
            values.extend((array, np.abs(array), np.square(array), np.sign(array)))
        for left, right in combinations(self.source_columns, 2):
            left_array = raw[left].to_numpy(dtype=np.float64)
            right_array = raw[right].to_numpy(dtype=np.float64)
            values.extend((left_array * right_array, np.sign(left_array) * np.sign(right_array)))
        matrix = np.column_stack(values).astype(np.float32, copy=False)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=np.finfo(np.float32).max, neginf=-np.finfo(np.float32).max)
        if self.feature_names_ is None or matrix.shape[1] != len(self.feature_names_):
            raise FeatureTransformError("feature name and matrix dimensions diverged")
        return matrix

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Fit on a training frame and transform it."""

        return self.fit(frame).transform(frame)

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.feature_names_ is None:
            raise FeatureTransformError("feature builder must be fit before feature_names is read")
        return self.feature_names_

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def estimated_matrix_bytes(self) -> int:
        """Return bytes per row for the float32 matrix."""

        return self.feature_count * np.dtype(np.float32).itemsize

