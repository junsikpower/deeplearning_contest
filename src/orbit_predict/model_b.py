"""Model B: formula-independent tabular machine learning and feature selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .constants import CV_FOLDS, INPUT_COLUMNS, RANDOM_SEED, SELECTION_SAMPLE_SIZE
from .features import GeneralFeatureBuilder
from .metrics import metric_dict


@dataclass(frozen=True)
class ModelBConfig:
    """Resource-bounded deterministic estimator settings."""

    max_iter: int = 120
    learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    random_state: int = RANDOM_SEED
    selection_sample_size: int = SELECTION_SAMPLE_SIZE
    cv_folds: int = CV_FOLDS

    def as_dict(self) -> dict[str, int | float]:
        return {key: value for key, value in asdict(self).items()}


def build_estimator(config: ModelBConfig) -> HistGradientBoostingRegressor:
    """Create the only estimator used by model B."""

    return HistGradientBoostingRegressor(
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        early_stopping=False,
        random_state=config.random_state,
    )


@dataclass
class ModelB:
    """A fitted model B with its training-only feature transformer."""

    source_columns: tuple[str, ...]
    config: ModelBConfig
    estimator_factory: Callable[[ModelBConfig], object] = build_estimator

    def __post_init__(self) -> None:
        self.source_columns = _validate_source_columns(self.source_columns)
        self.feature_builder: GeneralFeatureBuilder | None = None
        self.estimator: object | None = None

    def fit(self, frame: pd.DataFrame, target: object) -> "ModelB":
        """Fit preprocessing and estimator using this training fold only."""

        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(frame) != len(y):
            raise ValueError("frame and target lengths must match")
        if len(y) == 0:
            raise ValueError("model B requires at least one training row")
        if not np.isfinite(y).all():
            raise ValueError("model B target must be finite")
        builder = GeneralFeatureBuilder(self.source_columns)
        matrix = builder.fit_transform(frame)
        estimator = self.estimator_factory(self.config)
        estimator.fit(matrix, y)
        self.feature_builder = builder
        self.estimator = estimator
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.feature_builder is None:
            raise RuntimeError("model B is not fitted")
        return self.feature_builder.transform(frame)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("model B is not fitted")
        predictions = np.asarray(self.estimator.predict(self.transform(frame)), dtype=np.float64)
        if not np.isfinite(predictions).all():
            raise ValueError("model B generated a non-finite prediction")
        return predictions

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.feature_builder is None:
            raise RuntimeError("model B is not fitted")
        return self.feature_builder.feature_names


@dataclass(frozen=True)
class CrossValidationResult:
    """Fold metrics for one source-column candidate."""

    source_columns: tuple[str, ...]
    folds: tuple[dict[str, float], ...]
    mean_rmse: float
    mean_huber_score: float
    feature_count: int
    estimated_matrix_bytes_per_row: int
    estimated_matrix_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "source_columns": list(self.source_columns),
            "folds": [dict(fold) for fold in self.folds],
            "mean_rmse": self.mean_rmse,
            "mean_huber_score": self.mean_huber_score,
            "feature_count": self.feature_count,
            "estimated_matrix_bytes_per_row": self.estimated_matrix_bytes_per_row,
            "estimated_matrix_bytes": self.estimated_matrix_bytes,
        }


@dataclass(frozen=True)
class FeatureSelectionResult:
    """All data-driven selection decisions needed by the report."""

    sample_size: int
    importance: tuple[dict[str, float], ...]
    ranked_columns: tuple[str, ...]
    candidates: tuple[CrossValidationResult, ...]
    selected_k: int
    selected_columns: tuple[str, ...]
    best_reduced_k: int
    best_reduced_columns: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_size": self.sample_size,
            "importance": [dict(item) for item in self.importance],
            "ranked_columns": list(self.ranked_columns),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "selected_k": self.selected_k,
            "selected_columns": list(self.selected_columns),
            "best_reduced_k": self.best_reduced_k,
            "best_reduced_columns": list(self.best_reduced_columns),
        }


def _validate_source_columns(source_columns: Iterable[str]) -> tuple[str, ...]:
    columns = tuple(source_columns)
    unknown = [column for column in columns if column not in INPUT_COLUMNS]
    if unknown:
        raise ValueError(f"model B source columns are not approved inputs: {unknown}")
    if len(set(columns)) != len(columns):
        raise ValueError("model B source columns must be unique")
    if not columns:
        raise ValueError("model B requires at least one source column")
    return columns


def _make_folds(row_count: int, config: ModelBConfig) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if row_count < config.cv_folds:
        raise ValueError(f"at least {config.cv_folds} rows are required for cross-validation")
    splitter = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    placeholder = np.empty(row_count, dtype=np.float32)
    return tuple((train, validation) for train, validation in splitter.split(placeholder))


def cross_validate_candidate(
    frame: pd.DataFrame,
    target: object,
    source_columns: Sequence[str],
    *,
    config: ModelBConfig,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
) -> CrossValidationResult:
    """Fit one candidate in every fixed development-fold and average metrics."""

    columns = _validate_source_columns(source_columns)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if len(frame) != len(y):
        raise ValueError("frame and target lengths must match")
    actual_folds = folds if folds is not None else _make_folds(len(frame), config)
    fold_metrics: list[dict[str, float]] = []
    feature_count = len(columns) * 4 + (len(columns) * (len(columns) - 1))
    for fold_index, (train_indices, validation_indices) in enumerate(actual_folds):
        model_config = ModelBConfig(**{**config.as_dict(), "random_state": config.random_state + fold_index})
        model = ModelB(columns, model_config)
        model.fit(frame.iloc[train_indices], y[train_indices])
        predictions = model.predict(frame.iloc[validation_indices])
        fold_metrics.append(metric_dict(y[validation_indices], predictions))
    mean_rmse = float(np.mean([fold["rmse"] for fold in fold_metrics]))
    mean_huber = float(np.mean([fold["huber_score"] for fold in fold_metrics]))
    return CrossValidationResult(
        source_columns=columns,
        folds=tuple(fold_metrics),
        mean_rmse=mean_rmse,
        mean_huber_score=mean_huber,
        feature_count=feature_count,
        estimated_matrix_bytes_per_row=feature_count * np.dtype(np.float32).itemsize,
        estimated_matrix_bytes=feature_count * np.dtype(np.float32).itemsize * len(frame),
    )


def grouped_permutation_importance(
    frame: pd.DataFrame,
    target: object,
    source_columns: Sequence[str],
    *,
    config: ModelBConfig,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
) -> tuple[dict[str, float], ...]:
    """Measure each source column by permuting it and all derived features."""

    columns = _validate_source_columns(source_columns)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    actual_folds = folds if folds is not None else _make_folds(len(frame), config)
    effects: dict[str, list[float]] = {column: [] for column in columns}
    for fold_index, (train_indices, validation_indices) in enumerate(actual_folds):
        model_config = ModelBConfig(**{**config.as_dict(), "random_state": config.random_state + fold_index})
        model = ModelB(columns, model_config)
        train_frame = frame.iloc[train_indices]
        validation_frame = frame.iloc[validation_indices]
        model.fit(train_frame, y[train_indices])
        baseline_predictions = model.predict(validation_frame)
        baseline_huber = metric_dict(y[validation_indices], baseline_predictions)["huber_score"]
        for column_index, column in enumerate(columns):
            permutation_rng = np.random.default_rng(
                config.random_state + fold_index * 1_000 + column_index
            )
            permuted = validation_frame.copy()
            permuted[column] = permuted[column].to_numpy(copy=True)[permutation_rng.permutation(len(permuted))]
            permuted_predictions = model.predict(permuted)
            permuted_huber = metric_dict(y[validation_indices], permuted_predictions)["huber_score"]
            effects[column].append(float(permuted_huber - baseline_huber))
    return tuple(
        {
            "source_column": column,
            "mean_huber_increase": float(np.mean(effects[column])),
            "fold_huber_increases": [float(value) for value in effects[column]],
        }
        for column in columns
    )


def select_features(
    development_frame: pd.DataFrame,
    development_target: object,
    *,
    config: ModelBConfig = ModelBConfig(),
    source_columns: Sequence[str] = INPUT_COLUMNS,
) -> FeatureSelectionResult:
    """Run grouped importance and k=1..7 candidate selection on development data."""

    columns = _validate_source_columns(source_columns)
    target = np.asarray(development_target, dtype=np.float64).reshape(-1)
    if len(development_frame) != len(target):
        raise ValueError("development frame and target lengths must match")
    sample_size = min(len(development_frame), config.selection_sample_size)
    if sample_size < config.cv_folds:
        raise ValueError("selection sample is too small for the configured cross-validation")
    sample_rng = np.random.default_rng(config.random_state)
    if sample_size == len(development_frame):
        sample_indices = np.arange(len(development_frame))
    else:
        sample_indices = np.sort(sample_rng.choice(len(development_frame), size=sample_size, replace=False))
    sampled_frame = development_frame.iloc[sample_indices].reset_index(drop=True)
    sampled_target = target[sample_indices]
    folds = _make_folds(len(sampled_frame), config)

    importance = grouped_permutation_importance(
        sampled_frame,
        sampled_target,
        columns,
        config=config,
        folds=folds,
    )
    original_order = {column: position for position, column in enumerate(columns)}
    ranked_columns = tuple(
        sorted(
            columns,
            key=lambda column: (
                -next(item["mean_huber_increase"] for item in importance if item["source_column"] == column),
                original_order[column],
            ),
        )
    )

    candidates: list[CrossValidationResult] = []
    for k in range(1, len(ranked_columns) + 1):
        candidate_columns = ranked_columns[:k]
        candidates.append(
            cross_validate_candidate(
                sampled_frame,
                sampled_target,
                candidate_columns,
                config=config,
                folds=folds,
            )
        )
    best_candidate = min(candidates, key=lambda candidate: (candidate.mean_huber_score, len(candidate.source_columns)))
    reduced_candidates = [candidate for candidate in candidates if len(candidate.source_columns) < len(ranked_columns)]
    best_reduced = min(
        reduced_candidates or candidates,
        key=lambda candidate: (candidate.mean_huber_score, len(candidate.source_columns)),
    )
    return FeatureSelectionResult(
        sample_size=sample_size,
        importance=tuple(importance),
        ranked_columns=ranked_columns,
        candidates=tuple(candidates),
        selected_k=len(best_candidate.source_columns),
        selected_columns=best_candidate.source_columns,
        best_reduced_k=len(best_reduced.source_columns),
        best_reduced_columns=best_reduced.source_columns,
    )
