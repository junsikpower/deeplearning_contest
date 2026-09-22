"""Input, schema, distribution, and submission-contract helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import (
    EXPECTED_SAMPLE_ROWS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    ID_COLUMN,
    INPUT_COLUMNS,
    SUBMISSION_COLUMNS,
    TARGET_COLUMN,
)


class DataContractError(ValueError):
    """Raised when a CSV violates the project data contract."""


@dataclass(frozen=True)
class DatasetPaths:
    """Locations of the three supplied CSV files."""

    train: Path
    test: Path
    sample_submission: Path


@dataclass(frozen=True)
class LoadedDatasets:
    """Loaded and contract-checked input frames."""

    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


def _require_columns(frame: pd.DataFrame, expected: Iterable[str], label: str) -> None:
    expected_tuple = tuple(expected)
    actual_tuple = tuple(frame.columns)
    if actual_tuple != expected_tuple:
        raise DataContractError(
            f"{label} columns must be {expected_tuple}; received {actual_tuple}"
        )


def _require_numeric(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = numeric.isna() & frame[column].notna()
        if invalid.any():
            row = int(np.flatnonzero(invalid.to_numpy())[0])
            raise DataContractError(
                f"{label}.{column} contains a non-numeric value at row {row}"
            )
        if not np.isfinite(numeric.fillna(0).to_numpy(dtype=np.float64)).all():
            raise DataContractError(f"{label}.{column} contains an infinite value")


def _require_unique_ids(frame: pd.DataFrame, label: str) -> None:
    ids = frame[ID_COLUMN]
    if ids.isna().any():
        raise DataContractError(f"{label}.{ID_COLUMN} contains a missing ID")
    numeric_ids = pd.to_numeric(ids, errors="coerce")
    if numeric_ids.isna().any() or not np.isfinite(numeric_ids.to_numpy(dtype=np.float64)).all():
        raise DataContractError(f"{label}.{ID_COLUMN} must contain finite integer IDs")
    if not np.equal(numeric_ids.to_numpy(dtype=np.float64), np.floor(numeric_ids)).all():
        raise DataContractError(f"{label}.{ID_COLUMN} must contain integer IDs")
    if ids.duplicated().any():
        duplicate = ids[ids.duplicated()].iloc[0]
        raise DataContractError(f"{label} contains duplicate {ID_COLUMN}: {duplicate}")


def validate_train_frame(
    frame: pd.DataFrame,
    *,
    expected_rows: int | None = EXPECTED_TRAIN_ROWS,
) -> None:
    """Validate the training schema and values."""

    _require_columns(frame, (ID_COLUMN, *INPUT_COLUMNS, TARGET_COLUMN), "train")
    if expected_rows is not None and len(frame) != expected_rows:
        raise DataContractError(f"train must contain {expected_rows} rows; received {len(frame)}")
    _require_unique_ids(frame, "train")
    _require_numeric(frame, (*INPUT_COLUMNS, TARGET_COLUMN), "train")
    if frame[[*INPUT_COLUMNS, TARGET_COLUMN]].isna().any().any():
        raise DataContractError("train contains missing numeric values")


def validate_test_frame(
    frame: pd.DataFrame,
    *,
    expected_rows: int | None = EXPECTED_TEST_ROWS,
) -> None:
    """Validate the test schema and values."""

    _require_columns(frame, (ID_COLUMN, *INPUT_COLUMNS), "test")
    if expected_rows is not None and len(frame) != expected_rows:
        raise DataContractError(f"test must contain {expected_rows} rows; received {len(frame)}")
    _require_unique_ids(frame, "test")
    _require_numeric(frame, INPUT_COLUMNS, "test")


def validate_sample_submission_frame(
    frame: pd.DataFrame,
    *,
    expected_rows: int | None = EXPECTED_SAMPLE_ROWS,
) -> None:
    """Validate the sample submission schema."""

    _require_columns(frame, SUBMISSION_COLUMNS, "sample_submission")
    if expected_rows is not None and len(frame) != expected_rows:
        raise DataContractError(
            f"sample_submission must contain {expected_rows} rows; received {len(frame)}"
        )
    _require_unique_ids(frame, "sample_submission")
    _require_numeric(frame, (TARGET_COLUMN,), "sample_submission")


def validate_test_sample_alignment(
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    """Ensure the sample submission refers to the test rows in the same order."""

    if len(test) != len(sample_submission):
        raise DataContractError("test and sample submission row counts do not match")
    test_ids = pd.to_numeric(test[ID_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    sample_ids = pd.to_numeric(sample_submission[ID_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    if not np.array_equal(test_ids, sample_ids):
        raise DataContractError("test IDs and sample submission IDs are not aligned")


def load_datasets(
    paths: DatasetPaths,
    *,
    expected_train_rows: int | None = EXPECTED_TRAIN_ROWS,
    expected_test_rows: int | None = EXPECTED_TEST_ROWS,
    expected_sample_rows: int | None = EXPECTED_SAMPLE_ROWS,
) -> LoadedDatasets:
    """Read all supplied CSVs and validate them before modelling."""

    for path in (paths.train, paths.test, paths.sample_submission):
        if not path.is_file():
            raise FileNotFoundError(f"input CSV does not exist: {path}")
    train = pd.read_csv(paths.train)
    test = pd.read_csv(paths.test)
    sample_submission = pd.read_csv(paths.sample_submission)
    validate_train_frame(train, expected_rows=expected_train_rows)
    validate_test_frame(test, expected_rows=expected_test_rows)
    validate_sample_submission_frame(sample_submission, expected_rows=expected_sample_rows)
    validate_test_sample_alignment(test, sample_submission)
    return LoadedDatasets(train=train, test=test, sample_submission=sample_submission)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def summarize_distribution(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, dict[str, float | int]]:
    """Return reproducible range and quantile summaries for numeric columns."""

    summary: dict[str, dict[str, float | int]] = {}
    for column in columns:
        values = _numeric_series(frame, column)
        finite = values[np.isfinite(values.to_numpy(dtype=np.float64, na_value=np.nan))].dropna()
        if finite.empty:
            summary[column] = {
                "count": 0,
                "missing": int(values.isna().sum()),
                "min": float("nan"),
                "q01": float("nan"),
                "q25": float("nan"),
                "q50": float("nan"),
                "q75": float("nan"),
                "q99": float("nan"),
                "max": float("nan"),
            }
            continue
        quantiles = finite.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
        summary[column] = {
            "count": int(finite.size),
            "missing": int(values.isna().sum()),
            "min": float(finite.min()),
            "q01": float(quantiles.loc[0.01]),
            "q25": float(quantiles.loc[0.25]),
            "q50": float(quantiles.loc[0.50]),
            "q75": float(quantiles.loc[0.75]),
            "q99": float(quantiles.loc[0.99]),
            "max": float(finite.max()),
        }
    return summary


def compare_input_distributions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Summarize train/test input distributions side by side."""

    return {
        "train": summarize_distribution(train, INPUT_COLUMNS),
        "test": summarize_distribution(test, INPUT_COLUMNS),
    }


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    """Ensure a generated submission matches the sample ID and row contract."""

    validate_sample_submission_frame(submission, expected_rows=len(sample_submission))
    if not submission[ID_COLUMN].reset_index(drop=True).equals(
        sample_submission[ID_COLUMN].reset_index(drop=True)
    ):
        raise DataContractError("submission IDs or row order do not match the sample submission")
    predictions = pd.to_numeric(submission[TARGET_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(predictions).all():
        raise DataContractError("submission predictions must all be finite numeric values")


def make_submission(
    sample_submission: pd.DataFrame,
    predictions: object,
) -> pd.DataFrame:
    """Create and validate an ID-aligned submission frame."""

    values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if len(values) != len(sample_submission):
        raise DataContractError(
            f"prediction count {len(values)} does not match sample rows {len(sample_submission)}"
        )
    submission = pd.DataFrame(
        {
            ID_COLUMN: sample_submission[ID_COLUMN].to_numpy(copy=True),
            TARGET_COLUMN: values,
        }
    )
    validate_submission(submission, sample_submission)
    return submission
