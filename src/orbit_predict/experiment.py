"""End-to-end training, validation, and submission generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    EXPECTED_SAMPLE_ROWS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    INPUT_COLUMNS,
    TARGET_COLUMN,
)
from .data import DatasetPaths, compare_input_distributions, load_datasets, make_submission, validate_submission
from .metrics import metric_dict
from .model_a import fit_circle, predict_circle
from .model_b import FeatureSelectionResult, ModelB, ModelBConfig, select_features
from .reporting import write_json, write_markdown_report
from .split import DataSplit, make_split


@dataclass
class ExperimentResult:
    """Serializable summary of one complete experiment."""

    train_rows: int
    test_rows: int
    sample_rows: int
    input_columns: tuple[str, ...]
    development_rows: int
    comparison_rows: int
    split: DataSplit
    model_b_config: dict[str, int | float]
    model_a_parameters: dict[str, float]
    model_a_development_exception_count: int
    model_a_test_exception_count: int
    model_a_development_exceptions: tuple[dict[str, object], ...]
    model_a_test_exceptions: tuple[dict[str, object], ...]
    metrics: list[dict[str, Any]]
    feature_selection: dict[str, Any]
    distributions: dict[str, Any]
    error_segments: dict[str, Any]
    outputs: dict[str, str]


def _segment_metrics(
    frame: pd.DataFrame,
    target: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    """Report X-range deciles and sign branches without choosing a model."""

    work = pd.DataFrame(
        {
            "x": pd.to_numeric(frame["X_Position"], errors="coerce").to_numpy(dtype=np.float64),
            "branch_product": (
                pd.to_numeric(frame["Velocity"], errors="coerce").to_numpy(dtype=np.float64)
                * pd.to_numeric(frame["Altitude"], errors="coerce").to_numpy(dtype=np.float64)
            ),
            "target": target,
            "prediction": predictions,
        }
    )
    work["branch"] = np.select(
        [work["branch_product"] > 0, work["branch_product"] < 0],
        ["positive", "negative"],
        default="zero",
    )
    result: dict[str, Any] = {"branch": {}, "x_decile": {}}
    for name, grouped in (
        ("branch", work.groupby("branch", sort=True)),
        ("x_decile", work.assign(x_decile=pd.qcut(work["x"], q=10, duplicates="drop")).groupby("x_decile", observed=True)),
    ):
        for key, group in grouped:
            if len(group) == 0:
                continue
            values = metric_dict(group["target"].to_numpy(), group["prediction"].to_numpy())
            result[name][str(key)] = {"rows": int(len(group)), **values}
    return result


def _stage(stage: str, metrics: dict[str, float], note: str = "") -> dict[str, Any]:
    return {"stage": stage, **metrics, "note": note}


def _write_submission(path: Path, sample_submission: pd.DataFrame, predictions: np.ndarray) -> None:
    submission = make_submission(sample_submission, predictions)
    validate_submission(submission, sample_submission)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)


def run_experiment(
    paths: DatasetPaths,
    *,
    output_dir: Path = Path("outputs"),
    artifact_dir: Path = Path("artifacts"),
    report_path: Path = Path("docs/model-report.md"),
    model_b_config: ModelBConfig = ModelBConfig(),
    strict_row_counts: bool = True,
) -> ExperimentResult:
    """Execute the full PRD workflow and write both submission artifacts."""

    datasets = load_datasets(
        paths,
        expected_train_rows=EXPECTED_TRAIN_ROWS if strict_row_counts else None,
        expected_test_rows=EXPECTED_TEST_ROWS if strict_row_counts else None,
        expected_sample_rows=EXPECTED_SAMPLE_ROWS if strict_row_counts else None,
    )
    train = datasets.train
    test = datasets.test
    sample_submission = datasets.sample_submission
    target = train[TARGET_COLUMN].to_numpy(dtype=np.float64)
    split = make_split(len(train))
    development = train.iloc[split.development_indices].reset_index(drop=True)
    comparison = train.iloc[split.comparison_indices].reset_index(drop=True)
    development_target = development[TARGET_COLUMN].to_numpy(dtype=np.float64)
    comparison_target = comparison[TARGET_COLUMN].to_numpy(dtype=np.float64)
    development_inputs = development.loc[:, INPUT_COLUMNS]
    comparison_inputs = comparison.loc[:, INPUT_COLUMNS]
    test_inputs = test.loc[:, INPUT_COLUMNS]

    distributions = compare_input_distributions(train, test)

    baseline_prediction = np.full(len(comparison), float(np.mean(development_target)), dtype=np.float64)
    metrics: list[dict[str, Any]] = [
        _stage("baseline_development_mean", metric_dict(comparison_target, baseline_prediction))
    ]

    circle_parameters = fit_circle(development)
    model_a_development = predict_circle(comparison, circle_parameters)
    metrics.append(_stage("model_a", metric_dict(comparison_target, model_a_development.predictions)))

    feature_selection_result: FeatureSelectionResult = select_features(
        development_inputs,
        development_target,
        config=model_b_config,
        source_columns=INPUT_COLUMNS,
    )
    model_b_all = ModelB(tuple(INPUT_COLUMNS), model_b_config).fit(development_inputs, development_target)
    model_b_all_comparison = model_b_all.predict(comparison_inputs)
    metrics.append(
        _stage(
            "model_b_all_inputs",
            metric_dict(comparison_target, model_b_all_comparison),
            note="전체 일곱 원본 입력 기준",
        )
    )

    model_b_reduced = ModelB(feature_selection_result.best_reduced_columns, model_b_config).fit(
        development_inputs,
        development_target,
    )
    model_b_reduced_comparison = model_b_reduced.predict(comparison_inputs)
    metrics.append(
        _stage(
            "model_b_best_reduced",
            metric_dict(comparison_target, model_b_reduced_comparison),
            note=f"최고 축소 후보 k={feature_selection_result.best_reduced_k}",
        )
    )

    model_b_final_development = ModelB(feature_selection_result.selected_columns, model_b_config).fit(
        development_inputs,
        development_target,
    )
    model_b_final_comparison = model_b_final_development.predict(comparison_inputs)
    final_note = (
        "전체 입력과 동일한 최종 선택"
        if feature_selection_result.selected_k == len(INPUT_COLUMNS)
        else f"교차검증 선택 k={feature_selection_result.selected_k}"
    )
    metrics.append(
        _stage(
            "model_b_final",
            metric_dict(comparison_target, model_b_final_comparison),
            note=final_note,
        )
    )

    model_a_full_parameters = fit_circle(train)
    model_a_test = predict_circle(test, model_a_full_parameters)
    model_b_full = ModelB(feature_selection_result.selected_columns, model_b_config).fit(
        train.loc[:, INPUT_COLUMNS],
        target,
    )
    model_b_test = model_b_full.predict(test_inputs)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_a_submission_path = output_dir / "submission_model_a.csv"
    model_b_submission_path = output_dir / "submission_model_b.csv"
    _write_submission(model_a_submission_path, sample_submission, model_a_test.predictions)
    _write_submission(model_b_submission_path, sample_submission, model_b_test)

    error_segments = {
        "model_a": _segment_metrics(comparison, comparison_target, model_a_development.predictions),
        "model_b_final": _segment_metrics(comparison, comparison_target, model_b_final_comparison),
    }
    result = ExperimentResult(
        train_rows=len(train),
        test_rows=len(test),
        sample_rows=len(sample_submission),
        input_columns=tuple(INPUT_COLUMNS),
        development_rows=len(development),
        comparison_rows=len(comparison),
        split=split,
        model_b_config=model_b_config.as_dict(),
        model_a_parameters=model_a_full_parameters.as_dict(),
        model_a_development_exception_count=len(model_a_development.exceptions),
        model_a_test_exception_count=len(model_a_test.exceptions),
        model_a_development_exceptions=model_a_development.exceptions,
        model_a_test_exceptions=model_a_test.exceptions,
        metrics=metrics,
        feature_selection=feature_selection_result.as_dict(),
        distributions=distributions,
        error_segments=error_segments,
        outputs={
            "model_a_submission": str(model_a_submission_path),
            "model_b_submission": str(model_b_submission_path),
            "model_a_exceptions": str(artifact_dir / "model_a_exceptions.json"),
            "feature_selection": str(artifact_dir / "feature_selection.json"),
            "data_distribution": str(artifact_dir / "data_distribution.json"),
            "error_segments": str(artifact_dir / "error_segments.json"),
        },
    )
    write_json(artifact_dir / "feature_selection.json", feature_selection_result.as_dict())
    write_json(artifact_dir / "data_distribution.json", distributions)
    write_json(artifact_dir / "error_segments.json", error_segments)
    write_json(
        artifact_dir / "model_a_exceptions.json",
        {
            "comparison": list(model_a_development.exceptions),
            "test": list(model_a_test.exceptions),
        },
    )
    write_json(
        artifact_dir / "experiment.json",
        {
            "train_rows": result.train_rows,
            "test_rows": result.test_rows,
            "sample_rows": result.sample_rows,
            "development_rows": result.development_rows,
            "comparison_rows": result.comparison_rows,
            "split": {
                "seed": split.seed,
                "comparison_fraction": split.comparison_fraction,
                "development_count": len(split.development_indices),
                "comparison_count": len(split.comparison_indices),
            },
            "model_b_config": result.model_b_config,
            "model_a_parameters": result.model_a_parameters,
            "model_a_exceptions": {
                "comparison": list(result.model_a_development_exceptions),
                "test": list(result.model_a_test_exceptions),
            },
            "metrics": result.metrics,
            "outputs": result.outputs,
        },
    )
    write_markdown_report(report_path, result)
    return result
