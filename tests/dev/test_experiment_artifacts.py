from __future__ import annotations

from pathlib import Path

import pandas as pd

from orbit_predict.data import DatasetPaths
from orbit_predict.experiment import run_experiment


def test_AC13_1_두모델검증과제출_통합산출물생성(circle_frames, small_model_config, tmp_path: Path):
    train, test, sample = circle_frames
    train_path = tmp_path / "Train_DS..csv"
    test_path = tmp_path / "Test_DS..csv"
    sample_path = tmp_path / "Sample_Submission_DS..csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample.to_csv(sample_path, index=False)
    output_dir = tmp_path / "outputs"
    artifact_dir = tmp_path / "artifacts"
    report_path = tmp_path / "model-report.md"

    result = run_experiment(
        DatasetPaths(train_path, test_path, sample_path),
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        report_path=report_path,
        model_b_config=small_model_config,
        strict_row_counts=False,
    )

    assert {stage["stage"] for stage in result.metrics} == {
        "baseline_development_mean",
        "model_a",
        "model_b_all_inputs",
        "model_b_best_reduced",
        "model_b_final",
    }
    for name in ("submission_model_a.csv", "submission_model_b.csv"):
        submission = pd.read_csv(output_dir / name)
        assert tuple(submission.columns) == ("Satellite_ID", "Y_Position")
        assert len(submission) == len(sample)
    assert report_path.is_file()
    assert "Kaggle" in report_path.read_text(encoding="utf-8")
    assert (artifact_dir / "feature_selection.json").is_file()


def test_AC13_2_사용자제출전_공개점수미확인보고서(circle_frames, small_model_config, tmp_path: Path):
    train, test, sample = circle_frames
    paths = []
    for name, frame in (
        ("train.csv", train),
        ("test.csv", test),
        ("sample.csv", sample),
    ):
        path = tmp_path / name
        frame.to_csv(path, index=False)
        paths.append(path)
    report_path = tmp_path / "docs" / "report.md"
    run_experiment(
        DatasetPaths(*paths),
        output_dir=tmp_path / "out",
        artifact_dir=tmp_path / "artifacts",
        report_path=report_path,
        model_b_config=small_model_config,
        strict_row_counts=False,
    )
    report = report_path.read_text(encoding="utf-8")
    assert "실제 Kaggle 점수" in report
    assert "점수와 순위를 아직 기재하지 않는다" in report
    # EC03: 내부 비교만으로 어느 한 모델을 시험 정답이라고 단정하지 않는다.
    assert "어느 한 모델을 시험 정답이라고 자동 단정하지 않는다" in report
    # AC13.3: 사용자가 두 모델 차이, 피처 선택, 제출 절차를 읽을 수 있다.
    assert "모델 A" in report and "모델 B" in report
    assert "피처 선택" in report
    assert "Submit Predictions" in report
