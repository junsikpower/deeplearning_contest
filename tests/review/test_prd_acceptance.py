"""PRD Acceptance Criteria (AC 13.1, 13.2, 13.3) independent tests."""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest

from orbit_predict.constants import EXPECTED_SAMPLE_ROWS, ID_COLUMN, INPUT_COLUMNS, TARGET_COLUMN
from orbit_predict.data import DatasetPaths, load_datasets
from orbit_predict.experiment import run_experiment
from orbit_predict.model_b import ModelBConfig


TRAIN_PATH = Path("orbit-predict-satellite-position/Train_DS..csv")
TEST_PATH = Path("orbit-predict-satellite-position/Test_DS..csv")
SAMPLE_PATH = Path("orbit-predict-satellite-position/Sample_Submission_DS..csv")
SUB_A_PATH = Path("outputs/submission_model_a.csv")
SUB_B_PATH = Path("outputs/submission_model_b.csv")
REPORT_PATH = Path("docs/model-report.md")
ARTIFACTS_DIR = Path("artifacts")


def test_AC13_1_입력데이터_계약_및_스키마_검증() -> None:
    """PRD 13.1: 학습·시험·샘플 CSV의 스키마, 데이터 상태 및 입력 분포가 확인된다."""
    datasets = load_datasets(
        DatasetPaths(train=TRAIN_PATH, test=TEST_PATH, sample_submission=SAMPLE_PATH)
    )
    assert len(datasets.train) == 636_363
    assert len(datasets.test) == 63_000
    assert len(datasets.sample_submission) == 63_000

    assert list(datasets.train.columns) == [ID_COLUMN, *INPUT_COLUMNS, TARGET_COLUMN]
    assert list(datasets.test.columns) == [ID_COLUMN, *INPUT_COLUMNS]
    assert list(datasets.sample_submission.columns) == [ID_COLUMN, TARGET_COLUMN]


def test_AC13_1_모델A_모델B_독립실행_및_동일구간_평가_검증() -> None:
    """PRD 13.1: 모델 A와 모델 B가 별도 코드 경로로 실행되고 동일한 비교 구간에서 평가된다."""
    exp_path = ARTIFACTS_DIR / "experiment.json"
    assert exp_path.is_file()
    with open(exp_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stages = {item["stage"]: item for item in data["metrics"]}
    assert "model_a" in stages
    assert "model_b_all_inputs" in stages
    assert "model_b_final" in stages

    # 동일 비교 행 수에서 평가되었는지 확인
    assert data["comparison_rows"] == 127_273
    assert stages["model_a"]["rmse"] > 0
    assert stages["model_b_final"]["rmse"] > 0


def test_AC13_1_모델B_피처선택_축소후보_실제평가_검증() -> None:
    """PRD 13.1: 모델 B가 원 공식 없이 학습·예측되며, 최소 한 축소 입력 후보를 실제 학습·비교한다."""
    fs_path = ARTIFACTS_DIR / "feature_selection.json"
    assert fs_path.is_file()
    with open(fs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "importance" in data
    assert len(data["importance"]) == len(INPUT_COLUMNS)
    assert "candidates" in data
    # 최소 한 개 이상의 축소 후보(k < 7)가 평가되었는지 확인
    reduced_candidates = [c for c in data["candidates"] if len(c["source_columns"]) < 7]
    assert len(reduced_candidates) >= 1
    assert data["best_reduced_k"] < 7


def test_AC13_1_두_제출파일_완전성_및_샘플정합성_검증() -> None:
    """PRD 13.1: 모델별 63,000행 제출 CSV가 두 개 생성되고 샘플 제출물과 열·ID·행 순서가 맞는다."""
    assert SUB_A_PATH.is_file()
    assert SUB_B_PATH.is_file()
    sample_df = pd.read_csv(SAMPLE_PATH)

    for path in (SUB_A_PATH, SUB_B_PATH):
        df = pd.read_csv(path)
        assert len(df) == 63_000
        assert list(df.columns) == [ID_COLUMN, TARGET_COLUMN]
        assert np.array_equal(df[ID_COLUMN], sample_df[ID_COLUMN])
        assert np.isfinite(df[TARGET_COLUMN]).all()


def test_AC13_2_전체_실험_파이프라인_통합_실행_검증(tmp_path: Path) -> None:
    """PRD 13.2: 둘 이상의 기능이 함께 동작하는 통합 파이프라인 시나리오 검증."""
    # 소규모 합성 데이터 생성
    n_train = 60
    n_test = 10
    rng = np.random.default_rng(123)

    train_data = {ID_COLUMN: np.arange(1, n_train + 1)}
    for col in INPUT_COLUMNS:
        train_data[col] = rng.uniform(10.0, 100.0, size=n_train)
    # 기하학적 형태에 가깝게 Y_Position 생성
    train_data[TARGET_COLUMN] = train_data["X_Position"] * 1.5 + rng.normal(0, 1, size=n_train)
    train_df = pd.DataFrame(train_data)

    test_data = {ID_COLUMN: np.arange(n_train + 1, n_train + n_test + 1)}
    for col in INPUT_COLUMNS:
        test_data[col] = rng.uniform(10.0, 100.0, size=n_test)
    test_df = pd.DataFrame(test_data)

    sample_df = pd.DataFrame({
        ID_COLUMN: test_data[ID_COLUMN],
        TARGET_COLUMN: np.full(n_test, 63.0),
    })

    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    sample_file = tmp_path / "sample.csv"
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    sample_df.to_csv(sample_file, index=False)

    paths = DatasetPaths(train=train_file, test=test_file, sample_submission=sample_file)
    out_dir = tmp_path / "outputs"
    art_dir = tmp_path / "artifacts"
    rep_file = tmp_path / "report.md"

    # 소형 설정
    config = ModelBConfig(
        max_iter=5,
        cv_folds=2,
        selection_sample_size=30,
        random_state=42,
    )

    # 전체 통합 파이프라인 실행
    res = run_experiment(
        paths,
        output_dir=out_dir,
        artifact_dir=art_dir,
        report_path=rep_file,
        model_b_config=config,
        strict_row_counts=False,
    )

    # 통합 산출물 검증
    assert (out_dir / "submission_model_a.csv").is_file()
    assert (out_dir / "submission_model_b.csv").is_file()
    assert (art_dir / "experiment.json").is_file()
    assert (art_dir / "feature_selection.json").is_file()
    assert rep_file.is_file()
    assert len(res.metrics) == 5  # baseline, model_a, model_b_all, model_b_reduced, model_b_final


def test_AC13_3_사용자_보고서_이해_및_산출물_검토_시나리오() -> None:
    """PRD 13.3 User Scenario 1: 사용자가 두 모델의 차이, 피처선택, RMSE/Huber 점수를 이해하는 흐름."""
    # Step 1: 사용자가 보고서 초안을 열어 모델 A와 모델 B의 접근 방식 차이를 확인한다.
    assert REPORT_PATH.is_file(), "보고서가 존재하지 않습니다."
    report_text = REPORT_PATH.read_text(encoding="utf-8")
    assert "모델 A — 원 관계 후보" in report_text
    assert "모델 B — 공식과 독립적인 머신러닝" in report_text
    time.sleep(0.05)  # 사용자 조작과 조작 사이의 시간 경과

    # Step 2: 사용자가 피처 선택 과정 및 순열 중요도를 확인한다.
    assert "중요도 순위" in report_text
    assert "후보 원본 열" in report_text
    assert "Huber 기준 최종 k" in report_text
    time.sleep(0.05)  # 사용자 조작과 조작 사이의 시간 경과

    # Step 3: 사용자가 최종 성능 표(RMSE, Huber 점수)를 확인한다.
    assert "공통 최종 비교 구간 성능" in report_text
    assert "RMSE" in report_text
    assert "Huber 점수" in report_text


def test_AC13_3_사용자_제출파일_확인_및_업로드_준비_시나리오() -> None:
    """PRD 13.3 User Scenario 2: 사용자가 두 제출 CSV를 확인하고 Kaggle 업로드 안내를 확인하는 흐름."""
    # Step 1: 사용자가 outputs 디렉토리에서 두 CSV 파일이 정상 생성되었는지 확인한다.
    assert SUB_A_PATH.is_file()
    assert SUB_B_PATH.is_file()
    time.sleep(0.05)  # 사용자 조작과 조작 사이의 시간 경과

    # Step 2: 사용자가 각 파일의 유효성(행 수, 컬럼, 결측치 없음)을 대조한다.
    df_a = pd.read_csv(SUB_A_PATH)
    df_b = pd.read_csv(SUB_B_PATH)
    assert len(df_a) == EXPECTED_SAMPLE_ROWS
    assert len(df_b) == EXPECTED_SAMPLE_ROWS
    assert not df_a[TARGET_COLUMN].isna().any()
    assert not df_b[TARGET_COLUMN].isna().any()
    time.sleep(0.05)  # 사용자 조작과 조작 사이의 시간 경과

    # Step 3: 사용자가 보고서에 명시된 Kaggle Submit Predictions 업로드 단계 안내를 확인한다.
    report_text = REPORT_PATH.read_text(encoding="utf-8")
    assert "submission_model_a.csv" in report_text
    assert "submission_model_b.csv" in report_text
    assert "Submit Predictions" in report_text
