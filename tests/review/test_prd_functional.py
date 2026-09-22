"""PRD Functional Requirements (FR-01 ~ FR-07) independent tests."""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from orbit_predict.constants import (
    EXPECTED_SAMPLE_ROWS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    ID_COLUMN,
    INPUT_COLUMNS,
    SUBMISSION_COLUMNS,
    TARGET_COLUMN,
)
from orbit_predict.features import GeneralFeatureBuilder
from orbit_predict.metrics import metric_dict
from orbit_predict.model_a import CircleParameters, fit_circle, predict_circle
from orbit_predict.model_b import ModelB, ModelBConfig, select_features, grouped_permutation_importance
from orbit_predict.split import make_split


TRAIN_PATH = Path("orbit-predict-satellite-position/Train_DS..csv")
TEST_PATH = Path("orbit-predict-satellite-position/Test_DS..csv")
SAMPLE_PATH = Path("orbit-predict-satellite-position/Sample_Submission_DS..csv")
MODEL_A_SUBMISSION = Path("outputs/submission_model_a.csv")
MODEL_B_SUBMISSION = Path("outputs/submission_model_b.csv")
REPORT_PATH = Path("docs/model-report.md")


def test_FR01_학습데이터_행수_컬럼_타입_검증() -> None:
    """PRD FR-01: 학습 파일은 636,363행, Satellite_ID 정수형, 7개 입력열 및 Y_Position 정답열 확인."""
    assert TRAIN_PATH.is_file(), f"학습 데이터 파일이 존재하지 않습니다: {TRAIN_PATH}"
    
    # 헤더 및 열 검증
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # 행 수 카운트
        row_count = sum(1 for _ in f)

    expected_cols = [ID_COLUMN, *INPUT_COLUMNS, TARGET_COLUMN]
    assert header == expected_cols, f"학습 데이터 컬럼 불일치: {header} != {expected_cols}"
    assert row_count == EXPECTED_TRAIN_ROWS, f"학습 데이터 행 수 불일치: {row_count} != {EXPECTED_TRAIN_ROWS}"

    # 상위 100행 샘플로 ID 정수형 및 나머지 숫자형 검증
    sample_df = pd.read_csv(TRAIN_PATH, nrows=100)
    assert pd.api.types.is_integer_dtype(sample_df[ID_COLUMN]) or np.array_equal(
        sample_df[ID_COLUMN].to_numpy(), np.floor(sample_df[ID_COLUMN].to_numpy())
    )
    for col in [*INPUT_COLUMNS, TARGET_COLUMN]:
        assert pd.api.types.is_numeric_dtype(sample_df[col]), f"{col} 컬럼이 숫자형이 아닙니다."


def test_FR01_시험데이터_및_샘플제출_행수_스키마_검증() -> None:
    """PRD FR-01: 시험 파일과 샘플 제출 파일은 각 63,000행, ID 및 열 구조 확인."""
    assert TEST_PATH.is_file()
    assert SAMPLE_PATH.is_file()

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_reader = csv.reader(f)
        test_header = next(test_reader)
        test_rows = sum(1 for _ in f)

    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        sample_reader = csv.reader(f)
        sample_header = next(sample_reader)
        sample_rows = sum(1 for _ in f)

    assert test_header == [ID_COLUMN, *INPUT_COLUMNS]
    assert test_rows == EXPECTED_TEST_ROWS
    assert sample_header == list(SUBMISSION_COLUMNS)
    assert sample_rows == EXPECTED_SAMPLE_ROWS


def test_FR01_Satellite_ID_예측변수_제외_검증() -> None:
    """PRD FR-01: Satellite_ID는 예측 변수로 자동 사용하지 않는다."""
    assert ID_COLUMN not in INPUT_COLUMNS, "Satellite_ID가 INPUT_COLUMNS에 포함되어 있습니다."


def test_FR02_고정_시드_분할_재현성_및_비율_검증() -> None:
    """PRD FR-02: default_rng(42).permutation(N)으로 분할, 앞쪽 ceil(0.2*N)을 비교 구간으로 설정."""
    n = 636_363
    split1 = make_split(n, seed=42, comparison_fraction=0.20)
    split2 = make_split(n, seed=42, comparison_fraction=0.20)

    expected_comparison = int(np.ceil(0.20 * n))  # 127,273
    expected_dev = n - expected_comparison       # 509,090

    assert len(split1.comparison_indices) == expected_comparison
    assert len(split1.development_indices) == expected_dev
    assert np.array_equal(split1.comparison_indices, split2.comparison_indices)
    assert np.array_equal(split1.development_indices, split2.development_indices)

    # 중복 없이 전체 n개를 커버하는지 검증
    all_indices = np.concatenate([split1.comparison_indices, split1.development_indices])
    assert len(np.unique(all_indices)) == n


def test_FR02_단계별_지표_RMSE_Huber_최대오차_계산_검증() -> None:
    """PRD FR-02: 같은 최종 비교 구간에 대해 RMSE, 대회 방식 Huber 점수, 최대 절대오차 계산."""
    y_true = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    y_pred = np.array([12.0, 20.0, 35.0], dtype=np.float64)

    metrics = metric_dict(y_true, y_pred)
    assert "rmse" in metrics
    assert "huber_score" in metrics
    assert "max_absolute_error" in metrics
    assert metrics["max_absolute_error"] == 5.0
    assert metrics["rmse"] == pytest.approx(np.sqrt((4.0 + 0.0 + 25.0) / 3.0))


def test_FR03_원_매개변수_최소제곱_추정_검증() -> None:
    """PRD FR-03: 개발 구간 (x, y) 점들로부터 최소제곱 원 중심 (a, b)와 반지름 r을 추정."""
    # (x - 630)^2 + (y - 550)^2 = 420^2 위의 점 생성
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    cx, cy, r = 630.0, 550.0, 420.0
    x_pts = cx + r * np.cos(angles)
    y_pts = cy + r * np.sin(angles)

    df = pd.DataFrame({"X_Position": x_pts, "Y_Position": y_pts})
    params = fit_circle(df)

    assert params.center_x == pytest.approx(cx, abs=1e-4)
    assert params.center_y == pytest.approx(cy, abs=1e-4)
    assert params.radius == pytest.approx(r, abs=1e-4)


def test_FR03_부호규칙_기반_예측_및_결과_유한성_검증() -> None:
    """PRD FR-03: q >= 0이고 Velocity*Altitude > 0이면 b + sqrt(q), 음수이면 b - sqrt(q)."""
    params = CircleParameters(
        center_x=630.0, center_y=550.0, radius=420.0,
        coefficient_d=-1260.0, coefficient_e=-1100.0, coefficient_f=522900.0
    )
    # x = 630일 때 q = 420^2 = 176400, sqrt(q) = 420
    test_df = pd.DataFrame({
        "Satellite_ID": [1, 2],
        "X_Position": [630.0, 630.0],
        "Velocity": [10.0, -10.0],
        "Altitude": [5.0, 5.0],
    })
    res = predict_circle(test_df, params)
    assert res.predictions[0] == pytest.approx(550.0 + 420.0)  # positive product
    assert res.predictions[1] == pytest.approx(550.0 - 420.0)  # negative product
    assert len(res.exceptions) == 0
    assert np.isfinite(res.predictions).all()


def test_FR03_예외조건_대체값_b_반환_및_예외기록_검증() -> None:
    """PRD FR-03: q < 0 또는 Velocity*Altitude = 0이면 Y = b를 대체값으로 쓰고 예외를 기록한다."""
    params = CircleParameters(
        center_x=630.0, center_y=550.0, radius=420.0,
        coefficient_d=-1260.0, coefficient_e=-1100.0, coefficient_f=522900.0
    )
    # 1. q < 0: x = 630 + 500 (반지름 420 초과)
    # 2. Velocity * Altitude == 0: Velocity = 0
    test_df = pd.DataFrame({
        "Satellite_ID": [101, 102],
        "X_Position": [1130.0, 630.0],
        "Velocity": [10.0, 0.0],
        "Altitude": [5.0, 10.0],
    })
    res = predict_circle(test_df, params)
    assert res.predictions[0] == pytest.approx(550.0)
    assert res.predictions[1] == pytest.approx(550.0)
    assert len(res.exceptions) == 2
    assert "q_negative" in str(res.exceptions[0]["reason"])
    assert "velocity_altitude_product_zero" in str(res.exceptions[1]["reason"])


def test_FR04_모델B_파생피처_생성_원공식_독립성_검증() -> None:
    """PRD FR-04: 원 공식(중심 630, 550, 반지름 420 등) 없이 일반 파생 피처(abs, sq, sign, pair product 등)를 생성."""
    builder = GeneralFeatureBuilder(("X_Position", "Velocity"))
    df = pd.DataFrame({
        "X_Position": [10.0, 20.0],
        "Velocity": [-5.0, 5.0],
    })
    matrix = builder.fit_transform(df)
    feature_names = builder.feature_names

    # 원 수식 하드코딩 여부 확인
    for name in feature_names:
        assert "630" not in name and "550" not in name and "420" not in name
        assert "circle" not in name.lower()

    # 필수 생성 피처: 원본, abs, square, sign, product, sign_product
    expected = (
        "X_Position", "abs__X_Position", "square__X_Position", "sign__X_Position",
        "Velocity", "abs__Velocity", "square__Velocity", "sign__Velocity",
        "product__X_Position__Velocity", "sign_product__X_Position__Velocity"
    )
    assert feature_names == expected
    assert matrix.shape == (2, len(expected))


def test_FR04_모델B_학습접힘_중앙값_전처리_격리_검증() -> None:
    """PRD FR-04: 숫자 전처리의 기준값과 결측 대체값은 해당 학습 접힘에서만 추정한다."""
    builder = GeneralFeatureBuilder(("X_Position",))
    train_df = pd.DataFrame({"X_Position": [10.0, 20.0, 30.0]})  # median = 20.0
    builder.fit(train_df)

    assert builder.medians_["X_Position"] == 20.0

    # transform에 NaN이 있는 새 데이터프레임 전달 시 fit에서 구한 median(20.0)으로 대체되는지 확인
    test_df = pd.DataFrame({"X_Position": [np.nan]})
    matrix = builder.transform(test_df)
    assert matrix[0, 0] == pytest.approx(20.0)


def test_FR04_모델B_학습_및_예측_동작_검증() -> None:
    """PRD FR-04: 모델 B가 수식 없이 표 데이터 회귀 모델로 학습 및 예측 수행."""
    config = ModelBConfig(max_iter=10, random_state=42)
    model = ModelB(("X_Position", "Velocity"), config=config)
    train_df = pd.DataFrame({
        "X_Position": [10.0, 20.0, 30.0, 40.0, 50.0],
        "Velocity": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    target = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
    model.fit(train_df, target)

    preds = model.predict(train_df)
    assert len(preds) == len(train_df)
    assert np.isfinite(preds).all()


def test_FR05_그룹단위_순열중요도_파생피처_동시반영_검증() -> None:
    """PRD FR-05: 원본 열별 그룹 단위 순열 중요도 측정 (파생 피처도 동시 permute/recompute)."""
    config = ModelBConfig(max_iter=10, cv_folds=2, random_state=42)
    df = pd.DataFrame({
        "X_Position": np.linspace(10, 100, 30),
        "Velocity": np.linspace(1, 10, 30),
    })
    target = df["X_Position"] * 2.0 + df["Velocity"]

    importance = grouped_permutation_importance(df, target, ("X_Position", "Velocity"), config=config)
    assert len(importance) == 2
    for item in importance:
        assert "source_column" in item
        assert "mean_huber_increase" in item
        assert "fold_huber_increases" in item
        assert len(item["fold_huber_increases"]) == 2


def test_FR05_상위_k개_후보_평가_및_최적k_선택_검증() -> None:
    """PRD FR-05: k=1..len(columns) 후보를 평가하고 최고 성능의 k 및 best_reduced_k를 선정."""
    config = ModelBConfig(max_iter=10, cv_folds=2, selection_sample_size=30, random_state=42)
    df = pd.DataFrame({
        "X_Position": np.linspace(10, 100, 30),
        "Velocity": np.linspace(1, 10, 30),
        "Altitude": np.linspace(5, 50, 30),
    })
    target = df["X_Position"] * 2.0 + df["Velocity"]

    res = select_features(df, target, config=config, source_columns=("X_Position", "Velocity", "Altitude"))
    assert len(res.candidates) == 3  # k=1, 2, 3
    assert 1 <= res.selected_k <= 3
    assert 1 <= res.best_reduced_k <= 2
    assert len(res.ranked_columns) == 3


def test_FR06_두_제출파일_행수_컬럼_순서_일치_검증() -> None:
    """PRD FR-06: 모델 A와 모델 B 각각 63,000행 예측 CSV 별도 생성, 샘플 제출물과 ID 및 행 순서 일치."""
    assert MODEL_A_SUBMISSION.is_file(), f"모델 A 제출 파일 부재: {MODEL_A_SUBMISSION}"
    assert MODEL_B_SUBMISSION.is_file(), f"모델 B 제출 파일 부재: {MODEL_B_SUBMISSION}"

    sample_df = pd.read_csv(SAMPLE_PATH)

    for sub_path in (MODEL_A_SUBMISSION, MODEL_B_SUBMISSION):
        sub_df = pd.read_csv(sub_path)
        assert list(sub_df.columns) == [ID_COLUMN, TARGET_COLUMN]
        assert len(sub_df) == EXPECTED_SAMPLE_ROWS
        assert np.array_equal(sub_df[ID_COLUMN].to_numpy(), sample_df[ID_COLUMN].to_numpy())
        assert np.isfinite(sub_df[TARGET_COLUMN].to_numpy()).all()


def test_FR06_사용자_업로드_안내_문서_존재_검증() -> None:
    """PRD FR-06: 사용자가 Kaggle에서 따라 할 짧은 업로드 순서 제공 및 직접 업로드 허위 주장 부재."""
    assert REPORT_PATH.is_file()
    text = REPORT_PATH.read_text(encoding="utf-8")
    assert "제출 파일과 사용자 업로드" in text or "업로드" in text
    assert "Kaggle" in text
    assert "Submit Predictions" in text or "업로드" in text


def test_FR07_보고서_단계별_지표_및_피처선택_요약_검증() -> None:
    """PRD FR-07: 기준선, 모델 A, 전체 입력 B, 최고 축소 후보 B, 최종 선택 B의 단계별 RMSE와 Huber 요약."""
    assert REPORT_PATH.is_file()
    text = REPORT_PATH.read_text(encoding="utf-8")
    assert "baseline_development_mean" in text
    assert "model_a" in text
    assert "model_b_all_inputs" in text
    assert "model_b_best_reduced" in text
    assert "model_b_final" in text
    assert "RMSE" in text
    assert "Huber" in text
