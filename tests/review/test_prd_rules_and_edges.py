"""PRD Business Rules, Error Cases, Non-Functional Requirements, and Constraints tests."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from orbit_predict.constants import ID_COLUMN, INPUT_COLUMNS, TARGET_COLUMN
from orbit_predict.data import DataContractError, make_submission, validate_submission
from orbit_predict.features import GeneralFeatureBuilder
from orbit_predict.metrics import competition_huber_score
from orbit_predict.model_a import CircleParameters, predict_circle
from orbit_predict.model_b import ModelB, ModelBConfig
from orbit_predict.split import make_split


ARTIFACTS_DIR = Path("artifacts")
DOCS_DIR = Path("docs")
REPORT_PATH = DOCS_DIR / "model-report.md"
SRC_DIR = Path("src/orbit_predict")


def test_BR01_대회_Huber_점수_공식_검증() -> None:
    """PRD BR-01: 대회 Huber 점수 = |e|<=5: 0.5*e^2, |e|>5: 5*(|e|-2.5) 행별 적용 평균 * 1,000."""
    y_true = np.array([10.0, 10.0], dtype=np.float64)
    y_pred = np.array([12.0, 18.0], dtype=np.float64)  # error: 2.0, 8.0

    # error 2.0: <= 5 -> 0.5 * 4.0 = 2.0
    # error 8.0: > 5  -> 5.0 * (8.0 - 2.5) = 5.0 * 5.5 = 27.5
    # mean = (2.0 + 27.5) / 2 = 14.75
    # score = 14.75 * 1000 = 14750.0
    expected_score = 14750.0
    calculated_score = competition_huber_score(y_true, y_pred)
    assert calculated_score == pytest.approx(expected_score)


def test_BR02_공개_비공개_비율_명시_및_비공개성능_미단정_검증() -> None:
    """PRD BR-02: 공개 리더보드는 약 37%, 최종 비공개는 약 63%이며 비공개 순위를 보장하지 않음을 명시."""
    assert REPORT_PATH.is_file()
    text = REPORT_PATH.read_text(encoding="utf-8")
    assert "비공개" in text
    assert "보장하지 않는다" in text or "미확인" in text


def test_BR03_Kaggle_제출_역할분담_보고서_기재_검증() -> None:
    """PRD BR-03: 사용자가 Kaggle 제출을 담당하고 개발AI는 초안 상태로 인계함을 보고서에 명시."""
    assert REPORT_PATH.is_file()
    text = REPORT_PATH.read_text(encoding="utf-8")
    assert "사용자가 수행하며" in text
    assert "실제 Kaggle 점수를 기재하지 않는다" in text or "미확인" in text


def test_EC01_q음수_또는_부호0_대체값_b_및_예외로그_검증() -> None:
    """PRD EC-01: q < 0 또는 Velocity*Altitude = 0이면 중심 Y값 b를 대체값으로 내고 해당 ID·사유·건수를 기록."""
    params = CircleParameters(
        center_x=630.0, center_y=550.0, radius=420.0,
        coefficient_d=-1260.0, coefficient_e=-1100.0, coefficient_f=522900.0
    )
    # Row 0: q < 0 (x = 2000, abs(x - 630) = 1370 > 420)
    # Row 1: Velocity == 0
    # Row 2: Altitude == 0
    df = pd.DataFrame({
        "Satellite_ID": [10, 20, 30],
        "X_Position": [2000.0, 630.0, 630.0],
        "Velocity": [1.0, 0.0, 5.0],
        "Altitude": [1.0, 5.0, 0.0],
    })
    res = predict_circle(df, params)
    assert np.all(res.predictions == pytest.approx(550.0))
    assert len(res.exceptions) == 3
    reasons = [item["reason"] for item in res.exceptions]
    assert "q_negative" in reasons[0]
    assert "velocity_altitude_product_zero" in reasons[1]
    assert "velocity_altitude_product_zero" in reasons[2]


def test_EC02_모델B_결측치_중앙값_대체_검증() -> None:
    """PRD EC-02: 모델 B는 학습 접힘에서 얻은 중앙값으로 숫자 결측을 대체."""
    builder = GeneralFeatureBuilder(("X_Position", "Velocity"))
    train_df = pd.DataFrame({
        "X_Position": [10.0, 20.0, 30.0],  # median = 20.0
        "Velocity": [1.0, 2.0, 3.0],      # median = 2.0
    })
    builder.fit(train_df)

    # 결측치(NaN, inf) 포함 데이터
    test_df = pd.DataFrame({
        "X_Position": [np.nan, np.inf],
        "Velocity": [-np.inf, np.nan],
    })
    transformed = builder.transform(test_df)
    assert np.isfinite(transformed).all()
    # Row 0 X_Position should be replaced by median 20.0
    assert transformed[0, 0] == pytest.approx(20.0)
    # Row 1 Velocity should be replaced by median 2.0
    # feature order: X_Position(0..3), Velocity(4..7), pairs(8..9)
    assert transformed[1, 4] == pytest.approx(2.0)


def test_EC02_제출파일_비유한값_및_행불일치_검출_검증() -> None:
    """PRD EC-02: 제출 파일의 행 수·ID 순서·비유한 예측값 오류 발생 시 DataContractError 발생."""
    sample = pd.DataFrame({
        ID_COLUMN: [1, 2, 3],
        TARGET_COLUMN: [63.0, 63.0, 63.0],
    })
    # 1. 길이 불일치
    with pytest.raises(DataContractError):
        make_submission(sample, [10.0, 20.0])

    # 2. 비유한 예측값 (NaN / Inf)
    with pytest.raises(DataContractError):
        make_submission(sample, [10.0, np.nan, 30.0])

    with pytest.raises(DataContractError):
        make_submission(sample, [10.0, np.inf, 30.0])

    # 3. ID 불일치
    wrong_id_sub = pd.DataFrame({
        ID_COLUMN: [1, 999, 3],
        TARGET_COLUMN: [10.0, 20.0, 30.0],
    })
    with pytest.raises(DataContractError):
        validate_submission(wrong_id_sub, sample)


def test_EC03_구간별_오차_및_입력분포_비교_산출물_검증() -> None:
    """PRD EC-03: 내부 구간별 오차, 학습/시험 입력 분포 차이를 기록한 산출물 생성 확인."""
    error_segments_path = ARTIFACTS_DIR / "error_segments.json"
    data_distribution_path = ARTIFACTS_DIR / "data_distribution.json"

    assert error_segments_path.is_file(), f"{error_segments_path} 부재"
    assert data_distribution_path.is_file(), f"{data_distribution_path} 부재"

    with open(error_segments_path, "r", encoding="utf-8") as f:
        err_data = json.load(f)
    assert "model_a" in err_data
    assert "model_b_final" in err_data
    assert "branch" in err_data["model_a"]
    assert "x_decile" in err_data["model_a"]

    with open(data_distribution_path, "r", encoding="utf-8") as f:
        dist_data = json.load(f)
    assert "train" in dist_data
    assert "test" in dist_data
    for col in INPUT_COLUMNS:
        assert col in dist_data["train"]
        assert col in dist_data["test"]
        assert "min" in dist_data["train"][col]
        assert "max" in dist_data["train"][col]


def test_EC04_사용자_미제출시_초안_상태_유지_검증() -> None:
    """PRD EC-04: 사용자 제출 결과 전달 전에는 보고서 초안 상태이며 Kaggle 점수는 미확인 상태 유지."""
    assert REPORT_PATH.is_file()
    text = REPORT_PATH.read_text(encoding="utf-8")
    assert "초안" in text
    assert "실제 Kaggle 점수를 기재하지 않는다" in text or "미확인" in text


def test_NFR01_데이터분할_재현성_검증() -> None:
    """PRD NFR-01: 고정 분할과 기록된 실행 환경으로 분할을 완벽히 재현."""
    split_a = make_split(1000, seed=42)
    split_b = make_split(1000, seed=42)
    assert np.array_equal(split_a.comparison_indices, split_b.comparison_indices)
    assert np.array_equal(split_a.development_indices, split_b.development_indices)


def test_NFR02_전처리_추정치_학습데이터_격리_검증() -> None:
    """PRD NFR-02: 모델 B의 결측 대체 등 전처리 값은 학습 접힘에서만 추정하며 평가 데이터에 누수되지 않는다."""
    builder = GeneralFeatureBuilder(("X_Position",))
    train_data = pd.DataFrame({"X_Position": [1.0, 2.0, 3.0]})  # median = 2.0
    builder.fit(train_data)

    test_data = pd.DataFrame({"X_Position": [1000.0, 2000.0, 3000.0]})
    # transform 후에도 medians_가 변경되지 않아야 함
    builder.transform(test_data)
    assert builder.medians_["X_Position"] == pytest.approx(2.0)


def test_NFR03_보고서_비공개_정답_확정표현_부재_검증() -> None:
    """PRD NFR-03: 관찰하지 않은 비공개 정답·점수·순위를 확정된 사실처럼 쓰지 않는다."""
    assert REPORT_PATH.is_file()
    text = REPORT_PATH.read_text(encoding="utf-8")
    forbidden_terms = ["1위 확정", "만점 보장", "완벽 검증 완료"]
    for term in forbidden_terms:
        assert term not in text, f"부적절한 확정 표현 감지: {term}"


def test_OOS01_배깅_스태킹_미포함_검증() -> None:
    """PRD 3.2 Out of Scope: 배깅 메타 에스티메이터와 모델 스태킹은 필수 개발 범위에서 제외."""
    for py_file in SRC_DIR.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "BaggingRegressor" not in content
        assert "StackingRegressor" not in content
        assert "BaggingClassifier" not in content
        assert "StackingClassifier" not in content


def test_OOS02_Kaggle_직접_업로드_로직_미포함_검증() -> None:
    """PRD 3.2 Out of Scope: 개발AI의 Kaggle 계정 로그인·직접 업로드·리더보드 캡처는 범위 밖."""
    for py_file in SRC_DIR.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "kaggle.api" not in content
        assert "kaggle competitions submit" not in content
        assert "authenticate" not in content


def test_TC01_모델B_모델A_수식_미사용_검증() -> None:
    """PRD 12.1 Technical Constraints: 모델 B는 모델 A의 공식·매개변수·예측값을 학습 입력으로 사용하지 않는다."""
    model_b_code = (SRC_DIR / "model_b.py").read_text(encoding="utf-8")
    features_code = (SRC_DIR / "features.py").read_text(encoding="utf-8")

    for code in (model_b_code, features_code):
        assert "630" not in code
        assert "550" not in code
        assert "420" not in code
        assert "predict_circle" not in code
        assert "fit_circle" not in code


def test_TC02_피처선택_선택전후_성능_기록_검증() -> None:
    """PRD 12.1 Technical Constraints: 피처 선택은 모델 B에 실제로 적용하고 선택 전후 점수를 보고한다."""
    fs_path = ARTIFACTS_DIR / "feature_selection.json"
    assert fs_path.is_file()
    with open(fs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "selected_k" in data
    assert "best_reduced_k" in data
    assert "candidates" in data
    assert len(data["candidates"]) >= 2
