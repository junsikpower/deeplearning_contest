"""Code regression and internal contract tests (INT prefix)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orbit_predict.data import (
    DataContractError,
    make_submission,
    validate_sample_submission_frame,
    validate_test_frame,
    validate_test_sample_alignment,
    validate_train_frame,
)
from orbit_predict.features import FeatureTransformError, GeneralFeatureBuilder
from orbit_predict.metrics import competition_huber_score, metric_dict, rmse
from orbit_predict.model_a import CircleParameters, ModelAError, fit_circle, predict_circle
from orbit_predict.model_b import ModelB, ModelBConfig, _validate_source_columns
from orbit_predict.reporting import json_ready
from orbit_predict.split import make_split


# --- features.py INT tests ---

def test_INT_GeneralFeatureBuilder_빈_소스컬럼_예외처리() -> None:
    """GeneralFeatureBuilder: 빈 source_columns 입력 시 ValueError 발생."""
    with pytest.raises(ValueError, match="at least one source column is required"):
        GeneralFeatureBuilder(())


def test_INT_GeneralFeatureBuilder_중복_소스컬럼_예외처리() -> None:
    """GeneralFeatureBuilder: 중복 source_columns 입력 시 ValueError 발생."""
    with pytest.raises(ValueError, match="source columns must be unique"):
        GeneralFeatureBuilder(("X_Position", "X_Position"))


def test_INT_GeneralFeatureBuilder_fit_호출전_transform_예외처리() -> None:
    """GeneralFeatureBuilder: fit() 호출 전 transform() 호출 시 FeatureTransformError 발생."""
    builder = GeneralFeatureBuilder(("X_Position",))
    df = pd.DataFrame({"X_Position": [1.0, 2.0]})
    with pytest.raises(FeatureTransformError, match="must be fit before transform"):
        builder.transform(df)


def test_INT_GeneralFeatureBuilder_fit_호출전_feature_names_접근_예외처리() -> None:
    """GeneralFeatureBuilder: fit() 호출 전 feature_names 접근 시 FeatureTransformError 발생."""
    builder = GeneralFeatureBuilder(("X_Position",))
    with pytest.raises(FeatureTransformError, match="must be fit before feature_names is read"):
        _ = builder.feature_names


def test_INT_GeneralFeatureBuilder_누락컬럼_변환시_예외처리() -> None:
    """GeneralFeatureBuilder: transform 데이터에 지정된 source_column이 누락된 경우 FeatureTransformError 발생."""
    builder = GeneralFeatureBuilder(("X_Position", "Velocity"))
    train_df = pd.DataFrame({"X_Position": [1.0, 2.0], "Velocity": [3.0, 4.0]})
    builder.fit(train_df)

    test_df = pd.DataFrame({"X_Position": [1.0, 2.0]})  # Velocity 누락
    with pytest.raises(FeatureTransformError, match="missing source columns"):
        builder.transform(test_df)


def test_INT_GeneralFeatureBuilder_단일컬럼_피처개수_및_행렬_검증() -> None:
    """GeneralFeatureBuilder: source_columns가 1개일 때 feature_count가 4개(pair 없음)인지 검증."""
    builder = GeneralFeatureBuilder(("X_Position",))
    df = pd.DataFrame({"X_Position": [10.0, 20.0]})
    matrix = builder.fit_transform(df)
    assert builder.feature_count == 4
    assert matrix.shape == (2, 4)
    assert builder.feature_names == (
        "X_Position",
        "abs__X_Position",
        "square__X_Position",
        "sign__X_Position",
    )


# --- model_a.py INT tests ---

def test_INT_fit_circle_필수컬럼_누락시_예외처리() -> None:
    """fit_circle: X_Position 또는 Y_Position이 누락된 경우 ModelAError 발생."""
    df = pd.DataFrame({"X_Position": [1.0, 2.0, 3.0]})
    with pytest.raises(ModelAError, match="model A requires columns"):
        fit_circle(df)


def test_INT_fit_circle_유효행_부족시_예외처리() -> None:
    """fit_circle: 유효한 (x, y) 행이 3개 미만인 경우 ModelAError 발생."""
    df = pd.DataFrame({"X_Position": [1.0, np.nan, 3.0], "Y_Position": [1.0, 2.0, np.nan]})
    with pytest.raises(ModelAError, match="needs at least three finite"):
        fit_circle(df)


def test_INT_fit_circle_공선점_입력시_랭크부족_예외처리() -> None:
    """fit_circle: 세 점이 일직선상에 있어 원을 결정할 수 없는 경우 rank deficient ModelAError 발생."""
    # (0, 0), (1, 1), (2, 2)
    df = pd.DataFrame({"X_Position": [0.0, 1.0, 2.0], "Y_Position": [0.0, 1.0, 2.0]})
    with pytest.raises(ModelAError, match="rank deficient|invalid radius"):
        fit_circle(df)


def test_INT_predict_circle_필수컬럼_누락시_예외처리() -> None:
    """predict_circle: 필수 컬럼 누락 시 ModelAError 발생."""
    params = CircleParameters(0.0, 0.0, 1.0, 0.0, 0.0, -1.0)
    df = pd.DataFrame({"X_Position": [0.5], "Velocity": [1.0]})  # Altitude 누락
    with pytest.raises(ModelAError, match="model A requires columns"):
        predict_circle(df, params)


def test_INT_predict_circle_Satellite_ID_없는경우_정상동작() -> None:
    """predict_circle: frame에 Satellite_ID가 없어도 row_position을 기본 ID로 하여 정상 동작."""
    params = CircleParameters(0.0, 0.0, 1.0, 0.0, 0.0, -1.0)
    df = pd.DataFrame({
        "X_Position": [0.0, 5.0],  # 5.0은 q < 0 예외 발생
        "Velocity": [1.0, 1.0],
        "Altitude": [1.0, 1.0],
    })
    res = predict_circle(df, params)
    assert len(res.predictions) == 2
    assert len(res.exceptions) == 1
    assert res.exceptions[0]["Satellite_ID"] == 1


# --- model_b.py INT tests ---

def test_INT_ModelB_생성자_비승인_소스컬럼_예외처리() -> None:
    """ModelB: INPUT_COLUMNS에 없는 비승인 컬럼 사용 시 생성자에서 ValueError 발생."""
    config = ModelBConfig()
    with pytest.raises(ValueError, match="not approved inputs"):
        ModelB(("Unapproved_Column",), config)


def test_INT_ModelB_생성자_중복_소스컬럼_예외처리() -> None:
    """ModelB: 중복된 컬럼 전달 시 생성자에서 ValueError 발생."""
    config = ModelBConfig()
    with pytest.raises(ValueError, match="source columns must be unique"):
        ModelB(("X_Position", "X_Position"), config)


def test_INT_ModelB_생성자_빈_소스컬럼_예외처리() -> None:
    """ModelB: 빈 컬럼 전달 시 생성자에서 ValueError 발생."""
    config = ModelBConfig()
    with pytest.raises(ValueError, match="requires at least one source column"):
        ModelB((), config)


def test_INT_ModelB_생성자_리스트입력시_튜플변환_검증() -> None:
    """ModelB: 리스트 등 Iterable 입력 시 tuple로 변환되어 저장됨을 검증."""
    config = ModelBConfig()
    model = ModelB(["X_Position", "Velocity"], config)
    assert isinstance(model.source_columns, tuple)
    assert model.source_columns == ("X_Position", "Velocity")


def test_INT_ModelB_미승인_소스컬럼_예외처리() -> None:
    """_validate_source_columns: INPUT_COLUMNS에 없는 비승인 컬럼 사용 시 ValueError 발생."""
    with pytest.raises(ValueError, match="not approved inputs"):
        _validate_source_columns(("Unapproved_Column",))


def test_INT_ModelB_fit_전_predict_호출시_예외처리() -> None:
    """ModelB: fit() 호출 전 predict() 호출 시 RuntimeError 발생."""
    model = ModelB(("X_Position",), ModelBConfig())
    df = pd.DataFrame({"X_Position": [1.0, 2.0]})
    with pytest.raises(RuntimeError, match="model B is not fitted"):
        model.predict(df)


def test_INT_ModelB_타깃길이_불일치_예외처리() -> None:
    """ModelB: frame과 target의 길이가 일치하지 않을 때 ValueError 발생."""
    model = ModelB(("X_Position",), ModelBConfig())
    df = pd.DataFrame({"X_Position": [1.0, 2.0]})
    with pytest.raises(ValueError, match="lengths must match"):
        model.fit(df, [10.0])


def test_INT_ModelB_비유한_타깃_예외처리() -> None:
    """ModelB: target에 NaN 또는 Inf가 포함되어 있을 때 ValueError 발생."""
    model = ModelB(("X_Position",), ModelBConfig())
    df = pd.DataFrame({"X_Position": [1.0, 2.0]})
    with pytest.raises(ValueError, match="target must be finite"):
        model.fit(df, [10.0, np.nan])


# --- metrics.py INT tests ---

def test_INT_metrics_빈배열_입력시_예외처리() -> None:
    """metrics: 빈 배열 전달 시 ValueError 발생."""
    with pytest.raises(ValueError, match="require at least one row"):
        rmse([], [])
    with pytest.raises(ValueError, match="require at least one row"):
        competition_huber_score([], [])
    with pytest.raises(ValueError, match="require at least one row"):
        metric_dict([], [])


def test_INT_metrics_형상불일치_예외처리() -> None:
    """metrics: y_true와 y_pred의 shape가 다를 때 ValueError 발생."""
    with pytest.raises(ValueError, match="must have the same shape"):
        rmse([1.0, 2.0], [1.0])


def test_INT_metrics_비유한값_입력시_예외처리() -> None:
    """metrics: NaN 또는 Inf가 포함된 경우 ValueError 발생."""
    with pytest.raises(ValueError, match="require finite"):
        rmse([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError, match="require finite"):
        competition_huber_score([1.0, 2.0], [1.0, np.inf])


def test_INT_competition_huber_score_delta_음수_예외처리() -> None:
    """competition_huber_score: delta <= 0인 경우 ValueError 발생."""
    with pytest.raises(ValueError, match="delta must be positive"):
        competition_huber_score([1.0], [1.0], delta=0.0)


# --- data.py INT tests ---

def test_INT_data_validate_train_frame_컬럼불일치_예외처리() -> None:
    """validate_train_frame: 컬럼 순서나 이름이 다를 경우 DataContractError 발생."""
    df = pd.DataFrame({"X_Position": [1.0], "Y_Position": [2.0]})
    with pytest.raises(DataContractError, match="train columns must be"):
        validate_train_frame(df, expected_rows=None)


def test_INT_data_validate_train_frame_중복ID_예외처리() -> None:
    """validate_train_frame: Satellite_ID에 중복이 있을 경우 DataContractError 발생."""
    from orbit_predict.constants import ID_COLUMN, INPUT_COLUMNS, TARGET_COLUMN
    data = {ID_COLUMN: [100, 100]}  # 중복
    data.update({col: [1.0, 2.0] for col in INPUT_COLUMNS})
    data[TARGET_COLUMN] = [10.0, 20.0]
    df = pd.DataFrame(data)
    with pytest.raises(DataContractError, match="duplicate Satellite_ID"):
        validate_train_frame(df, expected_rows=None)


def test_INT_data_validate_test_sample_alignment_불일치_예외처리() -> None:
    """validate_test_sample_alignment: test와 sample_submission의 ID가 불일치할 때 DataContractError 발생."""
    test_df = pd.DataFrame({"Satellite_ID": [1, 2]})
    sample_df = pd.DataFrame({"Satellite_ID": [1, 3]})
    with pytest.raises(DataContractError, match="IDs are not aligned"):
        validate_test_sample_alignment(test_df, sample_df)


def test_INT_data_make_submission_예측값_길이불일치_예외처리() -> None:
    """make_submission: sample_submission 행 수와 예측값 개수가 다르면 DataContractError 발생."""
    sample_df = pd.DataFrame({"Satellite_ID": [1, 2], "Y_Position": [63.0, 63.0]})
    with pytest.raises(DataContractError, match="does not match sample rows"):
        make_submission(sample_df, [10.0])


# --- split.py INT tests ---

def test_INT_split_make_split_음수행_또는_잘못된_비율_예외처리() -> None:
    """make_split: row_count <= 0 또는 comparison_fraction 범위를 벗어날 때 ValueError 발생."""
    with pytest.raises(ValueError, match="row_count must be positive"):
        make_split(0)
    with pytest.raises(ValueError, match="comparison_fraction must be between 0 and 1"):
        make_split(100, comparison_fraction=1.5)


# --- reporting.py INT tests ---

def test_INT_reporting_json_ready_특수타입_직렬화_검증() -> None:
    """json_ready: numpy scalar, nan, inf, 중첩 구조 변환 정상 처리 검증."""
    payload = {
        "np_int": np.int64(42),
        "np_float": np.float64(3.14),
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "list_val": [np.int32(1), float("-inf")],
    }
    cleaned = json_ready(payload)
    assert cleaned["np_int"] == 42
    assert cleaned["np_float"] == pytest.approx(3.14)
    assert cleaned["nan_val"] is None
    assert cleaned["inf_val"] is None
    assert cleaned["list_val"] == [1, None]
