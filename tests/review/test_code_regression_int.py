"""Code regression and internal contract tests (INT prefix)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from orbit_predict.cli import main
from orbit_predict.data import (
    DataContractError,
    DatasetPaths,
    load_datasets,
    make_submission,
    validate_sample_submission_frame,
    validate_test_frame,
    validate_test_sample_alignment,
    validate_train_frame,
)
from orbit_predict.features import FeatureTransformError, GeneralFeatureBuilder
from orbit_predict.metrics import competition_huber_score, metric_dict, rmse
from orbit_predict.model_a import CircleParameters, ModelAError, fit_circle, predict_circle
from orbit_predict.model_b import (
    ModelB,
    ModelBConfig,
    _make_folds,
    _validate_source_columns,
    cross_validate_candidate,
    select_features,
)
from orbit_predict.reporting import (
    _fmt_metric,
    json_ready,
    write_json,
    write_markdown_report,
)
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


def test_INT_ModelB_빈_학습데이터_fit시_예외처리() -> None:
    """ModelB: 빈 데이터(0행) 전달 시 ValueError 발생."""
    model = ModelB(("X_Position",), ModelBConfig())
    df = pd.DataFrame({"X_Position": []})
    with pytest.raises(ValueError, match="requires at least one training row"):
        model.fit(df, [])


def test_INT_ModelB_fit_전_transform_호출시_예외처리() -> None:
    """ModelB: fit() 호출 전 transform() 호출 시 RuntimeError 발생."""
    model = ModelB(("X_Position",), ModelBConfig())
    df = pd.DataFrame({"X_Position": [1.0, 2.0]})
    with pytest.raises(RuntimeError, match="model B is not fitted"):
        model.transform(df)


def test_INT_ModelB_fit_전_feature_names_접근시_예외처리() -> None:
    """ModelB: fit() 호출 전 feature_names 접근 시 RuntimeError 발생."""
    model = ModelB(("X_Position",), ModelBConfig())
    with pytest.raises(RuntimeError, match="model B is not fitted"):
        _ = model.feature_names


def test_INT_ModelB_make_folds_행수부족_예외처리() -> None:
    """_make_folds: row_count < cv_folds인 경우 ValueError 발생."""
    config = ModelBConfig(cv_folds=5)
    with pytest.raises(ValueError, match="at least 5 rows are required"):
        _make_folds(3, config)


def test_INT_ModelB_cross_validate_candidate_타깃길이_불일치_예외처리() -> None:
    """cross_validate_candidate: frame과 target의 행 수 불일치 시 ValueError 발생."""
    config = ModelBConfig(cv_folds=2)
    df = pd.DataFrame({"X_Position": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="frame and target lengths must match"):
        cross_validate_candidate(df, [10.0, 20.0], ("X_Position",), config=config)


def test_INT_ModelB_select_features_타깃길이_불일치_예외처리() -> None:
    """select_features: development_frame과 target의 길이 불일치 시 ValueError 발생."""
    df = pd.DataFrame({"X_Position": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="development frame and target lengths must match"):
        select_features(df, [10.0, 20.0])


def test_INT_ModelB_select_features_표본수_CV접힘미달_예외처리() -> None:
    """select_features: sample_size < cv_folds인 경우 ValueError 발생."""
    config = ModelBConfig(cv_folds=5, selection_sample_size=3)
    df = pd.DataFrame({"X_Position": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="selection sample is too small"):
        select_features(df, [10.0, 20.0, 30.0], config=config)


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


def test_INT_data_validate_sample_submission_컬럼불일치_예외처리() -> None:
    """validate_sample_submission_frame: 컬럼 불일치 시 DataContractError 발생."""
    df = pd.DataFrame({"Wrong_ID": [1], "Y_Position": [63.0]})
    with pytest.raises(DataContractError, match="sample_submission columns must be"):
        validate_sample_submission_frame(df, expected_rows=None)


def test_INT_data_validate_train_frame_비숫자_타깃_예외처리() -> None:
    """validate_train_frame: 타깃 Y_Position에 비숫자 문자열 포함 시 DataContractError 발생."""
    from orbit_predict.constants import ID_COLUMN, INPUT_COLUMNS, TARGET_COLUMN
    data = {ID_COLUMN: [100]}
    data.update({col: [1.0] for col in INPUT_COLUMNS})
    data[TARGET_COLUMN] = ["not_a_number"]
    df = pd.DataFrame(data)
    with pytest.raises(DataContractError, match="contains a non-numeric value"):
        validate_train_frame(df, expected_rows=None)


def test_INT_data_validate_train_frame_소수점_ID_예외처리() -> None:
    """validate_train_frame: Satellite_ID에 소수점(실수) 포함 시 DataContractError 발생."""
    from orbit_predict.constants import ID_COLUMN, INPUT_COLUMNS, TARGET_COLUMN
    data = {ID_COLUMN: [1.5]}
    data.update({col: [1.0] for col in INPUT_COLUMNS})
    data[TARGET_COLUMN] = [10.0]
    df = pd.DataFrame(data)
    with pytest.raises(DataContractError, match="must contain integer IDs"):
        validate_train_frame(df, expected_rows=None)


def test_INT_data_load_datasets_존재하지않는_파일_예외처리() -> None:
    """load_datasets: 존재하지 않는 파일 경로 전달 시 FileNotFoundError 발생."""
    paths = DatasetPaths(
        train=Path("non_existent_train.csv"),
        test=Path("non_existent_test.csv"),
        sample_submission=Path("non_existent_sample.csv"),
    )
    with pytest.raises(FileNotFoundError, match="input CSV does not exist"):
        load_datasets(paths)


# --- cli.py INT tests ---

def test_INT_cli_인자_상호배타_예외처리() -> None:
    """cli: --check-inputs와 --run은 상호 배타적이며 둘 다 지정하거나 둘 다 누락 시 SystemExit 발생."""
    with pytest.raises(SystemExit, match="choose exactly one of --check-inputs or --run"):
        main([])
    with pytest.raises(SystemExit, match="choose exactly one of --check-inputs or --run"):
        main(["--check-inputs", "--run"])


def test_INT_cli_check_inputs_정상실행_검증() -> None:
    """cli: --check-inputs 플래그로 정상 실행 시 종료 코드 0 반환."""
    rc = main(["--check-inputs"])
    assert rc == 0


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


def test_INT_reporting_fmt_metric_다양한_입력_포매팅_검증() -> None:
    """_fmt_metric: None 처리('미기록') 및 부동소수점 포매팅 경계값 검증."""
    assert _fmt_metric(None) == "미기록"
    assert _fmt_metric(1234.56789) == "1,234.567890"
    assert _fmt_metric(0) == "0.000000"
    assert _fmt_metric("3.14") == "3.140000"


def test_INT_reporting_write_markdown_report_최신커밋_문구_및_내용_검증(tmp_path: Path) -> None:
    """write_markdown_report: 직전 커밋 변경 문구 및 보고서 구조/형식 검증."""
    class DummyResult:
        metrics = [
            {
                "stage": "model_a",
                "rmse": 1.234,
                "huber_score": 10.5,
                "max_absolute_error": 5.0,
                "note": "test note",
            },
            {
                "stage": "model_b_final",
                "rmse": None,
                "huber_score": 12.3,
                "max_absolute_error": None,
            },
        ]
        feature_selection = {
            "importance": [
                {"source_column": "X_Position", "mean_huber_increase": 100.0},
            ],
            "candidates": [
                {
                    "source_columns": ("X_Position",),
                    "mean_rmse": 5.0,
                    "mean_huber_score": 20.0,
                    "feature_count": 4,
                    "estimated_matrix_bytes": 1024,
                }
            ],
            "sample_size": 1000,
            "ranked_columns": ["X_Position"],
            "selected_k": 1,
            "selected_columns": ["X_Position"],
            "best_reduced_k": 1,
            "best_reduced_columns": ["X_Position"],
        }
        outputs = {"sub_a": "submission_model_a.csv"}
        train_rows = 636363
        test_rows = 63000
        sample_rows = 63000
        input_columns = ["X_Position", "Velocity"]
        comparison_rows = 127273
        development_rows = 509090
        model_b_config = "DummyConfig"
        model_a_parameters = "DummyParams"
        model_a_development_exception_count = 0
        model_a_test_exception_count = 0

    report_file = tmp_path / "report.md"
    write_markdown_report(report_file, DummyResult())

    assert report_file.is_file()
    content = report_file.read_text(encoding="utf-8")

    # 직전 커밋에서 수정한 정확한 문구 검증
    expected_sentence = "두 CSV는 `Satellite_ID,Y_Position` 순서이며 샘플 제출물의 ID·행 순서를 검증했다. 사용자가 Kaggle에서 직접 제출하며, 다음 순서로 진행한다."
    assert expected_sentence in content

    # 필수 섹션 포함 검증
    assert "## 1. 입력 데이터와 재현 조건" in content
    assert "## 2. 모델 경로" in content
    assert "## 3. 공통 최종 비교 구간 성능" in content
    assert "## 4. 제출 파일과 사용자 업로드" in content
    assert "## 5. 한계와 해석 주의" in content
    assert "| model_a | 1.234000 | 10.500000 | 5.000000 | test note |" in content
    assert "| model_b_final | 미기록 | 12.300000 | 미기록 |  |" in content


def test_INT_reporting_write_json_부모디렉토리_자동생성_및_무결성_검증(tmp_path: Path) -> None:
    """write_json: 부모 디렉토리 미존재 시 자동 생성 및 UTF-8 한글 JSON 무결성 검증."""
    nested_path = tmp_path / "deep" / "nested" / "dir" / "artifact.json"
    payload = {
        "title": "테스트 산출물",
        "score": 123.456,
        "items": [1, 2, np.float64(3.0)],
    }
    write_json(nested_path, payload)

    assert nested_path.is_file()
    text = nested_path.read_text(encoding="utf-8")
    assert "테스트 산출물" in text  # ensure_ascii=False 확인
    assert "\n" in text


def test_INT_reporting_json_ready_중첩_컨테이너_및_비유한값_변환_검증() -> None:
    """json_ready: 튜플, 리스트, 딕셔너리 중첩 및 nan/inf의 None 재귀 변환 검증."""
    data = {
        "tuple_data": (1, float("nan"), 3),
        "nested_dict": {"sub": [float("-inf"), float("inf"), "safe"]},
        "already_safe": "hello",
    }
    cleaned = json_ready(data)
    assert cleaned["tuple_data"] == [1, None, 3]
    assert cleaned["nested_dict"]["sub"] == [None, None, "safe"]
    assert cleaned["already_safe"] == "hello"
