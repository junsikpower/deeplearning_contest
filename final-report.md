# Orbit — Predict Satellite Position 머신러닝 경진대회 최종 보고서

- 팀: dku_22962
- 작성일: 2026-09-22
- 확인한 코드 기준: GitHub master 커밋 8b5d5cfe7776c90aff2df551c456c6704a249ad0
- 대회: https://www.kaggle.com/competitions/orbit-predict-satellite-position
- 목표: 시험 데이터의 Y_Position을 예측하고, 피처 선택을 포함한 개발 과정과 단계별 RMSE 및 리더보드 결과를 설명한다.

## 1. 결과 요약

이번 과제에서는 같은 데이터에 대해 두 가지 접근을 비교했다. 모델 B는 일반적인 표 데이터 머신러닝 모델이다. 일곱 개의 관측 열을 사용해 학습한 다음, 검증 점수를 근거로 중요한 열을 선택했다. 모델 A는 학습 데이터에서 확인한 원의 기하 관계를 이용한다. 마지막으로 모델 A가 추정한 중심과 반지름이 거의 정수라는 점을 확인해, 정수값을 적용한 추가 제출 후보를 만들었다.

최종 추가 후보의 Kaggle **공개 점수는 0.00000**, 첨부한 화면에서 팀 dku_22962는 **공개 7위**였다. 이 점수는 화면에 표시된 최고 점수와 같지만, 소수점 다섯 자리 표시만으로 실제 내부 점수가 완전히 같은지는 알 수 없다. 대회 화면에 따르면 공개 순위는 시험 데이터 약 37%로 계산되며, 나머지 약 63%에 따른 최종 순위는 아직 확정되지 않았다.

## 2. 데이터와 평가 방법

제공된 학습 CSV는 636,363행이며 정답 Y_Position이 있다. 시험 CSV와 샘플 제출 CSV는 각각 63,000행이고 시험 CSV에는 정답이 없다. Satellite_ID는 예측 변수로 넣지 않고 제출 행을 맞추는 데 사용했다. 모델에 검토한 일곱 입력은 X_Position, Velocity, Altitude, Fuel_Level, Signal_Strength, Battery_Temp, Solar_Exposure이다.

학습 데이터의 행을 고정 시드 42로 섞고 20%인 127,273행을 공통 비교 구간으로 남겼다. 나머지 509,090행은 모델 개발에 사용했다. 모델 B의 피처 선택은 개발 구간에서 추출한 최대 100,000행에 대해 3겹 교차검증으로 수행했다. 이렇게 한 이유는 모델을 선택하는 단계에 최종 비교 구간의 정답을 사용하지 않기 위해서다.

성능은 RMSE와 대회 방식의 Huber 점수를 함께 기록했다. RMSE는 예측 Y와 실제 Y의 차이를 제곱해 평균낸 뒤 제곱근을 취한 값으로, Y_Position과 단위가 같다. 대회 Huber 점수는 작은 오차에는 제곱 손실을, 큰 오차에는 선형 손실을 적용한 뒤 평균에 1,000을 곱한 값이다. 두 값 모두 **낮을수록 좋지만 서로 같은 숫자가 아니다.** Kaggle 공개 점수는 숨겨진 시험 정답 일부에 대한 Huber 점수이고, 보고서의 RMSE는 학습 CSV에서 떼어 둔 비교 구간의 점수다.

## 3. 모델을 개발하고 개선한 과정

### 3.1 어느 정도를 잘 맞춘 것으로 볼 것인가

복잡한 모델을 보기 전에 개발 구간의 Y 평균 하나로 모든 비교 행을 예측하는 기준선을 만들었다. 이 방법의 RMSE는 **296.444468**, Huber 점수는 **1,321,767.551843**이었다. 단순한 숫자 하나만으로는 위아래로 넓게 퍼진 Y값을 설명하기 어렵다는 것을 확인했다. 이후 모델의 성능은 이 기준선보다 얼마나 줄었는지와, Kaggle의 실제 제출 점수가 어떠한지를 함께 보았다.

### 3.2 첫 일반 학습 기준: 모델 B

원의 수식이 시험 데이터에도 적용된다고 단정할 수 없었으므로, 수식에 의존하지 않는 별도의 머신러닝 경로를 기준 모델로 두었다. 사용한 학습기는 HistGradientBoostingRegressor라는 비선형 회귀 모델이다. 이 모델은 입력값을 여러 구간으로 나누는 나무를 반복적으로 학습해 Y를 예측한다. 랜덤 포레스트를 시험했다고 쓰지는 않았다. 이 프로젝트에서 실제 사용한 모델이 아니기 때문이다.

처음에는 일곱 원본 입력을 모두 사용했다. 숫자의 절댓값·제곱·부호 및 두 열의 곱과 부호 조합을 일반적인 후보 피처로 만들되, 원의 중심이나 모델 A 예측값은 모델 B에 넣지 않았다. 전체 입력 모델 B의 공통 비교 구간 RMSE는 **1.068968**, Huber 점수는 **571.346334**였다. 평균 하나를 쓰는 기준선보다 큰 개선이었지만, 아직 Y를 거의 정확히 맞히는 수준은 아니었다.

### 3.3 필수 요건인 피처 선택

입력 열이 많다고 반드시 좋은 것은 아니다. 그래서 모델 B에서 각 원본 열을 검증 데이터 안에서 섞어 점수가 얼마나 나빠지는지 확인했다. 한 열을 섞을 때는 그 열에서 만든 파생 피처도 함께 다시 계산했다. 이를 그룹 단위 순열 중요도로 사용했다. 중요도가 높을수록 그 열을 흐트러뜨렸을 때 예측이 크게 나빠진다는 뜻이다.

Velocity, Altitude, X_Position의 Huber 증가량은 각각 약 1,324,399, 1,323,450, 715,861이었다. 나머지 네 열의 증가량은 각각 1보다 작았다. 그러나 이 세 열을 미리 정답으로 고정하지 않았다. 중요도 순서대로 원본 열을 1개부터 7개까지 늘려 실제로 다시 학습하고 3겹 교차검증에서 비교했다.

| 선택한 원본 열 수 | 교차검증 평균 RMSE | 교차검증 평균 Huber |
|---:|---:|---:|
| 1 | 297.391847 | 1,325,515.148380 |
| 2 | 129.291748 | 550,561.851565 |
| 3 | **1.082793** | **586.257193** |
| 4 | 1.083784 | 587.331180 |
| 5 | 1.084496 | 588.103500 |
| 6 | 1.085974 | 589.706690 |
| 7 | 1.085992 | 589.727555 |

세 열을 사용할 때 교차검증 Huber가 가장 낮았다. 최종 선택 열은 Velocity, Altitude, X_Position이다. 피처를 줄여 개발 구간 전체에 다시 학습한 모델 B의 공통 비교 구간 RMSE는 **1.068628**, Huber는 **570.982438**이었다. 전체 일곱 열 모델보다 개선 폭은 작지만, 제외한 열이 파생 피처를 통해 다시 들어오지 않도록 처리했고 실제 피처 선택 실험의 결과를 반영했다. 이 제출 파일의 Kaggle 공개 점수는 **570.14827**이었다. 해당 제출 당시의 정확한 공개 순위는 보관된 화면만으로 확인하지 못해 기재하지 않는다.

### 3.4 데이터 안의 규칙을 다시 살펴본 이유

모델 B에서 세 입력이 압도적으로 중요했고, 특히 X_Position은 Y_Position과 같은 위치 변수였다. 그래서 X와 Y를 평면상의 점으로 보고 두 값의 관계를 계산했다. 학습 데이터에서 중심 (630, 550), 반지름 420인 원을 가정했을 때, 636,363행의 원 방정식 잔차 최대값은 약 2.33×10⁻¹⁰이었다. 또 Y가 중심 550보다 위에 있는지 아래에 있는지는 Velocity와 Altitude의 부호 곱과 전 행에서 일치했다. 이는 단순한 상관관계보다 훨씬 강한, 데이터 생성 규칙의 후보였다.

X만 알면 같은 원 위에 가능한 Y가 위와 아래 두 개이므로, 가지 선택이 필요하다. Velocity×Altitude가 양수이면 위쪽, 음수이면 아래쪽을 선택하는 규칙을 적용했다. 다만 이는 학습 데이터에서 확인한 규칙이다. 숨겨진 시험 정답에도 항상 성립한다고 미리 가정하지 않고, 모델 B를 독립된 비교 대상으로 유지했다.

### 3.5 추정한 원을 사용하는 모델 A

모델 A에서는 개발 구간의 X·Y를 이용해 원의 중심과 반지름을 최소제곱으로 추정했다. 그다음 X로 제곱근 안의 값을 계산하고, Velocity×Altitude의 부호로 위아래 가지를 결정했다. 제곱근 안이 음수이거나 부호를 결정할 수 없으면 중심 Y를 대체값으로 사용하고 예외를 기록하도록 했다. 현재 제공된 비교·시험 데이터에는 이 예외가 없었다.

추정한 중심은 대략 (629.9999999996, 550.0000000004), 반지름은 약 420.0000000000이었다. 공통 비교 구간의 RMSE는 **1.0883×10⁻⁷**, Huber 점수는 **5.9221×10⁻¹²**로 모델 B보다 크게 낮았다. 처음 제출한 추정형 모델 A의 Kaggle 공개 점수는 화면상 **0.00000**이었다. 앞서 보관한 리더보드 화면에서는 팀 dku_22962가 **공개 8위**로 보였다.

### 3.6 마지막 수정: 추정값을 정수값과 비교

수학적 수식을 적용했는데도 모델 A의 오차가 정확히 0이 아닌 점을 검토했다. 원의 추정 중심은 630과 550에서 아주 조금 벗어났고, 원의 좌우 끝에서는 작은 중심 오차가 제곱근 결과에 더 크게 반영될 수 있다. 따라서 기존 모델 A를 덮어쓰지 않고, 학습 데이터에서 반복 확인한 정수값 (630, 550, 420)을 적용한 **추가 후보**를 만들었다.

같은 127,273행 비교 구간에서 정수값 후보의 RMSE는 **1.2492×10⁻¹¹**, Huber 점수는 **7.8022×10⁻²⁰**, 최대 절대오차는 약 **3.62×10⁻⁹**이었다. 기존 추정형 A보다 내부 오차가 더 작았다. 최종 추가 제출 CSV는 63,000행, 열은 Satellite_ID와 Y_Position 순서이며 샘플 ID·행 순서와 일치하고 예측값은 모두 유한함을 확인했다.

이 추가 후보인 submission_model_a_exact.csv의 Kaggle 공개 점수도 화면에는 **0.00000**으로 표시됐다. 최신 첨부 화면에서 팀 dku_22962는 **공개 7위**였다. 이전 화면의 8위와 최신 화면의 7위는 각각 그 시점의 순위다. 표시 점수가 같으므로, 순위 변동이 미세한 내부 점수 개선 때문인지 다른 팀의 제출·순위 변화 때문인지는 화면만으로 단정하지 않는다.

## 4. 단계별 성능과 실제 제출 결과

| 단계 | 공통 비교 구간 RMSE ↓ | 공통 비교 구간 Huber ↓ | Kaggle 공개 점수 ↓ | 화면에서 확인한 공개 순위 |
|---|---:|---:|---:|---|
| 개발 구간 Y 평균 기준선 | 296.444468 | 1,321,767.551843 | 제출하지 않음 | 해당 없음 |
| 모델 B, 원본 열 7개 | 1.068968 | 571.346334 | 별도 제출하지 않음 | 해당 없음 |
| 모델 B, 선택 열 3개 | 1.068628 | 570.982438 | 570.14827 | 확인 자료 없음 |
| 모델 A, 원 매개변수 추정 | 1.0883×10⁻⁷ | 5.9221×10⁻¹² | 0.00000 | 이전 화면에서 8위 |
| 모델 A 추가 후보, 정수값 적용 | 1.2492×10⁻¹¹ | 7.8022×10⁻²⁰ | 0.00000 | 최신 화면에서 7위 |

표의 RMSE와 왼쪽 Huber는 정답이 있는 학습 CSV의 공통 비교 구간 결과다. Kaggle 공개 점수와 순위는 숨겨진 시험 정답 일부를 이용한 별개 결과다. 0.00000은 소수점 다섯 자리 표시이며 수학적으로 오차가 정확히 0이라는 증명은 아니다. 처음 두 수식 모델의 공개 점수가 같은 자릿수로 보이더라도 내부 비교 결과는 다르다.

## 5. 리더보드 캡처

### 5.1 최종 추가 후보의 공개 7위

아래 사진은 저장소 최신 커밋에 포함된 원본 리더보드 캡처다. submission_model_a_exact.csv의 공개 점수 0.00000과 팀 dku_22962의 공개 7위가 함께 표시되어 있다.

![정수값 수식 후보의 Kaggle 공개 리더보드 7위](../outputs/Leaderboard_ranking.png)

### 5.2 이전 제출 기록

다음 두 사진은 이 보고서에 첨부한 사용자 제공 제출 화면이다. 첫 사진은 기존 모델 A 제출 당시 공개 8위와 이후 모델 B의 최근 제출 점수 570.14827을 보여준다. 두 번째 사진은 모델 A와 B 제출 파일 각각의 공개 점수를 보여준다.

![기존 모델 A 제출 당시 공개 8위와 모델 B의 최근 점수](../outputs/Leaderboard_rank8.png)

![모델 A와 B의 Kaggle 제출 점수](../outputs/Kaggle_submissions_initial.png)

## 6. 결과 해석과 한계

가장 큰 성능 변화는 모델 종류를 더 복잡하게 바꾸는 데서 나오지 않았다. 일반 머신러닝 모델 B의 예측 오차를 확인하고 중요한 열을 선택한 뒤, 입력과 정답의 구조를 조사한 결과 원의 관계를 발견했다. 이후 원 매개변수 추정에서 생긴 매우 작은 수치 오차를 검토하고 정수값 후보를 별도 시험한 것이 마지막 개선이었다. 피처 선택은 과제의 필수 요건으로 모델 B에 실제 적용했고, 제외한 열이 성능에 주는 영향도 일곱 후보로 비교했다.

모델 A의 원 관계는 이 보고서를 쓰기 전에 학습 CSV를 탐색하며 발견했다. 따라서 내부 20% 비교 점수를 완전히 새로운 데이터에서 처음 발견한 가설의 독립 검증이라고 주장하지 않는다. Kaggle 공개 점수는 시험 데이터 일부에 대한 실제 평가지만 개별 정답을 알려주지 않으며, 비공개 63%에서 같은 성능을 보장하지 않는다. 공개 7위 역시 최종 1위라는 뜻이 아니다. 이 보고서의 분석과 코드 작성에는 AI 도구의 보조를 활용했으며, 기재한 실험값과 제출 결과는 실제 데이터·산출물·화면과 대조했다.

## 7. 재현 방법과 첨부 소스코드 안내

실행 환경은 Python 3.14.7이며 필요한 패키지의 정확한 버전은 프로젝트의 requirements.lock에 기록되어 있다. 원본 CSV 세 개를 orbit-predict-satellite-position 폴더에 두고 프로젝트 최상위 폴더에서 다음을 실행한다. Windows PowerShell 기준이다.

~~~~powershell
python -m pip install -r .\requirements.lock
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m orbit_predict.cli --run
~~~~

이 명령은 기준선, 모델 B의 피처 선택·학습, 추정형 모델 A의 검증, 두 기본 제출 CSV와 docs/model-report.md 초안을 다시 만든다. **마지막 정수값 제출 CSV는 아래 부록의 코드를 담은 scripts/make_exact_submission.py를 별도로 실행해야 재현된다.** 실행 명령은 python .\scripts\make_exact_submission.py 이다. 모든 코드를 한 파일로 옮기면 가져오기 관계가 깨질 수 있으므로 src/orbit_predict 폴더 구조를 유지한다.

이 보고서 뒤에 실제 예측에 사용한 Python 모듈 전체와 정수값 후보 생성 코드를 첨부했다. 자동 테스트 코드는 저장소의 tests/dev 및 tests/review에 별도로 있다. 최신 GitHub Actions 실행 결과는 https://github.com/junsikpower/deeplearning_contest/actions/runs/35732411851 에서 확인할 수 있다.

## 부록 A. 최종 정수값 제출 CSV 생성 코드

아래 코드는 프로젝트 최상위 폴더에서 PYTHONPATH를 src로 설정한 뒤 실행한다. 기존 기본 제출 CSV를 바꾸지 않고 outputs/submission_model_a_exact.csv를 생성한다.

~~~~python
from pathlib import Path

from orbit_predict.data import DatasetPaths, load_datasets, make_submission
from orbit_predict.model_a import CircleParameters, predict_circle


def main() -> None:
    data = load_datasets(
        DatasetPaths(
            train=Path("orbit-predict-satellite-position/Train_DS..csv"),
            test=Path("orbit-predict-satellite-position/Test_DS..csv"),
            sample_submission=Path(
                "orbit-predict-satellite-position/Sample_Submission_DS..csv"
            ),
        )
    )
    parameters = CircleParameters(
        center_x=630.0,
        center_y=550.0,
        radius=420.0,
        coefficient_d=-1260.0,
        coefficient_e=-1100.0,
        coefficient_f=523000.0,
    )
    result = predict_circle(data.test, parameters)
    if result.exceptions:
        raise RuntimeError(f"모델 A 예외 행 {len(result.exceptions)}개를 확인해야 합니다.")
    submission = make_submission(data.sample_submission, result.predictions)
    path = Path("outputs/submission_model_a_exact.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    print(f"{path}: {len(submission):,}행")


if __name__ == "__main__":
    main()
~~~~

## 부록 B. 모델 학습·예측 전체 소스코드

아래에는 재현에 필요한 src/orbit_predict의 Python 모듈을 폴더 경로와 함께 첨부한다.

### src/orbit_predict/__init__.py

````python
"""Orbit satellite-position prediction package."""

__all__ = ["__version__"]
__version__ = "0.1.0"
````

### src/orbit_predict/constants.py

````python
"""Shared data-contract constants for the Orbit project."""

from __future__ import annotations

ID_COLUMN = "Satellite_ID"
TARGET_COLUMN = "Y_Position"
INPUT_COLUMNS = (
    "X_Position",
    "Velocity",
    "Altitude",
    "Fuel_Level",
    "Signal_Strength",
    "Battery_Temp",
    "Solar_Exposure",
)
SUBMISSION_COLUMNS = (ID_COLUMN, TARGET_COLUMN)

EXPECTED_TRAIN_ROWS = 636_363
EXPECTED_TEST_ROWS = 63_000
EXPECTED_SAMPLE_ROWS = 63_000

RANDOM_SEED = 42
COMPARISON_FRACTION = 0.20
SELECTION_SAMPLE_SIZE = 100_000
CV_FOLDS = 3
````

### src/orbit_predict/data.py

````python
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
````

### src/orbit_predict/split.py

````python
"""Deterministic development/comparison split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import COMPARISON_FRACTION, RANDOM_SEED


@dataclass(frozen=True)
class DataSplit:
    """Row positions for the development and final comparison partitions."""

    development_indices: np.ndarray
    comparison_indices: np.ndarray
    seed: int
    comparison_fraction: float


def make_split(
    row_count: int,
    *,
    seed: int = RANDOM_SEED,
    comparison_fraction: float = COMPARISON_FRACTION,
) -> DataSplit:
    """Apply the PRD's ``default_rng(42).permutation`` split exactly."""

    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if not 0 < comparison_fraction < 1:
        raise ValueError("comparison_fraction must be between 0 and 1")
    permutation = np.random.default_rng(seed).permutation(row_count)
    comparison_count = int(np.ceil(comparison_fraction * row_count))
    comparison_indices = permutation[:comparison_count]
    development_indices = permutation[comparison_count:]
    return DataSplit(
        development_indices=development_indices,
        comparison_indices=comparison_indices,
        seed=seed,
        comparison_fraction=comparison_fraction,
    )
````

### src/orbit_predict/metrics.py

````python
"""Competition metrics used by both model paths."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _as_float_arrays(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {actual.shape} and {predicted.shape}"
        )
    if actual.ndim != 1:
        actual = actual.reshape(-1)
        predicted = predicted.reshape(-1)
    if actual.size == 0:
        raise ValueError("metrics require at least one row")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("metrics require finite y_true and y_pred values")
    return actual, predicted


def rmse(y_true: object, y_pred: object) -> float:
    """Return root mean squared error."""

    actual, predicted = _as_float_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def competition_huber_score(y_true: object, y_pred: object, delta: float = 5.0) -> float:
    """Return the competition's scaled Huber score.

    The competition applies 0.5 * error^2 up to ``delta`` and
    ``delta * (abs(error) - delta / 2)`` above it, then averages and scales
    the value by 1,000.
    """

    if delta <= 0:
        raise ValueError("delta must be positive")
    actual, predicted = _as_float_arrays(y_true, y_pred)
    absolute_error = np.abs(actual - predicted)
    loss = np.where(
        absolute_error <= delta,
        0.5 * np.square(actual - predicted),
        delta * (absolute_error - delta / 2.0),
    )
    return float(np.mean(loss) * 1_000.0)


def metric_dict(y_true: object, y_pred: object) -> dict[str, float]:
    """Compute every metric reported for a prediction stage."""

    actual, predicted = _as_float_arrays(y_true, y_pred)
    return {
        "rmse": rmse(actual, predicted),
        "huber_score": competition_huber_score(actual, predicted),
        "max_absolute_error": float(np.max(np.abs(actual - predicted))),
    }


def merge_metric_dicts(*metrics: Mapping[str, float]) -> dict[str, float]:
    """Merge metric dictionaries while preserving a stable key order."""

    merged: dict[str, float] = {}
    for metric in metrics:
        merged.update({str(key): float(value) for key, value in metric.items()})
    return merged
````

### src/orbit_predict/features.py

````python
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
````

### src/orbit_predict/model_b.py

````python
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
````

### src/orbit_predict/model_a.py

````python
"""Model A: the circle relation discovered in the training data."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .constants import INPUT_COLUMNS


class ModelAError(ValueError):
    """Raised when the circle model cannot be estimated from valid rows."""


@dataclass(frozen=True)
class CircleParameters:
    """Estimated parameters for x^2 + y^2 + D*x + E*y + F = 0."""

    center_x: float
    center_y: float
    radius: float
    coefficient_d: float
    coefficient_e: float
    coefficient_f: float

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class ModelAPrediction:
    """Predictions plus explicit fallback records for model A."""

    predictions: np.ndarray
    exceptions: tuple[dict[str, object], ...]


def fit_circle(frame: pd.DataFrame) -> CircleParameters:
    """Estimate a circle from the supplied x/y columns by least squares."""

    required = ("X_Position", "Y_Position")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ModelAError(f"model A requires columns: {', '.join(missing)}")
    x = pd.to_numeric(frame["X_Position"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(frame["Y_Position"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        raise ModelAError("model A needs at least three finite (X_Position, Y_Position) rows")
    design = np.column_stack((x[valid], y[valid], np.ones(int(valid.sum()), dtype=np.float64)))
    rhs = -(np.square(x[valid]) + np.square(y[valid]))
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ModelAError("circle least-squares estimation failed") from exc
    if rank < 3:
        raise ModelAError("circle least-squares design is rank deficient")
    coefficient_d, coefficient_e, coefficient_f = coefficients
    center_x = -coefficient_d / 2.0
    center_y = -coefficient_e / 2.0
    radius_squared = center_x**2 + center_y**2 - coefficient_f
    if not np.isfinite(radius_squared) or radius_squared < 0:
        raise ModelAError("circle estimation produced an invalid radius")
    return CircleParameters(
        center_x=float(center_x),
        center_y=float(center_y),
        radius=float(np.sqrt(radius_squared)),
        coefficient_d=float(coefficient_d),
        coefficient_e=float(coefficient_e),
        coefficient_f=float(coefficient_f),
    )


def predict_circle(
    frame: pd.DataFrame,
    parameters: CircleParameters,
) -> ModelAPrediction:
    """Predict using the sign branch and center-y fallback required by the PRD."""

    missing = [column for column in ("X_Position", "Velocity", "Altitude") if column not in frame.columns]
    if missing:
        raise ModelAError(f"model A requires columns: {', '.join(missing)}")
    x = pd.to_numeric(frame["X_Position"], errors="coerce").to_numpy(dtype=np.float64)
    velocity = pd.to_numeric(frame["Velocity"], errors="coerce").to_numpy(dtype=np.float64)
    altitude = pd.to_numeric(frame["Altitude"], errors="coerce").to_numpy(dtype=np.float64)
    product = velocity * altitude
    q = parameters.radius**2 - np.square(x - parameters.center_x)
    valid_q = np.isfinite(q) & (q >= 0)
    valid_product = np.isfinite(product) & (product != 0)
    valid = valid_q & valid_product & np.isfinite(x)

    predictions = np.full(len(frame), parameters.center_y, dtype=np.float64)
    predictions[valid] = parameters.center_y + np.where(
        product[valid] > 0,
        np.sqrt(q[valid]),
        -np.sqrt(q[valid]),
    )

    ids = frame["Satellite_ID"].to_numpy() if "Satellite_ID" in frame.columns else np.arange(len(frame))
    exceptions: list[dict[str, object]] = []
    for position in np.flatnonzero(~valid):
        reasons: list[str] = []
        if not valid_q[position]:
            reasons.append("q_negative")
        if not valid_product[position]:
            reasons.append("velocity_altitude_product_zero")
        if not np.isfinite(x[position]):
            reasons.append("x_nonfinite")
        exceptions.append(
            {
                "row_position": int(position),
                "Satellite_ID": ids[position].item() if hasattr(ids[position], "item") else ids[position],
                "reason": "+".join(reasons),
            }
        )
    if not np.isfinite(predictions).all():
        raise ModelAError("model A generated a non-finite prediction")
    return ModelAPrediction(predictions=predictions, exceptions=tuple(exceptions))


def model_a_input_columns() -> tuple[str, ...]:
    """Document the model A inputs without leaking them into model B."""

    return (INPUT_COLUMNS[0], INPUT_COLUMNS[1], INPUT_COLUMNS[2], "Y_Position")
````

### src/orbit_predict/experiment.py

````python
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
````

### src/orbit_predict/reporting.py

````python
"""JSON and Markdown report writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def json_ready(value: Any) -> Any:
    """Convert NumPy/Pandas-like scalar containers into JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_ready(value.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write an indented UTF-8 JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fmt_metric(value: object) -> str:
    if value is None:
        return "미기록"
    return f"{float(value):,.6f}"


def write_markdown_report(path: Path, result: Any) -> None:
    """Write the handoff report from an experiment result object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = result.metrics
    metric_rows = []
    for stage in metrics:
        metric_rows.append(
            "| {stage} | {rmse} | {huber} | {maximum} | {note} |".format(
                stage=stage["stage"],
                rmse=_fmt_metric(stage.get("rmse")),
                huber=_fmt_metric(stage.get("huber_score")),
                maximum=_fmt_metric(stage.get("max_absolute_error")),
                note=stage.get("note", ""),
            )
        )
    selection = result.feature_selection
    importance_rows = "\n".join(
        f"| {item['source_column']} | {_fmt_metric(item['mean_huber_increase'])} |"
        for item in selection["importance"]
    )
    candidate_rows = "\n".join(
        f"| {candidate['source_columns']} | {_fmt_metric(candidate['mean_rmse'])} | "
        f"{_fmt_metric(candidate['mean_huber_score'])} | {candidate['feature_count']} | "
        f"{candidate['estimated_matrix_bytes']:,} |"
        for candidate in selection["candidates"]
    )
    output_rows = "\n".join(f"- `{key}`: `{value}`" for key, value in result.outputs.items())
    report = f"""# Orbit — Y_Position 예측 보고서 초안

이 문서는 제공된 학습·시험 CSV에 대해 실행 가능한 두 예측 경로를 기록한 초안이다. Kaggle 제출·업로드와 리더보드 캡처는 사용자가 수행하며, 이 문서에는 실제 Kaggle 점수를 기재하지 않는다.

## 1. 입력 데이터와 재현 조건

- 학습 행: `{result.train_rows:,}`, 시험 행: `{result.test_rows:,}`, 샘플 제출 행: `{result.sample_rows:,}`
- 입력 열: `{', '.join(result.input_columns)}`
- 정답 열: `Y_Position`; `Satellite_ID`는 모델 입력에서 제외하고 제출 행 연결에만 사용했다.
- 분할: `np.random.default_rng(42).permutation(N)` 후 앞쪽 `ceil(0.2 × N)`을 최종 비교 구간으로 사용했다.
- 비교 구간 행: `{result.comparison_rows:,}`; 개발 구간 행: `{result.development_rows:,}`
- Model B 설정: `{result.model_b_config}`

학습·시험 입력의 범위와 분위수는 `artifacts/data_distribution.json`에 저장했다. 데이터 분포 차이는 시험 정답을 확인한 사실이 아니므로 성능 보증으로 해석하지 않는다.

## 2. 모델 경로

### 모델 A — 원 관계 후보

개발 구간의 `(X_Position, Y_Position)`에 `x² + y² + D·x + E·y + F = 0`을 최소제곱으로 적합했다. `q = r² - (x-a)²`가 음수이거나 `Velocity × Altitude = 0`이면 `center_y`를 대체값으로 사용하고 행·사유를 기록했다. 개발 구간 추정값은 다음과 같다.

```text
{result.model_a_parameters}
```

개발 비교 구간 예외 건수: `{result.model_a_development_exception_count}`; 전체 학습으로 다시 적합한 시험 예측 예외 건수: `{result.model_a_test_exception_count}`.

### 모델 B — 공식과 독립적인 머신러닝

Model B는 일곱 원본 입력에서만 일반 피처를 만들었다. 각 학습 접힘에서 원본 열별 중앙값을 추정하고, 원본·절댓값·제곱·부호·모든 쌍별 곱·쌍별 부호 조합을 생성했다. 원 중심·반지름·Model A 예측값·정답 변환은 입력으로 사용하지 않았다.

피처 선택 표본 크기: `{selection['sample_size']:,}`. 고정된 3겹 교차검증에서 한 원본 열과 그 열에서 파생된 피처 전체를 함께 섞어 Huber 증가량을 계산했다.

| 원본 열 | 평균 Huber 증가량 |
|---|---:|
{importance_rows}

중요도 순위: `{', '.join(selection['ranked_columns'])}`

| 후보 원본 열 | CV 평균 RMSE | CV 평균 Huber | 피처 수 | 표본 행렬 예상 메모리(bytes) |
|---|---:|---:|---:|---:|
{candidate_rows}

- Huber 기준 최종 k: `{selection['selected_k']}` (`{', '.join(selection['selected_columns'])}`)
- 최고 축소 후보 k: `{selection['best_reduced_k']}` (`{', '.join(selection['best_reduced_columns'])}`)
- 최종 k가 7이면 전체 입력을 유지한 것이며, 축소 후보 실험 결과는 별도 행으로 남겼다.

## 3. 공통 최종 비교 구간 성능

| 단계 | RMSE | Huber 점수 | 최대 절대오차 | 비고 |
|---|---:|---:|---:|---|
{chr(10).join(metric_rows)}

RMSE와 Huber 점수는 서로 다른 목적의 지표이며, 대회 선택 기준은 PRD에 따라 Huber 점수다. X 구간과 위·아래 가지별 요약은 `artifacts/error_segments.json`에서 확인한다.

## 4. 제출 파일과 사용자 업로드

{output_rows}

두 CSV는 `Satellite_ID,Y_Position` 순서이며 샘플 제출물의 ID·행 순서를 검증했다. 사용자가 Kaggle에서 직접 제출하며, 다음 순서로 진행한다.

1. Kaggle 대회의 Submit Predictions 화면을 연다.
2. `submission_model_a.csv`를 업로드하고 제출 이름에 Model A를 표시한다.
3. 같은 방식으로 `submission_model_b.csv`를 업로드하고 Model B를 표시한다.
4. 각 공개 점수와 리더보드 캡처를 전달한다. 현재 문서에는 점수와 순위를 아직 기재하지 않는다.

## 5. 한계와 해석 주의

- Model A의 원 관계는 학습 CSV를 탐색해 발견한 가설이므로 이 내부 비교를 최초 독립 발견 검증으로 설명하지 않는다.
- Model B도 동일한 대회 학습 분포에서 선택되므로 비공개 시험 성능을 보장하지 않는다.
- 두 모델의 공통 비교 결과는 내부 검증 자료이며, 어느 한 모델을 시험 정답이라고 자동 단정하지 않는다.
- 공개 리더보드 점수와 최종 비공개 순위는 사용자가 제출한 뒤 전달한 자료만으로 보완한다.
- 배깅·스태킹은 이번 필수 개발 경로에 포함하지 않았다.
"""
    path.write_text(report, encoding="utf-8")
````

### src/orbit_predict/cli.py

````python
"""Command-line entry point for data checks and the full experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import DatasetPaths, load_datasets
from .experiment import run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orbit satellite-position prediction pipeline")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("orbit-predict-satellite-position/Train_DS..csv"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("orbit-predict-satellite-position/Test_DS..csv"),
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("orbit-predict-satellite-position/Sample_Submission_DS..csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--report", type=Path, default=Path("docs/model-report.md"))
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="validate the three supplied CSVs without fitting models",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="run both models, selection, validation, and submission generation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = DatasetPaths(
        train=args.train,
        test=args.test,
        sample_submission=args.sample_submission,
    )
    if args.check_inputs == args.run:
        raise SystemExit("choose exactly one of --check-inputs or --run")
    if args.check_inputs:
        datasets = load_datasets(paths)
        print(
            f"input contract OK: train={len(datasets.train):,}, "
            f"test={len(datasets.test):,}, sample={len(datasets.sample_submission):,}"
        )
        return 0
    result = run_experiment(
        paths,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        report_path=args.report,
    )
    print(f"experiment complete: {len(result.metrics)} comparison stages")
    for output in result.outputs.values():
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````
