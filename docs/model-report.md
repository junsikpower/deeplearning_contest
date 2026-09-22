# Orbit — Y_Position 예측 보고서 초안

이 문서는 제공된 학습·시험 CSV에 대해 실행 가능한 두 예측 경로를 기록한 초안이다. Kaggle 제출·업로드와 리더보드 캡처는 사용자가 수행하며, 이 문서에는 실제 Kaggle 점수를 기재하지 않는다.

## 1. 입력 데이터와 재현 조건

- 학습 행: `636,363`, 시험 행: `63,000`, 샘플 제출 행: `63,000`
- 입력 열: `X_Position, Velocity, Altitude, Fuel_Level, Signal_Strength, Battery_Temp, Solar_Exposure`
- 정답 열: `Y_Position`; `Satellite_ID`는 모델 입력에서 제외하고 제출 행 연결에만 사용했다.
- 분할: `np.random.default_rng(42).permutation(N)` 후 앞쪽 `ceil(0.2 × N)`을 최종 비교 구간으로 사용했다.
- 비교 구간 행: `127,273`; 개발 구간 행: `509,090`
- Model B 설정: `{'max_iter': 120, 'learning_rate': 0.08, 'max_leaf_nodes': 31, 'min_samples_leaf': 40, 'l2_regularization': 1.0, 'random_state': 42, 'selection_sample_size': 100000, 'cv_folds': 3}`

학습·시험 입력의 범위와 분위수는 `artifacts/data_distribution.json`에 저장했다. 데이터 분포 차이는 시험 정답을 확인한 사실이 아니므로 성능 보증으로 해석하지 않는다.

## 2. 모델 경로

### 모델 A — 원 관계 후보

개발 구간의 `(X_Position, Y_Position)`에 `x² + y² + D·x + E·y + F = 0`을 최소제곱으로 적합했다. `q = r² - (x-a)²`가 음수이거나 `Velocity × Altitude = 0`이면 `center_y`를 대체값으로 사용하고 행·사유를 기록했다. 개발 구간 추정값은 다음과 같다.

```text
{'center_x': 629.9999999996215, 'center_y': 550.0000000004369, 'radius': 420.0000000000012, 'coefficient_d': -1259.999999999243, 'coefficient_e': -1100.0000000008738, 'coefficient_f': 523000.00000000274}
```

개발 비교 구간 예외 건수: `0`; 전체 학습으로 다시 적합한 시험 예측 예외 건수: `0`.

### 모델 B — 공식과 독립적인 머신러닝

Model B는 일곱 원본 입력에서만 일반 피처를 만들었다. 각 학습 접힘에서 원본 열별 중앙값을 추정하고, 원본·절댓값·제곱·부호·모든 쌍별 곱·쌍별 부호 조합을 생성했다. 원 중심·반지름·Model A 예측값·정답 변환은 입력으로 사용하지 않았다.

피처 선택 표본 크기: `100,000`. 고정된 3겹 교차검증에서 한 원본 열과 그 열에서 파생된 피처 전체를 함께 섞어 Huber 증가량을 계산했다.

| 원본 열 | 평균 Huber 증가량 |
|---|---:|
| X_Position | 715,860.967804 |
| Velocity | 1,324,398.594515 |
| Altitude | 1,323,449.695619 |
| Fuel_Level | 0.470864 |
| Signal_Strength | 0.064165 |
| Battery_Temp | 0.032117 |
| Solar_Exposure | 0.344127 |

중요도 순위: `Velocity, Altitude, X_Position, Fuel_Level, Solar_Exposure, Signal_Strength, Battery_Temp`

| 후보 원본 열 | CV 평균 RMSE | CV 평균 Huber | 피처 수 | 표본 행렬 예상 메모리(bytes) |
|---|---:|---:|---:|---:|
| ['Velocity'] | 297.391847 | 1,325,515.148380 | 4 | 1,600,000 |
| ['Velocity', 'Altitude'] | 129.291748 | 550,561.851565 | 10 | 4,000,000 |
| ['Velocity', 'Altitude', 'X_Position'] | 1.082793 | 586.257193 | 18 | 7,200,000 |
| ['Velocity', 'Altitude', 'X_Position', 'Fuel_Level'] | 1.083784 | 587.331180 | 28 | 11,200,000 |
| ['Velocity', 'Altitude', 'X_Position', 'Fuel_Level', 'Solar_Exposure'] | 1.084496 | 588.103500 | 40 | 16,000,000 |
| ['Velocity', 'Altitude', 'X_Position', 'Fuel_Level', 'Solar_Exposure', 'Signal_Strength'] | 1.085974 | 589.706690 | 54 | 21,600,000 |
| ['Velocity', 'Altitude', 'X_Position', 'Fuel_Level', 'Solar_Exposure', 'Signal_Strength', 'Battery_Temp'] | 1.085992 | 589.727555 | 70 | 28,000,000 |

- Huber 기준 최종 k: `3` (`Velocity, Altitude, X_Position`)
- 최고 축소 후보 k: `3` (`Velocity, Altitude, X_Position`)
- 최종 k가 7이면 전체 입력을 유지한 것이며, 축소 후보 실험 결과는 별도 행으로 남겼다.

## 3. 공통 최종 비교 구간 성능

| 단계 | RMSE | Huber 점수 | 최대 절대오차 | 비고 |
|---|---:|---:|---:|---|
| baseline_development_mean | 296.444468 | 1,321,767.551843 | 420.005119 |  |
| model_a | 0.000000 | 0.000000 | 0.000024 |  |
| model_b_all_inputs | 1.068968 | 571.346334 | 2.824112 | 전체 일곱 원본 입력 기준 |
| model_b_best_reduced | 1.068628 | 570.982438 | 2.816826 | 최고 축소 후보 k=3 |
| model_b_final | 1.068628 | 570.982438 | 2.816826 | 교차검증 선택 k=3 |

RMSE와 Huber 점수는 서로 다른 목적의 지표이며, 대회 선택 기준은 PRD에 따라 Huber 점수다. X 구간과 위·아래 가지별 요약은 `artifacts/error_segments.json`에서 확인한다.

## 4. 제출 파일과 사용자 업로드

- `model_a_submission`: `outputs\submission_model_a.csv`
- `model_b_submission`: `outputs\submission_model_b.csv`
- `model_a_exceptions`: `artifacts\model_a_exceptions.json`
- `feature_selection`: `artifacts\feature_selection.json`
- `data_distribution`: `artifacts\data_distribution.json`
- `error_segments`: `artifacts\error_segments.json`

두 CSV는 `Satellite_ID,Y_Position` 순서이며 샘플 제출물의 ID·행 순서를 검증했다. 사용자는 다음 순서로 Kaggle에서 직접 제출한다.

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
