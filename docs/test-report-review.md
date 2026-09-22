# Orbit — 독립 테스트 설계 산출물 보고서

## 1. 개요

- **목적:** 기획서(PRD)의 의도대로 구현이 온전히 이뤄졌는지, 최신 커밋 코드에 구현 결함이나 회귀 오류가 없는지 독립적으로 검증하기 위한 테스트 설계 산출물이다.
- **실행 환경 및 도구:** `pytest` (Makefile `test-review` 타겟 기반)
- **CI 산출물 경로:** `reports/review.xml` (JUnit XML 형식)
- **테스트 코드 위치:** `tests/review/` (`tests/dev/`의 개발자 테스트 코드는 일체 참조하지 않음)
- **테스트 파일 구성:**
  - `tests/review/test_prd_functional.py`: PRD 기능 요구사항(FR-01 ~ FR-07) 검증 (19개 케이스)
  - `tests/review/test_prd_rules_and_edges.py`: 비즈니스 규칙(BR), 예외/에지 케이스(EC), 비기능 요구사항(NFR), 제외/기술 제약(OOS, TC) 검증 (15개 케이스)
  - `tests/review/test_prd_acceptance.py`: 인수 기준(AC 13.1, AC 13.2 통합, AC 13.3 사용자 시나리오) 검증 (7개 케이스)
  - `tests/review/test_code_regression_int.py`: 최신 커밋 코드 기반 회귀/경계값/내부 계약(`INT`) 검증 (42개 케이스)
  - **총 테스트 케이스:** 83개

---

## 2. 테스트 케이스 설계 목록 및 근거

### 2.1 PRD 기반 테스트 케이스 (총 41개)

| 테스트 함수명 | 근거 PRD 조항 | 검증 목적 및 내용 |
|---|---|---|
| `test_FR01_학습데이터_행수_컬럼_타입_검증` | FR-01 | 학습 데이터의 636,363행, ID 정수형, 7개 입력열 및 Y_Position 정답열 스키마 검증 |
| `test_FR01_시험데이터_및_샘플제출_행수_스키마_검증` | FR-01 | 시험 데이터(63,000행) 및 샘플 제출물(63,000행)의 스키마와 행 수 일치 검증 |
| `test_FR01_Satellite_ID_예측변수_제외_검증` | FR-01 | Satellite_ID가 모델 입력 변수(INPUT_COLUMNS)에서 제외되었는지 검증 |
| `test_FR02_고정_시드_분할_재현성_및_비율_검증` | FR-02 | default_rng(42) 기반 20% 비교 구간(127,273행) 및 80% 개발 구간(509,090행) 분할 재현성 및 무중복성 검증 |
| `test_FR02_단계별_지표_RMSE_Huber_최대오차_계산_검증` | FR-02 | 공통 검증 지표(RMSE, 대회 Huber 점수, 최대 절대오차)의 정상 산출 검증 |
| `test_FR03_원_매개변수_최소제곱_추정_검증` | FR-03 | 최소제곱법을 통한 원 중심 (a, b) 및 반지름 r의 기하학적 추정 정확성 검증 |
| `test_FR03_부호규칙_기반_예측_및_결과_유한성_검증` | FR-03 | Velocity * Altitude 부호에 따른 상·하 분기 예측 및 유한값 보장 검증 |
| `test_FR03_예외조건_대체값_b_반환_및_예외기록_검증` | FR-03 | q < 0 또는 곱이 0일 때 대체값 b 반환 및 예외 사유 기록 검증 |
| `test_FR03_모델A_학습데이터전체_재추정_검증` | FR-03 | 최종 제출 파일 생성 시 학습 데이터 전체에서 원 중심 (630, 550) 및 반지름 420 재추정 검증 |
| `test_FR04_모델B_파생피처_생성_원공식_독립성_검증` | FR-04 | 원 공식(630, 550, 420) 없이 일반 파생 피처(abs, square, sign, pair-product, pair-sign) 생성 및 독립성 검증 |
| `test_FR04_모델B_학습접힘_중앙값_전처리_격리_검증` | FR-04 | 결측치 대체 중앙값이 학습 접힘 데이터에서만 추정되고 평가 데이터에 누수되지 않는지 검증 |
| `test_FR04_모델B_학습_및_예측_동작_검증` | FR-04 | ModelB 인스턴스의 fit 및 predict 정상 수행과 유한한 예측값 산출 검증 |
| `test_FR04_모델B_타깃열_및_Satellite_ID_입력차단_검증` | FR-04 | ModelB가 Y_Position(타깃) 또는 Satellite_ID(비승인) 컬럼을 source_columns로 수신할 때의 유입 차단 검증 |
| `test_FR05_그룹단위_순열중요도_파생피처_동시반영_검증` | FR-05 | 원본 열 순열 시 파생 피처 동시 재계산을 통한 그룹 단위 순열 중요도(Huber 증가량) 산출 검증 |
| `test_FR05_상위_k개_후보_평가_및_최적k_선택_검증` | FR-05 | k=1..7 후보 평가를 통한 최적 k 및 최고 축소 후보(k<=6) 도출 검증 |
| `test_FR05_동점시_적은_원본열_후보선택_검증` | FR-05 | 평균 교차검증 Huber 점수가 동일한 경우 원본 열 개수가 더 적은 후보를 우선 선택하는 타이 브레이킹 규칙 검증 |
| `test_FR06_두_제출파일_행수_컬럼_순서_일치_검증` | FR-06 | outputs의 모델 A, 모델 B 제출 CSV 파일의 63,000행 및 샘플 대비 ID/순서 일치 검증 |
| `test_FR06_사용자_업로드_안내_문서_존재_검증` | FR-06 | 보고서 내 Kaggle 업로드 순서 안내 명시 및 개발AI의 직접 제출 완료 허위 주장 부재 검증 |
| `test_FR07_보고서_단계별_지표_및_피처선택_요약_검증` | FR-07 | 보고서 내 단계별 지표(baseline, model_a, model_b 3종) 및 중요도 순위 표 포함 검증 |
| `test_BR01_대회_Huber_점수_공식_검증` | BR-01 | 대회 Huber 점수 공식(오차 5 이하 0.5*e^2, 5 초과 5*(|e|-2.5), 평균*1000) 일치 검증 |
| `test_BR02_공개_비공개_비율_명시_및_비공개성능_미단정_검증` | BR-02 | 공개(37%) 및 비공개(63%) 비율 명시 및 비공개 시험 성능 미단정 검증 |
| `test_BR03_Kaggle_제출_역할분담_보고서_기재_검증` | BR-03 | 사용자가 제출을 수행하고 초안 상태로 인계됨을 보고서에 명시하였는지 검증 |
| `test_EC01_q음수_또는_부호0_대체값_b_및_예외로그_검증` | EC-01 | q < 0 및 Velocity * Altitude == 0 조건에서의 대체값 b 처리와 예외 로그(q_negative, zero) 기록 검증 |
| `test_EC02_모델B_결측치_중앙값_대체_검증` | EC-02 | 입력 결측치(NaN, inf) 발생 시 학습 중앙값 대체 및 유한 예측 보장 검증 |
| `test_EC02_제출파일_비유한값_및_행불일치_검출_검증` | EC-02 | 제출 파일 생성 시 NaN/Inf 포함 또는 ID 불일치 발생 시 DataContractError 검출 검증 |
| `test_EC03_구간별_오차_및_입력분포_비교_산출물_검증` | EC-03 | error_segments.json 및 data_distribution.json 산출물의 무결성 및 구간별 요약 통계 검증 |
| `test_EC04_사용자_미제출시_초안_상태_유지_검증` | EC-04 | 사용자 실제 제출 전 보고서가 초안 상태이며 Kaggle 점수가 미확인으로 유지되는지 검증 |
| `test_NFR01_데이터분할_재현성_검증` | NFR-01 | make_split의 시드 42 기반 다회 실행 시 완벽한 재현성 검증 |
| `test_NFR02_전처리_추정치_학습데이터_격리_검증` | NFR-02 | transform 수행 시 학습 추정치(medians_) 불변 및 데이터 누수 방지 검증 |
| `test_NFR03_보고서_비공개_정답_확정표현_부재_검증` | NFR-03 | 보고서 내 "1위 확정", "만점 보장" 등 비공개 정답에 대한 허위·확정 표현 부재 검증 |
| `test_OOS01_배깅_스태킹_미포함_검증` | 3.2 Out of Scope | 배깅(Bagging) 및 스태킹(Stacking) 앙상블 미포함 검증 |
| `test_OOS02_Kaggle_직접_업로드_로직_미포함_검증` | 3.2 Out of Scope | 개발AI의 Kaggle API 로그인 및 직접 업로드 코드 부재 검증 |
| `test_TC01_모델B_모델A_수식_미사용_검증` | 12.1 Technical Constraints | 모델 B 구현체 내에 모델 A의 원 수식 매개변수 및 예측값 미사용 검증 |
| `test_TC02_피처선택_선택전후_성능_기록_검증` | 12.1 Technical Constraints | feature_selection.json 내 피처 선택 전후(k=7 및 축소 k) 성능 지표 기록 검증 |
| `test_AC13_1_입력데이터_계약_및_스키마_검증` | AC 13.1 | 학습/시험/샘플 CSV 데이터 계약 및 스키마 검증 |
| `test_AC13_1_모델A_모델B_독립실행_및_동일구간_평가_검증` | AC 13.1 | 모델 A와 B의 별도 코드 경로 및 동일 비교 구간 지표 산출 검증 |
| `test_AC13_1_모델B_피처선택_축소후보_실제평가_검증` | AC 13.1 | 그룹 단위 순열 중요도 및 축소 후보 실제 학습/비교 검증 |
| `test_AC13_1_두_제출파일_완전성_및_샘플정합성_검증` | AC 13.1 | 두 제출 CSV의 63,000행 완전성 및 샘플 제출물 정합성 검증 |
| `test_AC13_2_전체_실험_파이프라인_통합_실행_검증` | AC 13.2 | 데이터 로드 -> 분할 -> 모델 A/B 학습 -> 평가 -> 제출 파일/산출물 생성 전체 파이프라인 통합 검증 |
| `test_AC13_3_사용자_보고서_이해_및_산출물_검토_시나리오` | AC 13.3 | 사용자가 보고서를 열어 두 모델의 차이, 피처 선택, 성능 지표를 검토하는 흐름 (조작 간 시간 경과 포함) |
| `test_AC13_3_사용자_제출파일_확인_및_업로드_준비_시나리오` | AC 13.3 | 사용자가 두 제출 CSV를 확인하고 Kaggle 업로드 안내를 확인하는 흐름 (조작 간 시간 경과 포함) |

---

### 2.2 최신 커밋 코드 기반 테스트 케이스 (`INT` 접두어, 총 42개)

| 테스트 함수명 | 근거 변경 파일 및 코드 경로 | 검증 목적 및 내용 |
|---|---|---|
| `test_INT_GeneralFeatureBuilder_빈_소스컬럼_예외처리` | `src/orbit_predict/features.py:28` | source_columns가 비어있을 때 ValueError 발생 검증 |
| `test_INT_GeneralFeatureBuilder_중복_소스컬럼_예외처리` | `src/orbit_predict/features.py:32` | source_columns에 중복 컬럼 존재 시 ValueError 발생 검증 |
| `test_INT_GeneralFeatureBuilder_fit_호출전_transform_예외처리` | `src/orbit_predict/features.py:48` | fit() 전 transform() 호출 시 FeatureTransformError 발생 검증 |
| `test_INT_GeneralFeatureBuilder_fit_호출전_feature_names_접근_예외처리` | `src/orbit_predict/features.py:104` | fit() 전 feature_names 프로퍼티 접근 시 FeatureTransformError 발생 검증 |
| `test_INT_GeneralFeatureBuilder_누락컬럼_변환시_예외처리` | `src/orbit_predict/features.py:38` | transform 대상에 필수 source_column 누락 시 FeatureTransformError 발생 검증 |
| `test_INT_GeneralFeatureBuilder_단일컬럼_피처개수_및_행렬_검증` | `src/orbit_predict/features.py:55` | source_columns=1일 때 feature_count=4(단항 피처만) 및 쌍 피처 0개 동작 검증 |
| `test_INT_fit_circle_필수컬럼_누락시_예외처리` | `src/orbit_predict/model_a.py:44` | frame에 X_Position 또는 Y_Position 누락 시 ModelAError 발생 검증 |
| `test_INT_fit_circle_유효행_부족시_예외처리` | `src/orbit_predict/model_a.py:50` | 유효한 (x, y) 행이 3개 미만일 때 ModelAError 발생 검증 |
| `test_INT_fit_circle_공선점_입력시_랭크부족_예외처리` | `src/orbit_predict/model_a.py:58` | 일직선상의 점들 입력으로 rank deficient 발생 시 ModelAError 처리 검증 |
| `test_INT_predict_circle_필수컬럼_누락시_예외처리` | `src/orbit_predict/model_a.py:82` | X_Position, Velocity, Altitude 중 누락 시 ModelAError 발생 검증 |
| `test_INT_predict_circle_Satellite_ID_없는경우_정상동작` | `src/orbit_predict/model_a.py:101` | frame에 Satellite_ID가 없는 경우 row_position을 기본 식별자로 대체 처리 검증 |
| `test_INT_ModelB_생성자_비승인_소스컬럼_예외처리` | `src/orbit_predict/model_b.py:58` | ModelB 생성 시 INPUT_COLUMNS에 없는 비승인 컬럼 지정 시 ValueError 발생 검증 |
| `test_INT_ModelB_생성자_중복_소스컬럼_예외처리` | `src/orbit_predict/model_b.py:58` | ModelB 생성 시 중복된 컬럼 지정 시 ValueError 발생 검증 |
| `test_INT_ModelB_생성자_빈_소스컬럼_예외처리` | `src/orbit_predict/model_b.py:58` | ModelB 생성 시 빈 컬럼 전달 시 ValueError 발생 검증 |
| `test_INT_ModelB_생성자_리스트입력시_튜플변환_검증` | `src/orbit_predict/model_b.py:58` | ModelB 생성 시 리스트 입력이 tuple 타입으로 변환 및 보존되는지 검증 |
| `test_INT_ModelB_미승인_소스컬럼_예외처리` | `src/orbit_predict/model_b.py:154` | _validate_source_columns 함수 단위 비승인 컬럼 예외 검증 |
| `test_INT_ModelB_fit_전_predict_호출시_예외처리` | `src/orbit_predict/model_b.py:88` | fit() 전 predict() 호출 시 RuntimeError 발생 검증 |
| `test_INT_ModelB_타깃길이_불일치_예외처리` | `src/orbit_predict/model_b.py:67` | frame과 target의 행 수 불일치 시 ValueError 발생 검증 |
| `test_INT_ModelB_비유한_타깃_예외처리` | `src/orbit_predict/model_b.py:71` | target에 NaN 또는 Inf 포함 시 ValueError 발생 검증 |
| `test_INT_ModelB_빈_학습데이터_fit시_예외처리` | `src/orbit_predict/model_b.py:69` | 0행의 빈 학습 데이터 전달 시 ValueError 발생 검증 |
| `test_INT_ModelB_fit_전_transform_호출시_예외처리` | `src/orbit_predict/model_b.py:82` | fit() 전 transform() 호출 시 RuntimeError 발생 검증 |
| `test_INT_ModelB_fit_전_feature_names_접근시_예외처리` | `src/orbit_predict/model_b.py:96` | fit() 전 feature_names 프로퍼티 접근 시 RuntimeError 발생 검증 |
| `test_INT_ModelB_make_folds_행수부족_예외처리` | `src/orbit_predict/model_b.py:163` | row_count < cv_folds인 경우 ValueError 발생 검증 |
| `test_INT_ModelB_cross_validate_candidate_타깃길이_불일치_예외처리` | `src/orbit_predict/model_b.py:183` | cross_validate_candidate에서 frame과 target의 길이 불일치 시 ValueError 발생 검증 |
| `test_INT_ModelB_select_features_타깃길이_불일치_예외처리` | `src/orbit_predict/model_b.py:259` | select_features에서 development_frame과 target의 길이 불일치 시 ValueError 발생 검증 |
| `test_INT_ModelB_select_features_표본수_CV접힘미달_예외처리` | `src/orbit_predict/model_b.py:262` | sample_size < cv_folds인 경우 ValueError 발생 검증 |
| `test_INT_metrics_빈배열_입력시_예외처리` | `src/orbit_predict/metrics.py:20` | rmse, competition_huber_score 등에 빈 배열 입력 시 ValueError 발생 검증 |
| `test_INT_metrics_형상불일치_예외처리` | `src/orbit_predict/metrics.py:13` | y_true와 y_pred 형상 불일치 시 ValueError 발생 검증 |
| `test_INT_metrics_비유한값_입력시_예외처리` | `src/orbit_predict/metrics.py:22` | y_true 또는 y_pred에 NaN/Inf 입력 시 ValueError 발생 검증 |
| `test_INT_competition_huber_score_delta_음수_예외처리` | `src/orbit_predict/metrics.py:42` | delta <= 0 입력 시 ValueError 발생 검증 |
| `test_INT_data_validate_train_frame_컬럼불일치_예외처리` | `src/orbit_predict/data.py:48` | 학습 데이터 컬럼 순서/이름 불일치 시 DataContractError 발생 검증 |
| `test_INT_data_validate_train_frame_중복ID_예외처리` | `src/orbit_predict/data.py:76` | 학습 데이터에 중복 Satellite_ID 존재 시 DataContractError 발생 검증 |
| `test_INT_data_validate_test_sample_alignment_불일치_예외처리` | `src/orbit_predict/data.py:137` | test와 sample_submission의 ID 불일치 시 DataContractError 발생 검증 |
| `test_INT_data_make_submission_예측값_길이불일치_예외처리` | `src/orbit_predict/data.py:238` | 샘플 행 수와 예측값 개수 불일치 시 DataContractError 발생 검증 |
| `test_INT_data_validate_sample_submission_컬럼불일치_예외처리` | `src/orbit_predict/data.py:118` | sample_submission 컬럼 불일치 시 DataContractError 발생 검증 |
| `test_INT_data_validate_train_frame_비숫자_타깃_예외처리` | `src/orbit_predict/data.py:57` | train_frame의 Y_Position에 비숫자 값 존재 시 DataContractError 발생 검증 |
| `test_INT_data_validate_train_frame_소수점_ID_예외처리` | `src/orbit_predict/data.py:75` | train_frame의 Satellite_ID가 실(소수)수일 때 DataContractError 발생 검증 |
| `test_INT_data_load_datasets_존재하지않는_파일_예외처리` | `src/orbit_predict/data.py:152` | 입력 파일 경로 부재 시 FileNotFoundError 발생 검증 |
| `test_INT_cli_인자_상호배타_예외처리` | `src/orbit_predict/cli.py:53` | --check-inputs와 --run 인자 상호배타 위반 시 SystemExit 발생 검증 |
| `test_INT_cli_check_inputs_정상실행_검증` | `src/orbit_predict/cli.py:54` | --check-inputs 플래그 단독 실행 시 반환값 0 검증 |
| `test_INT_split_make_split_음수행_또는_잘못된_비율_예외처리` | `src/orbit_predict/split.py:30` | row_count <= 0 또는 fraction 범위 오류 시 ValueError 발생 검증 |
| `test_INT_reporting_json_ready_특수타입_직렬화_검증` | `src/orbit_predict/reporting.py:10` | numpy scalar, nan, inf, 중첩 딕셔너리/리스트 JSON 안전 직렬화 검증 |

---

## 3. 미작성 항목 및 사유

- **미작성 항목:** 없음
- **사유:** PRD의 모든 기능 요구사항, 비즈니스 규칙, 에러/에지 케이스, 비기능 요구사항, 인수 기준 및 최신 커밋 코드의 모든 내부 모듈/경계값에 대한 테스트 코드가 누락 없이 작성되었습니다.

---

## 4. 외부 서비스 실호출 대체 및 미검증 항목

| 대상 항목 | 대체 방식 | 상태 | 미검증 사유 |
|---|---|---|---|
| Kaggle 대회 실제 로그인 및 파일 업로드 | 제출 CSV 파일 계약 검증 및 업로드 안내문 검증 | 미검증 | CI 클린 환경에서의 거짓 실패 방지 및 사용자 직접 수행 제약(PRD 3.2, 12.1) |
| 실제 리더보드 점수 및 캡처 획득 | 보고서 초안 상태 및 미확인 표기 검증 | 미검증 | 실제 제출은 사용자만 수행 가능하며 대회 플랫폼 접속 불가(PRD 1.3, 11) |
