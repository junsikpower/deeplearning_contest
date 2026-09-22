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
"""
    path.write_text(report, encoding="utf-8")
