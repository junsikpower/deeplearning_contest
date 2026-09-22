from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orbit_predict.constants import INPUT_COLUMNS
from orbit_predict.data import (
    DataContractError,
    make_submission,
    summarize_distribution,
    validate_test_sample_alignment,
    validate_test_frame,
    validate_train_frame,
)


def test_FR01_학습시험샘플_입력열과타깃열구분(circle_frames):
    train, test, sample = circle_frames
    validate_train_frame(train, expected_rows=None)
    validate_test_frame(test, expected_rows=None)
    assert "Y_Position" in train.columns
    assert "Y_Position" not in test.columns
    assert tuple(sample.columns) == ("Satellite_ID", "Y_Position")


def test_FR01_입력분포_주요분위수와범위기록(circle_frames):
    train, _, _ = circle_frames
    summary = summarize_distribution(train, INPUT_COLUMNS)
    assert set(summary) == set(INPUT_COLUMNS)
    assert summary["X_Position"]["count"] == len(train)
    assert summary["X_Position"]["q50"] == pytest.approx(train["X_Position"].median())


def test_EC02_비숫자입력_데이터계약오류(circle_frames):
    train, _, _ = circle_frames
    broken = train.copy()
    broken["Velocity"] = broken["Velocity"].astype(object)
    broken.loc[0, "Velocity"] = "not-a-number"
    with pytest.raises(DataContractError, match="Velocity"):
        validate_train_frame(broken, expected_rows=None)


def test_FR06_제출생성_샘플ID와행순서보존(circle_frames):
    _, _, sample = circle_frames
    predictions = np.linspace(1.0, 2.0, len(sample))
    submission = make_submission(sample, predictions)
    assert tuple(submission.columns) == ("Satellite_ID", "Y_Position")
    assert submission["Satellite_ID"].tolist() == sample["Satellite_ID"].tolist()
    assert np.isfinite(submission["Y_Position"]).all()


def test_FR01_시험샘플ID_행순위불일치차단(circle_frames):
    _, test, sample = circle_frames
    misaligned = sample.iloc[::-1].reset_index(drop=True)
    with pytest.raises(DataContractError, match="not aligned"):
        validate_test_sample_alignment(test, misaligned)


def test_EC02_제출예측개수불일치_오류발생(circle_frames):
    _, _, sample = circle_frames
    with pytest.raises(DataContractError, match="prediction count"):
        make_submission(sample, [1.0])
