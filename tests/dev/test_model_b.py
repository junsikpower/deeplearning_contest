from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from orbit_predict.constants import INPUT_COLUMNS
from orbit_predict.features import GeneralFeatureBuilder
from orbit_predict.model_b import ModelB, select_features


def test_FR04_일반피처모델B_학습과유한예측(small_input_frame, small_model_config):
    frame, target = small_input_frame
    model = ModelB((INPUT_COLUMNS[0], INPUT_COLUMNS[1], INPUT_COLUMNS[2]), small_model_config)
    model.fit(frame, target)
    predictions = model.predict(frame.iloc[:5])
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()
    assert len(model.feature_names) == 3 * 4 + 3 * 2


def test_EC02_결측과비숫자값_학습접힘중앙값으로처리(small_input_frame, small_model_config):
    frame, target = small_input_frame
    train = frame.iloc[:20].copy()
    train[INPUT_COLUMNS[1]] = train[INPUT_COLUMNS[1]].astype(object)
    train.loc[0, INPUT_COLUMNS[0]] = np.nan
    train.loc[1, INPUT_COLUMNS[1]] = "bad"
    model = ModelB((INPUT_COLUMNS[0], INPUT_COLUMNS[1]), small_model_config)
    model.fit(train, target[:20])
    predictions = model.predict(frame.iloc[20:])
    assert np.isfinite(predictions).all()


def test_FR05_그룹순열중요도와k후보_Huber기준선택(small_input_frame, small_model_config):
    frame, target = small_input_frame
    selected = select_features(
        frame,
        target,
        config=small_model_config,
        source_columns=INPUT_COLUMNS[:3],
    )
    assert len(selected.importance) == 3
    assert len(selected.candidates) == 3
    assert 1 <= selected.selected_k <= 3
    assert 1 <= selected.best_reduced_k <= 2
    assert all(len(item["fold_huber_increases"]) == 3 for item in selected.importance)


def test_FR04_모델B_모델A공식상수와예측값을참조하지않음():
    source = inspect.getsource(__import__("orbit_predict.model_b", fromlist=["model_b"]))
    assert "model_a" not in source.lower()
    assert "630" not in source
    assert "550" not in source
    assert "420" not in source


def test_INT_피처빌더_선택열에따라파생피처수일관성():
    frame = pd.DataFrame({"X_Position": [1.0, 2.0], "Velocity": [3.0, 4.0]})
    builder = GeneralFeatureBuilder(("X_Position", "Velocity"))
    matrix = builder.fit_transform(frame)
    assert matrix.shape == (2, 2 * 4 + 2)
    assert matrix.dtype == np.float32
