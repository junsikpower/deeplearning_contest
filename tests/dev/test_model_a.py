from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orbit_predict.model_a import fit_circle, predict_circle


def test_FR03_원수식모델_개발자료로중심반지름추정(circle_frames):
    train, _, _ = circle_frames
    parameters = fit_circle(train)
    assert parameters.center_x == pytest.approx(10.0, abs=1e-8)
    assert parameters.center_y == pytest.approx(20.0, abs=1e-8)
    assert parameters.radius == pytest.approx(5.0, abs=1e-8)


def test_FR03_부호가지_양수상단음수하단예측(circle_frames):
    train, _, _ = circle_frames
    parameters = fit_circle(train)
    result = predict_circle(train, parameters)
    assert np.max(np.abs(result.predictions - train["Y_Position"].to_numpy())) < 1e-7
    assert result.exceptions == ()


def test_EC01_q음수와곱0_중심Y대체값과사유기록(circle_frames):
    train, _, _ = circle_frames
    parameters = fit_circle(train)
    edge = pd.DataFrame(
        {
            "Satellite_ID": [9001, 9002],
            "X_Position": [100.0, 10.0],
            "Velocity": [2.0, 0.0],
            "Altitude": [1.0, 4.0],
        }
    )
    result = predict_circle(edge, parameters)
    assert result.predictions.tolist() == pytest.approx([parameters.center_y, parameters.center_y])
    assert len(result.exceptions) == 2
    assert "q_negative" in str(result.exceptions[0]["reason"])
    assert "velocity_altitude_product_zero" in str(result.exceptions[1]["reason"])


def test_EC02_모델A필수열누락_근거없는예측대신오류(circle_frames):
    train, _, _ = circle_frames
    parameters = fit_circle(train)
    with pytest.raises(ValueError, match="Altitude"):
        predict_circle(train.drop(columns=["Altitude"]), parameters)

