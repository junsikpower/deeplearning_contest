from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orbit_predict.constants import INPUT_COLUMNS


@pytest.fixture
def circle_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    center_x, center_y, radius = 10.0, 20.0, 5.0
    x_values = np.linspace(5.5, 14.5, 30)
    upper = center_y + np.sqrt(radius**2 - np.square(x_values - center_x))
    lower = center_y - np.sqrt(radius**2 - np.square(x_values - center_x))
    y_values = np.where(np.arange(len(x_values)) % 2 == 0, upper, lower)
    velocity = np.where(np.arange(len(x_values)) % 2 == 0, 2.0, -2.0)
    altitude = np.ones(len(x_values))
    train = pd.DataFrame(
        {
            "Satellite_ID": np.arange(1000, 1030),
            "X_Position": x_values,
            "Velocity": velocity,
            "Altitude": altitude,
            "Fuel_Level": np.linspace(1.0, 3.0, 30),
            "Signal_Strength": np.linspace(-2.0, 2.0, 30),
            "Battery_Temp": np.linspace(10.0, 15.0, 30),
            "Solar_Exposure": np.linspace(3.0, 6.0, 30),
            "Y_Position": y_values,
        }
    )
    test = train.drop(columns=["Y_Position"]).iloc[:10].copy()
    test["Satellite_ID"] = np.arange(2000, 2010)
    sample = pd.DataFrame({"Satellite_ID": test["Satellite_ID"], "Y_Position": 63.0})
    return train, test, sample


@pytest.fixture
def small_model_config():
    from orbit_predict.model_b import ModelBConfig

    return ModelBConfig(
        max_iter=2,
        learning_rate=0.1,
        max_leaf_nodes=7,
        min_samples_leaf=1,
        l2_regularization=0.1,
        selection_sample_size=24,
        cv_folds=3,
    )


@pytest.fixture
def small_input_frame() -> tuple[pd.DataFrame, np.ndarray]:
    rows = 30
    values = np.arange(rows, dtype=np.float64)
    frame = pd.DataFrame(
        {
            column: values * (index + 1) / 10.0
            for index, column in enumerate(INPUT_COLUMNS)
        }
    )
    target = 2.0 * frame[INPUT_COLUMNS[0]].to_numpy() - frame[INPUT_COLUMNS[1]].to_numpy() + 0.5
    return frame, target

