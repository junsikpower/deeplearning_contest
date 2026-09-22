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

