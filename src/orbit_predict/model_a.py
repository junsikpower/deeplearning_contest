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
