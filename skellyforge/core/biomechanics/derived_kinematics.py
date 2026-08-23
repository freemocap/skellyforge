"""Derived kinematics: the whole-body center of mass's velocity and acceleration.

The ground-reference measures need the CoM's velocity (the extrapolated center of mass)
and acceleration (centroidal angular momentum terms). These are recovered from the CoM
trajectory by finite differences - central in the interior, one-sided at the ends - rather
than from segment velocities, so this module depends only on a time series of CoM
positions.
"""

from __future__ import annotations

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_TIME_DELTA_SECONDS
from skellyforge.type_overloads import FloatArray


def center_of_mass_velocity(
    *, positions: FloatArray, timestamps: FloatArray
) -> FloatArray:
    """The CoM velocity at each sample, by finite differences of position.

    Args:
        positions: the CoM world position at each sample, shape (N, 3).
        timestamps: the time of each sample in seconds, shape (N,), strictly increasing.

    Returns:
        The CoM velocity at each sample, shape (N, 3).

    Raises:
        ValueError: fewer than two samples, or timestamps that are not strictly increasing.
    """
    positions = np.asarray(positions, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    _validate_series(positions=positions, timestamps=timestamps, minimum_samples=2)
    return _differentiate(values=positions, timestamps=timestamps)


def center_of_mass_acceleration(
    *, velocities: FloatArray, timestamps: FloatArray
) -> FloatArray:
    """The CoM acceleration at each sample, by finite differences of velocity.

    Args:
        velocities: the CoM velocity at each sample, shape (N, 3).
        timestamps: the time of each sample in seconds, shape (N,), strictly increasing.

    Returns:
        The CoM acceleration at each sample, shape (N, 3).
    """
    velocities = np.asarray(velocities, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    _validate_series(positions=velocities, timestamps=timestamps, minimum_samples=2)
    return _differentiate(values=velocities, timestamps=timestamps)


def _differentiate(*, values: FloatArray, timestamps: FloatArray) -> FloatArray:
    """Central differences in the interior, one-sided at the ends."""
    count = values.shape[0]
    derivative = np.empty_like(values)
    # Forward difference at the first sample.
    derivative[0] = (values[1] - values[0]) / (timestamps[1] - timestamps[0])
    # Backward difference at the last sample.
    derivative[-1] = (values[-1] - values[-2]) / (timestamps[-1] - timestamps[-2])
    if count > 2:
        # Central difference for the interior.
        delta = (timestamps[2:] - timestamps[:-2])[:, np.newaxis]
        derivative[1:-1] = (values[2:] - values[:-2]) / delta
    return derivative


def _validate_series(
    *, positions: FloatArray, timestamps: FloatArray, minimum_samples: int
) -> None:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions must have shape (N, 3) - got {positions.shape}"
        )
    if timestamps.ndim != 1:
        raise ValueError(f"timestamps must be 1-dimensional - got {timestamps.shape}")
    if positions.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"positions and timestamps must have the same length - got "
            f"{positions.shape[0]} and {timestamps.shape[0]}"
        )
    if positions.shape[0] < minimum_samples:
        raise ValueError(
            f"need at least {minimum_samples} samples to differentiate - got "
            f"{positions.shape[0]}"
        )
    deltas = np.diff(timestamps)
    if np.any(deltas <= MINIMUM_TIME_DELTA_SECONDS):
        raise ValueError("timestamps must be strictly increasing")
