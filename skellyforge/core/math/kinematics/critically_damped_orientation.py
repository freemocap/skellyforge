"""Critically damped temporal smoothing for orientations.

Used by the orientation solver's ``DAMPED_MINIMAL`` twist tier: when a segment's
twist reference is occluded or degenerate, the roll estimate is noisy, and the
solver holds it steady by filtering toward the raw solve rather than tracking it
frame-to-frame. Specification:
``freemocap/docs/streaming-compatibility/12-standard-human-model.md``
(section "Critical damping"); tests per
``freemocap/docs/streaming-compatibility/14-engine-testing-strategy.md`` section 5.

The system
----------
A second-order system with damping ratio exactly 1 — the fastest response that
does not overshoot. Writing the error as ``e`` (rotation vector from the target to
the filtered output) and its rate as ``v``::

    e'' + 2*omega*e' + omega**2 * e = 0,     omega = 1 / tau

``tau`` is the **time constant in seconds**, so behaviour is identical at any
framerate. A per-frame blend factor cannot express this: the same nominal factor
means a different time constant at 30, 60 and 120 fps, so a rig tuned on one
machine misbehaves on another.

Solved exactly, not approximated
--------------------------------
For a target held constant across a step of length ``dt``, the critically damped
solution is closed-form::

    B     = v + omega * e
    decay = exp(-omega * dt)
    e_new = (e + B * dt) * decay
    v_new = (v - omega * B * dt) * decay

This is the analytic solution, not the ``1/(1 + x + 0.48x^2 + 0.235x^3)`` series
used by game engines to avoid calling ``exp``. We are scientific software; one
``exp`` per segment per frame is not a cost worth trading correctness for.

Long gaps self-heal, so no special case is needed: as ``dt`` grows, ``decay``
falls exponentially while ``B * dt`` grows only linearly, so both ``e_new`` and
``v_new`` go to zero — the filter lands on the target with zero velocity, exactly
as if it had been re-initialized.

Orientations are not a vector space
-----------------------------------
Rotations live on SO(3), so the filter runs in the **tangent space** at the
current target: the error is a rotation vector (``log`` of the error quaternion),
integrated in R^3, then mapped back (``exp``). Re-linearizing about the target
every frame keeps the approximation tight in the regime this filter is for
(suppressing small, noisy twist deviations). For large errors the update remains
stable and convergent, but the path is not exactly geodesic — noted here so the
limitation is on the record rather than discovered later.

The double cover (``q`` and ``-q`` are the same rotation) is resolved by
``to_rotation_vector``, which always returns the shortest arc. Without that, the
filter would happily smooth the long way around.
"""

from dataclasses import dataclass

import numpy as np
from skellyforge.type_overloads import FloatArray

from skellyforge.kinematics.quaternion_math import RotationQuaternion


@dataclass(frozen=True)
class CriticallyDampedOrientationState:
    """One segment's filter state: the smoothed orientation and its rate.

    ``angular_velocity_radians_per_second`` is a rotation vector in the world
    tangent space — the rate of the *error*, not the segment's physical angular
    velocity.
    """

    orientation: RotationQuaternion
    angular_velocity_radians_per_second: FloatArray

    @classmethod
    def at_rest(cls, orientation: RotationQuaternion) -> "CriticallyDampedOrientationState":
        """Seed the filter at an orientation with zero velocity.

        Used on the first frame, where there is no history to damp against.
        """
        return cls(
            orientation=orientation,
            angular_velocity_radians_per_second=np.zeros(3, dtype=np.float64),
        )


def advance_critically_damped_orientation(
    *,
    state: CriticallyDampedOrientationState,
    target_orientation: RotationQuaternion,
    time_constant_seconds: float,
    timestep_seconds: float,
) -> CriticallyDampedOrientationState:
    """Advance the filter one step toward *target_orientation*.

    Parameters
    ----------
    state
        The previous step's orientation and error rate.
    target_orientation
        This frame's raw (unfiltered) solve.
    time_constant_seconds
        ``tau``. Larger means heavier smoothing and a slower approach. Framerate
        independent.
    timestep_seconds
        ``dt`` since the previous step. Must be strictly positive — a
        non-advancing clock is a caller error, not something to paper over.

    Returns
    -------
    CriticallyDampedOrientationState
        The filtered orientation and the updated error rate.

    Raises
    ------
    ValueError
        If ``time_constant_seconds`` or ``timestep_seconds`` is not positive.
    """
    if time_constant_seconds <= 0.0:
        raise ValueError(
            f"time_constant_seconds must be > 0, got {time_constant_seconds}"
        )
    if timestep_seconds <= 0.0:
        raise ValueError(
            f"timestep_seconds must be > 0, got {timestep_seconds}"
        )

    angular_frequency = 1.0 / time_constant_seconds

    # Error as a rotation vector in the world tangent space:
    #   orientation = exp(error) * target   =>   error = log(orientation * conj(target))
    error_quaternion = state.orientation * target_orientation.conjugate()
    error_vector = error_quaternion.to_rotation_vector()

    velocity_vector = np.asarray(
        state.angular_velocity_radians_per_second, dtype=np.float64
    )

    # Closed-form critically damped step (see module docstring).
    coefficient = velocity_vector + angular_frequency * error_vector
    decay = float(np.exp(-angular_frequency * timestep_seconds))
    advanced_error = (error_vector + coefficient * timestep_seconds) * decay
    advanced_velocity = (
        velocity_vector - angular_frequency * coefficient * timestep_seconds
    ) * decay

    advanced_orientation = (
        RotationQuaternion.from_rotation_vector(advanced_error) * target_orientation
    )

    return CriticallyDampedOrientationState(
        orientation=advanced_orientation,
        angular_velocity_radians_per_second=advanced_velocity,
    )
