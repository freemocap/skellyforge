"""Critical damping must be critical damping, and must be framerate independent.

Spec: ``freemocap/docs/streaming-compatibility/14-engine-testing-strategy.md`` section 5;
behaviour defined in ``12-standard-human-model.md`` (section "Critical damping").

These tests distinguish a genuine second-order critically damped system from the
first-order exponential lag it is easy to mistake for one. The two differ in ways
that matter:

* a first-order lag has no velocity state, so it cannot be framerate independent
  when parameterized by a per-frame blend factor;
* a first-order lag never overshoots, but neither does critical damping, so
  "no overshoot" alone proves nothing — it must be paired with a settling-time
  assertion.

All quaternion literals are ``wxyz``; identity is ``(1, 0, 0, 0)``.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from skellyforge.kinematics.critically_damped_orientation import (
    CriticallyDampedOrientationState,
    advance_critically_damped_orientation,
)
from skellyforge.kinematics.quaternion_math import RotationQuaternion

ANGLE_TOLERANCE_RADIANS = 1e-9


def rotation_about_axis(
    *, axis: tuple[float, float, float], degrees: float
) -> RotationQuaternion:
    axis_array: NDArray[np.float64] = np.asarray(axis, dtype=np.float64)
    axis_array = axis_array / np.linalg.norm(axis_array)
    half_angle = np.deg2rad(degrees) / 2.0
    sine = float(np.sin(half_angle))
    return RotationQuaternion(
        w=float(np.cos(half_angle)),
        x=float(axis_array[0] * sine),
        y=float(axis_array[1] * sine),
        z=float(axis_array[2] * sine),
    )


def angle_between(
    first: RotationQuaternion, second: RotationQuaternion
) -> float:
    """Shortest-arc angle in radians between two orientations."""
    _, angle = (first * second.conjugate()).to_axis_angle()
    return float(angle)


def run_step_response(
    *,
    target: RotationQuaternion,
    time_constant_seconds: float,
    timestep_seconds: float,
    duration_seconds: float,
) -> list[tuple[float, RotationQuaternion]]:
    """Step from identity to *target*, sampling ``(time, orientation)`` per frame."""
    state = CriticallyDampedOrientationState.at_rest(RotationQuaternion.identity())
    samples: list[tuple[float, RotationQuaternion]] = [(0.0, state.orientation)]

    frame_count = int(round(duration_seconds / timestep_seconds))
    for frame_index in range(1, frame_count + 1):
        state = advance_critically_damped_orientation(
            state=state,
            target_orientation=target,
            time_constant_seconds=time_constant_seconds,
            timestep_seconds=timestep_seconds,
        )
        samples.append((frame_index * timestep_seconds, state.orientation))
    return samples


# ── Exponential / logarithmic map round-trip ──────────────────────────


@pytest.mark.parametrize(
    ("axis", "degrees"),
    [
        ((0.0, 0.0, 1.0), 90.0),
        ((1.0, 0.0, 0.0), 179.0),
        ((1.0, 1.0, 1.0), 45.0),
        ((0.0, 1.0, 0.0), 1e-6),
    ],
)
def test_rotation_vector_round_trip(
    axis: tuple[float, float, float], degrees: float
) -> None:
    """``exp(log(q)) == q`` — the filter's tangent-space transport must be lossless."""
    original = rotation_about_axis(axis=axis, degrees=degrees)
    recovered = RotationQuaternion.from_rotation_vector(original.to_rotation_vector())
    assert angle_between(original, recovered) < ANGLE_TOLERANCE_RADIANS


def test_rotation_vector_takes_the_shortest_arc() -> None:
    """A 350 deg rotation is a -10 deg rotation; the log map must say so.

    Without shortest-arc resolution the filter smooths the long way round.
    """
    long_way = rotation_about_axis(axis=(0.0, 0.0, 1.0), degrees=350.0)
    magnitude = float(np.linalg.norm(long_way.to_rotation_vector()))
    assert magnitude == pytest.approx(np.deg2rad(10.0), abs=1e-9)


# ── Framerate independence — the load-bearing property ────────────────


@pytest.mark.parametrize("frames_per_second", [30.0, 60.0, 120.0, 240.0])
def test_settling_time_is_framerate_independent(frames_per_second: float) -> None:
    """The same motion at any framerate must settle in the same number of *seconds*.

    This is the property a per-frame blend factor cannot have, and the reason the
    parameter is a time constant. Reference: a critically damped step response has
    error envelope ``(1 + t/tau) * exp(-t/tau)``, so at ``t = 5*tau`` the residual
    is ``6 * exp(-5) ~= 4%`` of the initial error.
    """
    time_constant_seconds = 0.1
    target = rotation_about_axis(axis=(0.0, 0.0, 1.0), degrees=60.0)
    initial_error_radians = np.deg2rad(60.0)

    samples = run_step_response(
        target=target,
        time_constant_seconds=time_constant_seconds,
        timestep_seconds=1.0 / frames_per_second,
        duration_seconds=5.0 * time_constant_seconds,
    )

    _, final_orientation = samples[-1]
    residual_radians = angle_between(final_orientation, target)
    residual_fraction = residual_radians / initial_error_radians

    expected_fraction = 6.0 * float(np.exp(-5.0))
    assert residual_fraction == pytest.approx(expected_fraction, rel=0.02), (
        f"at {frames_per_second} fps the residual after 5*tau was "
        f"{residual_fraction:.4f}, expected ~{expected_fraction:.4f}"
    )


def test_settling_agrees_across_framerates_to_high_precision() -> None:
    """Directly compare 30 fps against 240 fps rather than each against theory.

    The exact closed-form step means these should agree to numerical precision,
    not merely to a loose tolerance.
    """
    time_constant_seconds = 0.08
    target = rotation_about_axis(axis=(1.0, 1.0, 0.0), degrees=45.0)
    duration_seconds = 0.4

    _, slow_final = run_step_response(
        target=target,
        time_constant_seconds=time_constant_seconds,
        timestep_seconds=1.0 / 30.0,
        duration_seconds=duration_seconds,
    )[-1]
    _, fast_final = run_step_response(
        target=target,
        time_constant_seconds=time_constant_seconds,
        timestep_seconds=1.0 / 240.0,
        duration_seconds=duration_seconds,
    )[-1]

    assert angle_between(slow_final, fast_final) < 1e-6


# ── Critically damped, not under- or over-damped ──────────────────────


def test_step_response_does_not_overshoot() -> None:
    """Critical damping approaches the target monotonically, never past it."""
    time_constant_seconds = 0.05
    target = rotation_about_axis(axis=(0.0, 1.0, 0.0), degrees=90.0)

    samples = run_step_response(
        target=target,
        time_constant_seconds=time_constant_seconds,
        timestep_seconds=1.0 / 120.0,
        duration_seconds=10.0 * time_constant_seconds,
    )

    errors = [angle_between(orientation, target) for _, orientation in samples]
    for previous_error, current_error in zip(errors, errors[1:]):
        assert current_error <= previous_error + 1e-12, (
            "error increased between frames — the response overshot, so the system "
            "is under-damped"
        )


def test_step_response_matches_the_analytic_envelope() -> None:
    """Pin the actual curve, not just its endpoints.

    A first-order lag decays as ``exp(-t/tau)``; critical damping decays as
    ``(1 + t/tau) * exp(-t/tau)``. Asserting the envelope at several points is what
    separates the two — an endpoint-only test cannot.
    """
    time_constant_seconds = 0.1
    initial_error_radians = np.deg2rad(80.0)
    target = rotation_about_axis(axis=(0.0, 0.0, 1.0), degrees=80.0)

    samples = run_step_response(
        target=target,
        time_constant_seconds=time_constant_seconds,
        timestep_seconds=1.0 / 480.0,
        duration_seconds=4.0 * time_constant_seconds,
    )

    for elapsed_seconds, orientation in samples[1:]:
        normalized_time = elapsed_seconds / time_constant_seconds
        expected_radians = (
            initial_error_radians
            * (1.0 + normalized_time)
            * float(np.exp(-normalized_time))
        )
        actual_radians = angle_between(orientation, target)
        assert actual_radians == pytest.approx(expected_radians, rel=1e-6, abs=1e-9), (
            f"at t={elapsed_seconds:.4f}s the error was {actual_radians:.6f} rad, "
            f"expected {expected_radians:.6f} rad from the critically damped envelope"
        )


def test_first_order_lag_would_fail_the_envelope_check() -> None:
    """Guard the guard: confirm the envelope test can tell the two filters apart.

    If a first-order lag satisfied the same envelope, the test above would be
    vacuous.
    """
    normalized_time = 2.0
    critically_damped = (1.0 + normalized_time) * np.exp(-normalized_time)
    first_order_lag = np.exp(-normalized_time)
    assert abs(critically_damped - first_order_lag) > 0.1


# ── Edge cases ────────────────────────────────────────────────────────


def test_already_at_target_stays_put() -> None:
    target = rotation_about_axis(axis=(0.0, 0.0, 1.0), degrees=30.0)
    state = CriticallyDampedOrientationState.at_rest(target)

    advanced = advance_critically_damped_orientation(
        state=state,
        target_orientation=target,
        time_constant_seconds=0.05,
        timestep_seconds=1.0 / 60.0,
    )

    assert angle_between(advanced.orientation, target) < ANGLE_TOLERANCE_RADIANS
    assert np.allclose(
        advanced.angular_velocity_radians_per_second, np.zeros(3), atol=1e-12
    )


def test_long_gap_lands_on_target_with_zero_velocity() -> None:
    """A large ``dt`` self-heals — no gap threshold needed.

    ``decay`` falls exponentially while ``B * dt`` grows only linearly, so both the
    error and its rate go to zero. The filter re-initializes itself.
    """
    target = rotation_about_axis(axis=(1.0, 0.0, 0.0), degrees=120.0)
    state = CriticallyDampedOrientationState.at_rest(RotationQuaternion.identity())

    advanced = advance_critically_damped_orientation(
        state=state,
        target_orientation=target,
        time_constant_seconds=0.05,
        timestep_seconds=10.0,  # 200x the time constant
    )

    assert angle_between(advanced.orientation, target) < 1e-9
    assert np.allclose(
        advanced.angular_velocity_radians_per_second, np.zeros(3), atol=1e-9
    )


@pytest.mark.parametrize("bad_time_constant", [0.0, -0.1])
def test_non_positive_time_constant_raises(bad_time_constant: float) -> None:
    with pytest.raises(ValueError, match="time_constant_seconds"):
        advance_critically_damped_orientation(
            state=CriticallyDampedOrientationState.at_rest(
                RotationQuaternion.identity()
            ),
            target_orientation=RotationQuaternion.identity(),
            time_constant_seconds=bad_time_constant,
            timestep_seconds=1.0 / 60.0,
        )


@pytest.mark.parametrize("bad_timestep", [0.0, -0.016])
def test_non_advancing_clock_raises(bad_timestep: float) -> None:
    """A non-advancing clock is a caller error — fail loudly, do not paper over it."""
    with pytest.raises(ValueError, match="timestep_seconds"):
        advance_critically_damped_orientation(
            state=CriticallyDampedOrientationState.at_rest(
                RotationQuaternion.identity()
            ),
            target_orientation=RotationQuaternion.identity(),
            time_constant_seconds=0.05,
            timestep_seconds=bad_timestep,
        )


def test_smaller_time_constant_converges_faster() -> None:
    """The parameter must act in the documented direction."""
    target = rotation_about_axis(axis=(0.0, 1.0, 0.0), degrees=45.0)
    duration_seconds = 0.1
    timestep_seconds = 1.0 / 120.0

    _, snappy = run_step_response(
        target=target,
        time_constant_seconds=0.02,
        timestep_seconds=timestep_seconds,
        duration_seconds=duration_seconds,
    )[-1]
    _, sluggish = run_step_response(
        target=target,
        time_constant_seconds=0.20,
        timestep_seconds=timestep_seconds,
        duration_seconds=duration_seconds,
    )[-1]

    assert angle_between(snappy, target) < angle_between(sluggish, target)
