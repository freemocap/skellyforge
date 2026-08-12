"""The solver must damp on every fallback path, and carry state per instance.

Spec: ``freemocap/docs/streaming-compatibility/14-engine-testing-strategy.md``
section 4 ("Fallback paths still damp") and section 5.

The defect these guard against: ``_solve_chain_resolved`` used to hand the damped
tier ``previous_world_quaternion=None`` on both of its fallback paths, so damping
was skipped in exactly the situation it exists for — an occluded twist source, or
a limb straightened into the singularity gate. The result popped precisely when it
was supposed to be steadiest.

All quaternion literals are ``wxyz``; identity is ``(1, 0, 0, 0)``.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from skellyforge.kinematics.orientation_solver import (
    FrameOrientationResult,
    solve_frame_orientations,
)
from skellyforge.kinematics.quaternion_math import RotationQuaternion
from skellyforge.skellymodels.standard_human.human_bones import (
    BoneReferenceGeometry,
    CoordinateFrameDefinition,
    HumanBone,
    TwistPolicy,
    TwistTier,
)
from skellyforge.skellymodels.standard_human.standard_human_model import StandardHuman

SEGMENT_LENGTH_MM = 100.0
TIME_CONSTANT_SECONDS = 0.1
FRAME_INTERVAL_SECONDS = 1.0 / 60.0


def build_chain_model(*, tier: TwistTier, twist_source: str | None = None) -> StandardHuman:
    """A 3-segment vertical chain whose middle segment uses the given twist tier.

    ``upper`` is the segment under test; ``middle`` and ``lower`` give it a distal
    joint and (for CHAIN_RESOLVED) a twist source.
    """
    segment_names = ("upper", "middle", "lower")
    bones: list[HumanBone] = []
    for index, segment_name in enumerate(segment_names):
        is_segment_under_test = index == 0
        bones.append(
            HumanBone(
                name=segment_name,
                parent=segment_names[index - 1] if index > 0 else None,
                required=True,
                reference_geometry=BoneReferenceGeometry(
                    proximal_joint_center=np.array(
                        [0.0, 0.0, index * SEGMENT_LENGTH_MM], dtype=np.float64
                    ),
                    distal_joint_center=np.array(
                        [0.0, 0.0, (index + 1) * SEGMENT_LENGTH_MM], dtype=np.float64
                    ),
                    coordinate_frame=CoordinateFrameDefinition(
                        exact_axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                        approximate_axis=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                    ),
                ),
                twist_policy=(
                    TwistPolicy(
                        tier=tier,
                        twist_source_bone=twist_source,
                        twist_time_constant_seconds=TIME_CONSTANT_SECONDS,
                    )
                    if is_segment_under_test
                    else TwistPolicy(
                        tier=TwistTier.FULL_FRAME,
                        twist_time_constant_seconds=TIME_CONSTANT_SECONDS,
                    )
                ),
            )
        )
    return StandardHuman(name="damping_chain", bones=bones, blendshape_channels=[])


def _segment_direction(*, tilt_degrees: float) -> NDArray[np.float64]:
    """Unit vector *tilt_degrees* away from +Z, tilting toward +X."""
    tilt_radians = np.deg2rad(tilt_degrees)
    return np.array(
        [np.sin(tilt_radians), 0.0, np.cos(tilt_radians)], dtype=np.float64
    )


def straight_chain_positions(*, tilt_degrees: float = 0.0) -> dict[str, NDArray[np.float64]]:
    """A collinear chain tilted *tilt_degrees* from +Z.

    ``upper`` and its twist source ``middle`` point the same way, so a
    CHAIN_RESOLVED ``upper`` has no usable roll reference and the singularity gate
    must trip. The tilt is what changes ``upper``'s own orientation — the segment
    under test runs ``upper -> middle``, so **moving only ``lower`` would leave it
    unchanged** and make any assertion about it vacuous.
    """
    direction = _segment_direction(tilt_degrees=tilt_degrees)
    upper_origin = np.zeros(3, dtype=np.float64)
    middle_origin = upper_origin + SEGMENT_LENGTH_MM * direction
    return {
        "upper": upper_origin,
        "middle": middle_origin,
        "lower": middle_origin + SEGMENT_LENGTH_MM * direction,
    }


def articulated_chain_positions(
    *, tilt_degrees: float = 0.0
) -> dict[str, NDArray[np.float64]]:
    """A chain tilted *tilt_degrees*, with ``lower`` bent perpendicular to it.

    The perpendicular bend gives ``upper`` a usable twist reference, so a
    CHAIN_RESOLVED ``upper`` resolves normally instead of degrading.
    """
    positions = straight_chain_positions(tilt_degrees=tilt_degrees)
    perpendicular = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    positions["lower"] = positions["middle"] + SEGMENT_LENGTH_MM * perpendicular
    return positions


def angle_between(first: RotationQuaternion, second: RotationQuaternion) -> float:
    _, angle = (first * second.conjugate()).to_axis_angle()
    return float(angle)


def quaternion_from_result(
    result: FrameOrientationResult, segment_name: str
) -> RotationQuaternion:
    components = result.world_quaternions[segment_name]
    return RotationQuaternion(
        w=float(components[0]),
        x=float(components[1]),
        y=float(components[2]),
        z=float(components[3]),
    )


# ── The fallback paths must produce damping state ─────────────────────


def test_singularity_gate_fallback_produces_damping_state() -> None:
    """A straightened chain trips the gate, and the fallback must still damp.

    Before the fix this path passed no previous state, so no damping state existed
    and the output tracked the noisy raw solve frame to frame.
    """
    model = build_chain_model(tier=TwistTier.CHAIN_RESOLVED, twist_source="middle")

    result = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=straight_chain_positions(tilt_degrees=20.0),
        timestamp_seconds=0.0,
    )

    assert "upper" in result.damping_states, (
        "the singularity-gate fallback must register damping state — it is the "
        "case damping exists for"
    )


def test_declared_damped_minimal_tier_produces_damping_state() -> None:
    model = build_chain_model(tier=TwistTier.DAMPED_MINIMAL)

    result = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=straight_chain_positions(tilt_degrees=30.0),
        timestamp_seconds=0.0,
    )

    assert "upper" in result.damping_states


def test_resolved_chain_does_not_damp() -> None:
    """A segment with a usable twist source is tracked directly, not smoothed."""
    model = build_chain_model(tier=TwistTier.CHAIN_RESOLVED, twist_source="middle")

    result = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=articulated_chain_positions(tilt_degrees=45.0),
        timestamp_seconds=0.0,
    )

    assert "upper" not in result.damping_states


# ── Damping actually smooths ──────────────────────────────────────────


def test_damped_segment_lags_a_step_change() -> None:
    """The whole point: a sudden jump must be approached, not tracked instantly."""
    model = build_chain_model(tier=TwistTier.DAMPED_MINIMAL)

    settled = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=straight_chain_positions(tilt_degrees=0.0),
        timestamp_seconds=0.0,
    )
    jumped_positions = straight_chain_positions(tilt_degrees=0.0)
    jumped_positions["middle"] = np.array(
        [SEGMENT_LENGTH_MM, 0.0, 0.0], dtype=np.float64
    )

    stepped = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=jumped_positions,
        timestamp_seconds=FRAME_INTERVAL_SECONDS,
        previous_result=settled,
    )

    undamped = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=jumped_positions,
        timestamp_seconds=0.0,
    )

    lag = angle_between(
        quaternion_from_result(stepped, "upper"),
        quaternion_from_result(undamped, "upper"),
    )
    assert lag > np.deg2rad(1.0), (
        "one frame after a step change the damped output should still be well "
        "short of the raw target"
    )


def test_repeated_frames_converge_along_the_analytic_envelope() -> None:
    """Held steady, the residual must follow ``(1 + t/tau) * exp(-t/tau)``.

    Asserted against the closed-form envelope rather than an arbitrary "close
    enough" threshold — a hand-picked tolerance would pass for a filter with
    roughly the right shape and the wrong time constant, which is the failure this
    whole change exists to prevent.
    """
    model = build_chain_model(tier=TwistTier.DAMPED_MINIMAL)
    initial_tilt_degrees = 0.0
    target_tilt_degrees = 40.0
    positions = straight_chain_positions(tilt_degrees=target_tilt_degrees)

    raw = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=positions,
        timestamp_seconds=0.0,
    )
    target = quaternion_from_result(raw, "upper")

    result = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=straight_chain_positions(tilt_degrees=initial_tilt_degrees),
        timestamp_seconds=0.0,
    )
    initial_error_radians = angle_between(
        quaternion_from_result(result, "upper"), target
    )

    frame_count = 30
    for frame_index in range(1, frame_count + 1):
        result = solve_frame_orientations(
            standard_human=model,
            live_joint_positions=positions,
            timestamp_seconds=frame_index * FRAME_INTERVAL_SECONDS,
            previous_result=result,
        )

    elapsed_seconds = frame_count * FRAME_INTERVAL_SECONDS
    normalized_time = elapsed_seconds / TIME_CONSTANT_SECONDS
    expected_residual_radians = (
        initial_error_radians
        * (1.0 + normalized_time)
        * float(np.exp(-normalized_time))
    )
    actual_residual_radians = angle_between(
        quaternion_from_result(result, "upper"), target
    )

    assert actual_residual_radians == pytest.approx(
        expected_residual_radians, rel=1e-6
    ), (
        f"after {elapsed_seconds:.3f}s ({normalized_time:.1f} time constants) the "
        f"residual was {np.rad2deg(actual_residual_radians):.4f} deg, expected "
        f"{np.rad2deg(expected_residual_radians):.4f} deg"
    )


# ── State ownership and clock handling ────────────────────────────────


def test_damping_state_lives_on_the_result_not_module_scope() -> None:
    """Two independent solves must not contaminate each other.

    Module-level state would let concurrent pipelines share smoothing history and
    would carry stale damping across recordings.
    """
    model = build_chain_model(tier=TwistTier.DAMPED_MINIMAL)
    positions = straight_chain_positions(tilt_degrees=25.0)

    first = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=positions,
        timestamp_seconds=0.0,
    )
    second = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=positions,
        timestamp_seconds=0.0,
    )

    assert angle_between(
        quaternion_from_result(first, "upper"),
        quaternion_from_result(second, "upper"),
    ) < 1e-12


def test_non_advancing_clock_reseeds_rather_than_fabricating_a_timestep() -> None:
    """Equal timestamps mean no elapsed time; the filter re-seeds at the raw value."""
    model = build_chain_model(tier=TwistTier.DAMPED_MINIMAL)

    first = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=straight_chain_positions(tilt_degrees=0.0),
        timestamp_seconds=1.0,
    )
    repeated = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=straight_chain_positions(tilt_degrees=40.0),
        timestamp_seconds=1.0,
        previous_result=first,
    )
    fresh = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=straight_chain_positions(tilt_degrees=40.0),
        timestamp_seconds=0.0,
    )

    assert angle_between(
        quaternion_from_result(repeated, "upper"),
        quaternion_from_result(fresh, "upper"),
    ) < 1e-12
    assert np.allclose(
        repeated.damping_states["upper"].angular_velocity_radians_per_second,
        np.zeros(3),
        atol=1e-12,
    )


def test_result_carries_its_timestamp() -> None:
    model = build_chain_model(tier=TwistTier.DAMPED_MINIMAL)
    result = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=straight_chain_positions(tilt_degrees=10.0),
        timestamp_seconds=12.5,
    )
    assert result.timestamp_seconds == pytest.approx(12.5)


# ── Fail loudly ───────────────────────────────────────────────────────


def test_coincident_joints_raise_rather_than_returning_identity() -> None:
    """A zero-length segment has no direction; silently returning identity hides it."""
    model = build_chain_model(tier=TwistTier.DAMPED_MINIMAL)
    degenerate = straight_chain_positions(tilt_degrees=0.0)
    degenerate["middle"] = degenerate["upper"].copy()

    with pytest.raises(ValueError, match="coincident"):
        solve_frame_orientations(
            standard_human=model,
            live_joint_positions=degenerate,
            timestamp_seconds=0.0,
        )
