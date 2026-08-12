"""The orientation solver's world/local outputs must satisfy the declared convention.

Spec: ``freemocap/docs/streaming-compatibility/14-engine-testing-strategy.md`` section 2.

``test_quaternion_composition.py`` pins the *convention* in isolation. This module
asserts that :func:`solve_frame_orientations` actually **produces** output obeying
it, by running a real skeleton through a **differential bend** — every segment
pointing along a different axis — and checking the round-trip::

    recompose(q_parent_world, q_child_local) == q_child_world

A uniform bend cannot detect an operand-order error here, so the fixture is built
so that no two connected segments share a direction.

All quaternion literals are ``wxyz``; identity is ``(1, 0, 0, 0)``.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from skellyforge.kinematics.orientation_solver import solve_frame_orientations
from skellyforge.kinematics.quaternion_math import RotationQuaternion
from skellyforge.skellymodels.standard_human.human_bones import (
    BoneReferenceGeometry,
    CoordinateFrameDefinition,
    HumanBone,
    TwistPolicy,
    TwistTier,
)
from skellyforge.skellymodels.standard_human.standard_human_model import StandardHuman

ROUND_TRIP_TOLERANCE = 1e-6

# Rest skeleton: a straight vertical chain, each segment 100 mm along +Z.
SEGMENT_LENGTH_MM = 100.0
CHAIN_SEGMENT_NAMES = ("hips", "spine", "chest", "head")


def build_vertical_chain_model() -> StandardHuman:
    """A 4-segment vertical chain at rest: hips -> spine -> chest -> head.

    Every segment's reference geometry points along +Z with +Y as its twist
    reference, so identity rotation means "vertical", and any live bend produces a
    non-identity world quaternion.
    """
    bones: list[HumanBone] = []
    for index, segment_name in enumerate(CHAIN_SEGMENT_NAMES):
        parent_name = CHAIN_SEGMENT_NAMES[index - 1] if index > 0 else None
        proximal = np.array([0.0, 0.0, index * SEGMENT_LENGTH_MM], dtype=np.float64)
        distal = np.array([0.0, 0.0, (index + 1) * SEGMENT_LENGTH_MM], dtype=np.float64)
        bones.append(
            HumanBone(
                name=segment_name,
                parent=parent_name,
                required=True,
                reference_geometry=BoneReferenceGeometry(
                    proximal_joint_center=proximal,
                    distal_joint_center=distal,
                    coordinate_frame=CoordinateFrameDefinition(
                        exact_axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                        approximate_axis=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                    ),
                ),
                # DAMPED_MINIMAL with no previous frame returns the un-damped
                # current value, keeping this fixture deterministic.
                twist_policy=TwistPolicy(tier=TwistTier.DAMPED_MINIMAL),
            )
        )
    return StandardHuman(name="vertical_chain", bones=bones, blendshape_channels=[])


def build_differential_bend_positions() -> dict[str, NDArray[np.float64]]:
    """Live proximal positions putting each segment along a **different** axis.

    ``hips`` runs +Y, ``spine`` runs +X, ``chest`` runs along the (1,1,1) diagonal.
    A segment's distal joint is its first child's proximal joint, so these four
    points define the three segment directions.

    Two independent ways a pair can go order-blind, both of which this fixture must
    avoid:

    1. **Either segment at its rest orientation.** Its world quaternion is identity,
       and ``conj(I) * q == q * conj(I)`` (likewise with identity on the right), so
       the pair cannot detect an operand-order error however the other side is
       rotated. Hence no segment lies along the rest axis (+Z).
    2. **Both segments rotated about the same axis.** Coaxial quaternions
       **commute**, so the two operand orders give identical results. Distinct
       segment *directions* are not sufficient — what must differ is each segment's
       *rotation axis* from rest to live. ``chest`` therefore leaves the XZ plane,
       so its rotation axis is not parallel to ``spine``'s.

    Guards: :func:`test_tested_segments_are_not_at_rest_orientation` and
    :func:`test_tested_segment_rotation_axes_are_not_parallel`.
    """
    diagonal_offset = SEGMENT_LENGTH_MM / np.sqrt(3.0)
    return {
        "hips": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "spine": np.array([0.0, SEGMENT_LENGTH_MM, 0.0], dtype=np.float64),
        "chest": np.array([SEGMENT_LENGTH_MM, SEGMENT_LENGTH_MM, 0.0], dtype=np.float64),
        "head": np.array(
            [
                SEGMENT_LENGTH_MM + diagonal_offset,
                SEGMENT_LENGTH_MM + diagonal_offset,
                diagonal_offset,
            ],
            dtype=np.float64,
        ),
    }


def rotation_axis_of(quaternion: RotationQuaternion) -> NDArray[np.float64]:
    """Unit rotation axis of a quaternion; zero vector for (near-)identity."""
    vector_part = np.array(
        [quaternion.x, quaternion.y, quaternion.z], dtype=np.float64
    )
    magnitude = float(np.linalg.norm(vector_part))
    if magnitude < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return vector_part / magnitude


def quaternion_from_wxyz(components: NDArray[np.float64]) -> RotationQuaternion:
    return RotationQuaternion(
        w=float(components[0]),
        x=float(components[1]),
        y=float(components[2]),
        z=float(components[3]),
    )


def assert_same_rotation(
    actual: RotationQuaternion,
    expected: RotationQuaternion,
    *,
    message: str,
) -> None:
    """Compare under the double cover — ``q`` and ``-q`` are the same rotation."""
    actual_wxyz = np.array([actual.w, actual.x, actual.y, actual.z], dtype=np.float64)
    expected_wxyz = np.array(
        [expected.w, expected.x, expected.y, expected.z], dtype=np.float64
    )
    matches = np.allclose(
        actual_wxyz, expected_wxyz, atol=ROUND_TRIP_TOLERANCE
    ) or np.allclose(actual_wxyz, -expected_wxyz, atol=ROUND_TRIP_TOLERANCE)
    assert matches, (
        f"{message}\n"
        f"  expected (wxyz): {np.round(expected_wxyz, 6)}\n"
        f"  actual   (wxyz): {np.round(actual_wxyz, 6)}"
    )


# ── The fixture must actually be differential ─────────────────────────


def test_fixture_segments_point_along_different_axes() -> None:
    """Guard the guard — if the bend stops being differential, it stops testing."""
    positions = build_differential_bend_positions()
    directions: list[NDArray[np.float64]] = []
    for index in range(len(CHAIN_SEGMENT_NAMES) - 1):
        proximal = positions[CHAIN_SEGMENT_NAMES[index]]
        distal = positions[CHAIN_SEGMENT_NAMES[index + 1]]
        direction = distal - proximal
        directions.append(direction / np.linalg.norm(direction))

    for first_index in range(len(directions)):
        for second_index in range(first_index + 1, len(directions)):
            alignment = abs(float(np.dot(directions[first_index], directions[second_index])))
            assert alignment < 0.9, (
                "connected segments must not share a direction, or the fixture "
                "becomes order-blind"
            )


def test_tested_segments_are_not_at_rest_orientation() -> None:
    """An identity world quaternion on **either** side of a pair is order-blind.

    ``conj(I) * q == q * conj(I) == q`` and ``conj(p) * I == I * conj(p) == conj(p)``,
    so a pair with an identity parent *or* an identity child cannot detect an
    operand-order error however the other side is rotated.

    This guard is here because two successive fixtures had exactly that hole: the
    first ran ``hips`` along the rest axis (identity parent) and only ``spine``/
    ``chest`` failed; the second fixed ``hips`` but left ``chest`` on the rest axis
    (identity child) and only ``hips``/``spine`` failed. Both times half the suite
    passed for the wrong reason.
    """
    result = solve_frame_orientations(
        standard_human=build_vertical_chain_model(),
        live_joint_positions=build_differential_bend_positions(),
        timestamp_seconds=0.0,
    )
    for segment_name in ("hips", "spine", "chest"):
        world = quaternion_from_wxyz(result.world_quaternions[segment_name])
        assert abs(world.w) < 0.999, (
            f"{segment_name} is at (or near) its rest orientation, so any pair it "
            f"takes part in is order-blind — bend it in the fixture"
        )


def test_tested_segment_rotation_axes_are_not_parallel() -> None:
    """Coaxial rotations commute, so a coaxial pair is order-blind.

    If a parent and child are both rotated about the same axis, ``conj(p) * c`` and
    ``c * conj(p)`` are equal and the pair proves nothing — regardless of how
    different the two segment *directions* look.

    This guard exists because the fixture had exactly that hole: ``spine`` (+Z→+X)
    and ``chest`` (+Z→ a direction in the XZ plane) were both rotations about +Y,
    so that pair passed while ``hips``/``spine`` failed. Distinct directions are not
    the invariant; distinct rotation axes are.
    """
    result = solve_frame_orientations(
        standard_human=build_vertical_chain_model(),
        live_joint_positions=build_differential_bend_positions(),
        timestamp_seconds=0.0,
    )
    for parent_name, child_name in (("hips", "spine"), ("spine", "chest")):
        parent_axis = rotation_axis_of(quaternion_from_wxyz(result.world_quaternions[parent_name]))
        child_axis = rotation_axis_of(quaternion_from_wxyz(result.world_quaternions[child_name]))
        alignment = abs(float(np.dot(parent_axis, child_axis)))
        assert alignment < 0.99, (
            f"{parent_name} and {child_name} rotate about (near-)parallel axes "
            f"(|dot| = {alignment:.4f}); coaxial rotations commute, so this pair "
            f"cannot detect an operand-order error"
        )


def test_solver_reaches_every_non_leaf_segment() -> None:
    """The round-trip below is only meaningful for segments the solver resolved."""
    result = solve_frame_orientations(
        standard_human=build_vertical_chain_model(),
        live_joint_positions=build_differential_bend_positions(),
        timestamp_seconds=0.0,
    )
    for segment_name in ("hips", "spine", "chest"):
        assert segment_name in result.world_quaternions
        assert segment_name in result.local_quaternions


# ── The decisive assertion ────────────────────────────────────────────


@pytest.mark.parametrize(("parent_name", "child_name"), [("hips", "spine"), ("spine", "chest")])
def test_local_quaternion_round_trips_to_world(parent_name: str, child_name: str) -> None:
    """``q_parent_world * q_child_local`` must reproduce ``q_child_world``.

    This is the structural check on the convention rather than a sample of it. It
    fails if the solver composes the parent-relative rotation with reversed
    operands, which yields the correct angle about the wrong axis.
    """
    result = solve_frame_orientations(
        standard_human=build_vertical_chain_model(),
        live_joint_positions=build_differential_bend_positions(),
        timestamp_seconds=0.0,
    )

    parent_world = quaternion_from_wxyz(result.world_quaternions[parent_name])
    child_world = quaternion_from_wxyz(result.world_quaternions[child_name])
    child_local = quaternion_from_wxyz(result.local_quaternions[child_name])

    assert_same_rotation(
        parent_world * child_local,
        child_world,
        message=(
            f"recompose({parent_name}_world, {child_name}_local) must equal "
            f"{child_name}_world"
        ),
    )


@pytest.mark.parametrize(("parent_name", "child_name"), [("hips", "spine"), ("spine", "chest")])
def test_local_quaternion_equals_conjugate_parent_times_child(
    parent_name: str, child_name: str
) -> None:
    """State the convention directly: ``q_local = conj(q_parent) * q_child``."""
    result = solve_frame_orientations(
        standard_human=build_vertical_chain_model(),
        live_joint_positions=build_differential_bend_positions(),
        timestamp_seconds=0.0,
    )

    parent_world = quaternion_from_wxyz(result.world_quaternions[parent_name])
    child_world = quaternion_from_wxyz(result.world_quaternions[child_name])

    assert_same_rotation(
        quaternion_from_wxyz(result.local_quaternions[child_name]),
        parent_world.conjugate() * child_world,
        message=f"{child_name} local must be conj(parent_world) * child_world",
    )


def test_root_local_equals_root_world() -> None:
    """The root has no parent, so its parent-relative rotation is its world one."""
    result = solve_frame_orientations(
        standard_human=build_vertical_chain_model(),
        live_joint_positions=build_differential_bend_positions(),
        timestamp_seconds=0.0,
    )
    assert_same_rotation(
        quaternion_from_wxyz(result.local_quaternions["hips"]),
        quaternion_from_wxyz(result.world_quaternions["hips"]),
        message="root local must equal root world",
    )


def test_rest_pose_input_yields_identity_rotations() -> None:
    """Feeding the model its own rest geometry must return identity everywhere.

    This is the ``identity == T-pose`` contract that every downstream adapter
    assumes.
    """
    model = build_vertical_chain_model()
    rest_positions = {
        bone.name: bone.reference_geometry.proximal_joint_center.copy()
        for bone in model.bones
    }

    result = solve_frame_orientations(
        standard_human=model,
        live_joint_positions=rest_positions,
        timestamp_seconds=0.0,
    )

    for segment_name, world_wxyz in result.world_quaternions.items():
        assert_same_rotation(
            quaternion_from_wxyz(world_wxyz),
            RotationQuaternion.identity(),
            message=f"{segment_name} must be identity at rest pose",
        )
