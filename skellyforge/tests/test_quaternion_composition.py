"""Parent-relative quaternion composition — the decisive convention test.

Spec: ``freemocap/docs/streaming-compatibility/14-engine-testing-strategy.md`` section 2.
Convention: ``freemocap/docs/streaming-compatibility/07-coordinate-conventions.md``
section "Segment rotation conventions".

The convention under test
-------------------------
``q_world`` maps segment-frame -> world: it carries a segment from its declared
rest orientation to its current one, which is the ``identity == T-pose`` contract.
With Hamilton semantics ``R(q1 * q2) = R(q1) . R(q2)`` (q2 applied first)::

    q_child_world = q_parent_world * q_child_local
    q_child_local = conj(q_parent_world) * q_child_world

Why these fixtures are asymmetric
---------------------------------
A *uniform* bend — every segment rotated identically — cannot distinguish
``conj(q_p) * q_c`` from ``q_c * conj(q_p)``, because ``q_child == q_parent`` makes
both reduce to identity. Every case below rotates parent and child about
**different axes** by **different magnitudes** so the operand order is observable.

All quaternion literals are ``wxyz``. Identity is ``(1, 0, 0, 0)``.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from skellyforge.kinematics.quaternion_math import RotationQuaternion

QUATERNION_TOLERANCE = 1e-9


def rotation_about_axis(
    *,
    axis: tuple[float, float, float],
    degrees: float,
) -> RotationQuaternion:
    """Build a rotation quaternion from an axis and an angle in degrees."""
    axis_array: NDArray[np.float64] = np.asarray(axis, dtype=np.float64)
    axis_array = axis_array / np.linalg.norm(axis_array)
    half_angle_radians = np.deg2rad(degrees) / 2.0
    sine = float(np.sin(half_angle_radians))
    return RotationQuaternion(
        w=float(np.cos(half_angle_radians)),
        x=float(axis_array[0] * sine),
        y=float(axis_array[1] * sine),
        z=float(axis_array[2] * sine),
    )


def to_wxyz(quaternion: RotationQuaternion) -> NDArray[np.float64]:
    """Quaternion components as a ``wxyz`` array — the canonical order."""
    return np.array(
        [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
        dtype=np.float64,
    )


def assert_same_rotation(
    actual: RotationQuaternion,
    expected: RotationQuaternion,
    *,
    message: str,
) -> None:
    """Assert two quaternions represent the same rotation.

    Compares under the double cover: ``q`` and ``-q`` are the same rotation, so a
    sign flip is not a failure. An axis difference is.
    """
    actual_wxyz = to_wxyz(actual)
    expected_wxyz = to_wxyz(expected)
    matches_directly = np.allclose(actual_wxyz, expected_wxyz, atol=QUATERNION_TOLERANCE)
    matches_negated = np.allclose(actual_wxyz, -expected_wxyz, atol=QUATERNION_TOLERANCE)
    assert matches_directly or matches_negated, (
        f"{message}\n"
        f"  expected (wxyz): {np.round(expected_wxyz, 6)}\n"
        f"  actual   (wxyz): {np.round(actual_wxyz, 6)}"
    )


def compose_local_from_world(
    *,
    parent_world: RotationQuaternion,
    child_world: RotationQuaternion,
) -> RotationQuaternion:
    """The parent-relative rotation, per the declared convention.

    ``q_child_local = conj(q_parent_world) * q_child_world``
    """
    return parent_world.conjugate() * child_world


def recompose_world_from_local(
    *,
    parent_world: RotationQuaternion,
    child_local: RotationQuaternion,
) -> RotationQuaternion:
    """Invert :func:`compose_local_from_world`.

    ``q_child_world = q_parent_world * q_child_local``
    """
    return parent_world * child_local


# ── Hamilton product semantics ────────────────────────────────────────


def test_hamilton_product_applies_right_operand_first() -> None:
    """``R(q1 * q2)`` must equal "apply q2, then q1" — the documented contract."""
    first_applied = rotation_about_axis(axis=(0.0, 0.0, 1.0), degrees=90.0)
    second_applied = rotation_about_axis(axis=(1.0, 0.0, 0.0), degrees=90.0)
    test_vector: NDArray[np.float64] = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    composed = second_applied * first_applied
    sequential = second_applied.rotate_vector(first_applied.rotate_vector(test_vector))

    assert np.allclose(composed.rotate_vector(test_vector), sequential, atol=QUATERNION_TOLERANCE), (
        "q1 * q2 must apply q2 first, then q1"
    )


def test_hamilton_product_does_not_commute() -> None:
    """Pin non-commutativity, so operand order is never assumed to be free."""
    about_z = rotation_about_axis(axis=(0.0, 0.0, 1.0), degrees=90.0)
    about_x = rotation_about_axis(axis=(1.0, 0.0, 0.0), degrees=90.0)

    forward_wxyz = to_wxyz(about_z * about_x)
    reversed_wxyz = to_wxyz(about_x * about_z)

    assert not np.allclose(forward_wxyz, reversed_wxyz, atol=QUATERNION_TOLERANCE)
    assert not np.allclose(forward_wxyz, -reversed_wxyz, atol=QUATERNION_TOLERANCE)


def test_conjugate_is_inverse_for_unit_quaternions() -> None:
    rotation = rotation_about_axis(axis=(1.0, 2.0, 3.0), degrees=57.0)
    assert_same_rotation(
        rotation * rotation.conjugate(),
        RotationQuaternion.identity(),
        message="q * conj(q) must be identity for a unit quaternion",
    )


# ── The decisive case: differential bend ──────────────────────────────


def test_differential_bend_recovers_local_rotation_including_axis() -> None:
    """Parent and child rotated about **different** axes by **different** amounts.

    This is the case a uniform bend cannot test. The reversed operand order
    (``q_child * conj(q_parent)``) produces a rotation of the correct *angle* about
    the *wrong axis*, so an angle-only assertion would pass it — the axis is
    asserted here.
    """
    parent_world = rotation_about_axis(axis=(0.0, 0.0, 1.0), degrees=90.0)
    expected_local = rotation_about_axis(axis=(1.0, 0.0, 0.0), degrees=90.0)
    child_world = recompose_world_from_local(
        parent_world=parent_world,
        child_local=expected_local,
    )

    computed_local = compose_local_from_world(
        parent_world=parent_world,
        child_world=child_world,
    )

    assert_same_rotation(
        computed_local,
        expected_local,
        message="parent-relative rotation must be conj(parent) * child",
    )


def test_reversed_operand_order_is_observably_wrong() -> None:
    """Guard the guard: prove the fixture can actually detect the reversed order.

    If this test ever fails, the differential-bend fixture above has become
    order-blind and is no longer testing what it claims.
    """
    parent_world = rotation_about_axis(axis=(0.0, 0.0, 1.0), degrees=90.0)
    expected_local = rotation_about_axis(axis=(1.0, 0.0, 0.0), degrees=90.0)
    child_world = recompose_world_from_local(
        parent_world=parent_world,
        child_local=expected_local,
    )

    reversed_order = child_world * parent_world.conjugate()

    reversed_wxyz = to_wxyz(reversed_order)
    expected_wxyz = to_wxyz(expected_local)
    assert not np.allclose(reversed_wxyz, expected_wxyz, atol=QUATERNION_TOLERANCE)
    assert not np.allclose(reversed_wxyz, -expected_wxyz, atol=QUATERNION_TOLERANCE)


def test_uniform_bend_is_order_blind() -> None:
    """Document *why* the previously-passing uniform-bend check proved nothing.

    When ``q_child == q_parent`` both operand orders collapse to identity. This
    test asserts that blindness explicitly so nobody reintroduces a uniform-only
    fixture believing it covers the convention.
    """
    uniform = rotation_about_axis(axis=(0.0, 1.0, 0.0), degrees=90.0)

    correct_order = uniform.conjugate() * uniform
    reversed_order = uniform * uniform.conjugate()

    assert_same_rotation(
        correct_order,
        RotationQuaternion.identity(),
        message="uniform bend: correct order gives identity",
    )
    assert_same_rotation(
        reversed_order,
        RotationQuaternion.identity(),
        message="uniform bend: reversed order ALSO gives identity — hence order-blind",
    )


# ── Round-trip: pins the convention structurally ──────────────────────


@pytest.mark.parametrize(
    ("parent_axis", "parent_degrees", "local_axis", "local_degrees"),
    [
        ((0.0, 0.0, 1.0), 90.0, (1.0, 0.0, 0.0), 90.0),
        ((1.0, 0.0, 0.0), 33.0, (0.0, 1.0, 0.0), 71.0),
        ((1.0, 1.0, 0.0), 120.0, (0.0, 1.0, 1.0), 45.0),
        ((0.0, 1.0, 0.0), 179.0, (1.0, 0.0, 1.0), 15.0),
    ],
)
def test_local_world_round_trip(
    parent_axis: tuple[float, float, float],
    parent_degrees: float,
    local_axis: tuple[float, float, float],
    local_degrees: float,
) -> None:
    """``recompose(parent, local) == child`` for arbitrary asymmetric rotations."""
    parent_world = rotation_about_axis(axis=parent_axis, degrees=parent_degrees)
    original_local = rotation_about_axis(axis=local_axis, degrees=local_degrees)
    child_world = recompose_world_from_local(
        parent_world=parent_world,
        child_local=original_local,
    )

    recovered_local = compose_local_from_world(
        parent_world=parent_world,
        child_world=child_world,
    )
    recomposed_world = recompose_world_from_local(
        parent_world=parent_world,
        child_local=recovered_local,
    )

    assert_same_rotation(
        recovered_local,
        original_local,
        message="local recovered from world must match the original local",
    )
    assert_same_rotation(
        recomposed_world,
        child_world,
        message="recompose(parent, local) must reproduce child world",
    )


def test_three_deep_chain_composes_without_cancellation() -> None:
    """hips -> spine -> chest, each rotated differently.

    A depth-2 chain can hide an error that cancels between two levels; three
    distinct rotations cannot.
    """
    hips_world = rotation_about_axis(axis=(0.0, 0.0, 1.0), degrees=25.0)
    spine_local = rotation_about_axis(axis=(1.0, 0.0, 0.0), degrees=40.0)
    chest_local = rotation_about_axis(axis=(0.0, 1.0, 0.0), degrees=15.0)

    spine_world = recompose_world_from_local(parent_world=hips_world, child_local=spine_local)
    chest_world = recompose_world_from_local(parent_world=spine_world, child_local=chest_local)

    assert_same_rotation(
        compose_local_from_world(parent_world=hips_world, child_world=spine_world),
        spine_local,
        message="spine local must be recoverable from hips + spine world",
    )
    assert_same_rotation(
        compose_local_from_world(parent_world=spine_world, child_world=chest_world),
        chest_local,
        message="chest local must be recoverable from spine + chest world",
    )


def test_identity_parent_leaves_local_equal_to_world() -> None:
    """A root segment has no parent, so its local rotation equals its world one."""
    child_world = rotation_about_axis(axis=(0.3, 0.5, 0.8), degrees=64.0)
    computed_local = compose_local_from_world(
        parent_world=RotationQuaternion.identity(),
        child_world=child_world,
    )
    assert_same_rotation(
        computed_local,
        child_world,
        message="with an identity parent, local must equal world",
    )
