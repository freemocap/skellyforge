"""Euler sequence math: decompose/compose is an exact inverse everywhere.

The round trip is the whole contract: for every one of the twelve valid
sequences, random rotations, and forced gimbal-lock rotations, composing the
decomposed angles must reproduce the original rotation to float tolerance.
The tests are also the executable definition of the intrinsic-order convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.kinematics.euler_sequence import (
    angle_axis_quaternion,
    compose_euler_angles,
    decompose_euler_angles,
    is_symmetric_sequence,
    raise_unless_valid_sequence,
)

_TAIT_BRYAN_SEQUENCES = ["".join(p) for p in
                         (("x", "y", "z"), ("x", "z", "y"), ("y", "x", "z"),
                          ("y", "z", "x"), ("z", "x", "y"), ("z", "y", "x"))]
_SYMMETRIC_SEQUENCES = ["xzx", "yxy", "yzy", "zxz", "zyz", "xyx"]
ALL_SEQUENCES = _TAIT_BRYAN_SEQUENCES + _SYMMETRIC_SEQUENCES


def _random_quaternions(count: int, seed: int) -> list[RotationQuaternion]:
    rng = np.random.default_rng(seed)
    return [
        RotationQuaternion.from_components(
            w=float(rng.standard_normal()),
            x=float(rng.standard_normal()),
            y=float(rng.standard_normal()),
            z=float(rng.standard_normal()),
        )
        for _ in range(count)
    ]


def _assert_round_trip(sequence: str, quaternion: RotationQuaternion) -> None:
    angles = decompose_euler_angles(quaternion=quaternion, sequence=sequence)
    recomposed = compose_euler_angles(sequence=sequence, angles=np.asarray(angles))
    rotation_error = quaternion.angle_to(other=recomposed)
    assert rotation_error < 1e-9, (
        f"sequence {sequence!r}: decompose/compose lost {np.degrees(rotation_error):.3e} deg"
    )


def test_the_twelve_valid_sequences_sort_into_two_families() -> None:
    assert len(ALL_SEQUENCES) == 12
    assert sorted(s for s in ALL_SEQUENCES if is_symmetric_sequence(sequence=s)) == sorted(
        _SYMMETRIC_SEQUENCES
    )
    assert sorted(s for s in ALL_SEQUENCES if not is_symmetric_sequence(sequence=s)) == sorted(
        _TAIT_BRYAN_SEQUENCES
    )


def test_invalid_sequences_are_rejected() -> None:
    for bad in ("", "xy", "xyzw", "xxz!", "aax", "xxy", "xyy", "zzx"):
        with pytest.raises(ValueError):
            raise_unless_valid_sequence(sequence=bad)


def test_non_string_sequences_are_rejected_by_the_type_gate() -> None:
    from beartype.roar import BeartypeCallHintParamViolation

    with pytest.raises(BeartypeCallHintParamViolation):
        raise_unless_valid_sequence(sequence=123)


@pytest.mark.parametrize("sequence", ALL_SEQUENCES)
def test_decompose_compose_round_trip_on_random_rotations(sequence: str) -> None:
    for quaternion in _random_quaternions(count=40, seed=abs(hash(sequence)) % (2**32)):
        _assert_round_trip(sequence, quaternion)


@pytest.mark.parametrize("sequence", ALL_SEQUENCES)
@pytest.mark.parametrize("first", [-2.5, -0.7, 0.0, 0.4, 1.9])
def test_gimbal_lock_round_trips(sequence: str, first: float) -> None:
    """Force every family's lock exactly: ±90° middle for Tait-Bryan, 0/π for symmetric."""
    if is_symmetric_sequence(sequence=sequence):
        locked_middles = [0.0, np.pi]
    else:
        locked_middles = [-np.pi / 2, np.pi / 2]
    for middle in locked_middles:
        locked = compose_euler_angles(
            sequence=sequence,
            angles=np.asarray([first, middle, 0.6]),
        )
        _assert_round_trip(sequence, locked)


@pytest.mark.parametrize("sequence", ALL_SEQUENCES)
@pytest.mark.parametrize("epsilon", [1e-9, -1e-9])
def test_near_lock_round_trips(sequence: str, epsilon: float) -> None:
    """One radian of distance from lock makes the angles ill-conditioned (the
    observable combination is stable but its split is not), so the recomposition
    tolerance here is loosened accordingly - conditioning, not correctness."""
    middle = np.pi / 2 + epsilon if not is_symmetric_sequence(sequence=sequence) else epsilon
    near_locked = compose_euler_angles(
        sequence=sequence, angles=np.asarray([1.1, middle, 0.4])
    )
    angles = decompose_euler_angles(quaternion=near_locked, sequence=sequence)
    recomposed = compose_euler_angles(sequence=sequence, angles=np.asarray(angles))
    rotation_error = near_locked.angle_to(other=recomposed)
    assert rotation_error < 1e-6, (
        f"sequence {sequence!r}: near-lock round trip lost "
        f"{np.degrees(rotation_error):.3e} deg"
    )


def test_known_angle_composes_and_decomposes_exactly() -> None:
    quarter_turn = angle_axis_quaternion(axis="z", angle=np.pi / 2)
    composed = compose_euler_angles(sequence="zyx", angles=np.array([np.pi / 2, 0.0, 0.0]))
    assert quarter_turn.is_same_rotation(other=composed)

    decomposed = decompose_euler_angles(quaternion=quarter_turn, sequence="zyx")
    np.testing.assert_allclose(decomposed, [np.pi / 2, 0.0, 0.0], atol=1e-12)
