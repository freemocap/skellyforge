"""Tests for the quaternion algebra, scalar and vectorized.

The vectorized half of `rotation_quaternion.py` exists to do in one numpy call what the
scalar half does in a Python loop. That is the contract, so most of what is below is a
parity test: build the same rotations both ways and require the same answer. It is worth
saying why that shape was chosen over testing each batched function against a
hand-computed expectation - the scalar half is already covered by round trips and by
identities that do not depend on the implementation (`R` orthogonal, `q * q^-1 = 1`,
exp/log inverse), so anchoring the batched half to it inherits all of that, and a drift
between the two halves is the specific failure this file was written to catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.numeric_tolerances import (
    MINIMUM_SLERP_SEPARATION_COSINE,
)
from skellyforge.core.math.geometry.rotation_quaternion import (
    RotationQuaternion,
    compute_angular_velocity,
    conjugate_quaternion_array,
    hamilton_product,
    normalize_quaternion_array,
    post_multiply_by_constant,
    pre_multiply_by_constant,
    quaternions_to_axis_angles,
    quaternions_to_roll_pitch_yaw,
    quaternions_to_rotation_matrices,
    rotate_vectors_batch,
    slerp_batch,
    slerp_resample,
)

NUMBER_OF_RANDOM_ROTATIONS: int = 200


def _random_rotations(*, seed: int, count: int = NUMBER_OF_RANDOM_ROTATIONS
) -> list[RotationQuaternion]:
    """A spread of rotations, including some large enough to cross the double cover."""
    generator = np.random.default_rng(seed)
    return [
        RotationQuaternion.from_rotation_vector(
            rotation_vector=generator.normal(scale=2.0, size=3)
        )
        for _ in range(count)
    ]


def _as_array(rotations: list[RotationQuaternion]) -> np.ndarray:
    return np.stack([rotation.as_array() for rotation in rotations], axis=0)


# ── scalar identities ─────────────────────────────────────────────────


def test_rotation_matrices_are_orthogonal_with_unit_determinant() -> None:
    for rotation in _random_rotations(seed=1):
        matrix = rotation.to_rotation_matrix()
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(matrix)) == pytest.approx(1.0, abs=1e-12)


def test_matrix_round_trip_recovers_the_rotation() -> None:
    for rotation in _random_rotations(seed=2):
        recovered = RotationQuaternion.from_rotation_matrix(
            matrix=rotation.to_rotation_matrix()
        )
        assert recovered.is_same_rotation(other=rotation, tolerance_radians=1e-9)


def test_rotation_vector_round_trip_recovers_the_rotation() -> None:
    for rotation in _random_rotations(seed=3):
        recovered = RotationQuaternion.from_rotation_vector(
            rotation_vector=rotation.to_rotation_vector()
        )
        assert recovered.is_same_rotation(other=rotation, tolerance_radians=1e-9)


def test_a_rotation_times_its_inverse_is_the_identity() -> None:
    identity = RotationQuaternion.identity()
    for rotation in _random_rotations(seed=4):
        assert (rotation * rotation.inverse()).is_same_rotation(
            other=identity, tolerance_radians=1e-9
        )


def test_rotate_vector_matches_the_matrix_product() -> None:
    generator = np.random.default_rng(5)
    for rotation in _random_rotations(seed=5, count=50):
        vector = generator.normal(size=3)
        np.testing.assert_allclose(
            rotation.rotate_vector(vector=vector),
            rotation.to_rotation_matrix() @ vector,
            atol=1e-12,
        )


def test_axis_angle_stays_on_the_shorter_arc() -> None:
    for rotation in _random_rotations(seed=6):
        axis, angle = rotation.to_axis_angle()
        assert 0.0 <= angle <= np.pi + 1e-12
        assert float(np.linalg.norm(axis)) == pytest.approx(1.0, abs=1e-12)


def test_a_quaternion_and_its_negation_are_the_same_rotation() -> None:
    rotation = RotationQuaternion.from_rotation_vector(
        rotation_vector=np.array([0.3, -1.1, 0.7])
    )
    negated = RotationQuaternion(
        w=-rotation.w, x=-rotation.x, y=-rotation.y, z=-rotation.z
    )
    assert rotation.is_same_rotation(other=negated)
    assert rotation.angle_to(other=negated) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        rotation.to_rotation_matrix(), negated.to_rotation_matrix(), atol=1e-12
    )


def test_roll_pitch_yaw_round_trips_through_its_own_composition() -> None:
    for rotation in _random_rotations(seed=7, count=50):
        roll, pitch, yaw = rotation.to_roll_pitch_yaw()
        rebuilt = (
            RotationQuaternion.from_rotation_vector(
                rotation_vector=np.array([0.0, 0.0, yaw])
            )
            * RotationQuaternion.from_rotation_vector(
                rotation_vector=np.array([0.0, pitch, 0.0])
            )
            * RotationQuaternion.from_rotation_vector(
                rotation_vector=np.array([roll, 0.0, 0.0])
            )
        )
        assert rebuilt.is_same_rotation(other=rotation, tolerance_radians=1e-8)


# ── scalar / batch parity ─────────────────────────────────────────────


def test_hamilton_product_matches_the_scalar_product() -> None:
    left = _random_rotations(seed=11)
    right = _random_rotations(seed=12)
    batched = hamilton_product(left=_as_array(left), right=_as_array(right))
    for index, (first, second) in enumerate(zip(left, right)):
        expected = first * second
        recovered = RotationQuaternion.from_array(array=batched[index])
        assert recovered.is_same_rotation(other=expected, tolerance_radians=1e-9)


def test_conjugate_array_matches_the_scalar_conjugate() -> None:
    rotations = _random_rotations(seed=13)
    batched = conjugate_quaternion_array(quaternions=_as_array(rotations))
    for index, rotation in enumerate(rotations):
        np.testing.assert_allclose(
            batched[index], rotation.conjugate().as_array(), atol=1e-15
        )


def test_rotation_matrices_match_the_scalar_conversion() -> None:
    rotations = _random_rotations(seed=14)
    batched = quaternions_to_rotation_matrices(quaternions=_as_array(rotations))
    for index, rotation in enumerate(rotations):
        np.testing.assert_allclose(
            batched[index], rotation.to_rotation_matrix(), atol=1e-15
        )


def test_axis_angles_match_the_scalar_conversion() -> None:
    rotations = _random_rotations(seed=15)
    axes, angles = quaternions_to_axis_angles(quaternions=_as_array(rotations))
    for index, rotation in enumerate(rotations):
        expected_axis, expected_angle = rotation.to_axis_angle()
        assert float(angles[index]) == pytest.approx(expected_angle, abs=1e-12)
        np.testing.assert_allclose(axes[index], expected_axis, atol=1e-12)


def test_roll_pitch_yaw_matches_the_scalar_conversion() -> None:
    rotations = _random_rotations(seed=16)
    batched = quaternions_to_roll_pitch_yaw(quaternions=_as_array(rotations))
    for index, rotation in enumerate(rotations):
        np.testing.assert_allclose(
            batched[index], rotation.to_roll_pitch_yaw(), atol=1e-12
        )


def test_rotate_vectors_batch_matches_the_scalar_rotation() -> None:
    generator = np.random.default_rng(17)
    rotations = _random_rotations(seed=17, count=30)
    vectors = generator.normal(size=(4, 3))
    batched = rotate_vectors_batch(quaternions=_as_array(rotations), vectors=vectors)
    assert batched.shape == (len(rotations), len(vectors), 3)
    for rotation_index, rotation in enumerate(rotations):
        for vector_index, vector in enumerate(vectors):
            np.testing.assert_allclose(
                batched[rotation_index, vector_index],
                rotation.rotate_vector(vector=vector),
                atol=1e-12,
            )


def test_slerp_batch_matches_the_scalar_slerp() -> None:
    start = _random_rotations(seed=18, count=60)
    end = _random_rotations(seed=19, count=60)
    fractions = np.linspace(0.0, 1.0, len(start))
    batched = slerp_batch(
        start=_as_array(start), end=_as_array(end), fractions=fractions
    )
    for index, (first, second) in enumerate(zip(start, end)):
        expected = RotationQuaternion.slerp(
            start=first, end=second, fraction=float(fractions[index])
        )
        recovered = RotationQuaternion.from_array(array=batched[index])
        assert recovered.is_same_rotation(other=expected, tolerance_radians=1e-8)


def test_the_two_slerps_agree_across_the_nlerp_threshold() -> None:
    """Both halves must switch to NLERP at the same separation, not at two thresholds.

    They used to disagree - the scalar fell back at about 1.8 degrees and the batch at
    about 0.002 - so the same inputs took different code paths depending on which was
    called. This walks the separation straight through the shared threshold.
    """
    threshold_angle = 2.0 * np.arccos(MINIMUM_SLERP_SEPARATION_COSINE)
    for multiplier in (0.1, 0.5, 0.99, 1.0, 1.01, 2.0, 10.0, 1000.0):
        angle = threshold_angle * multiplier
        start = RotationQuaternion.identity()
        end = RotationQuaternion.from_rotation_vector(
            rotation_vector=np.array([0.0, 0.0, angle])
        )
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            scalar = RotationQuaternion.slerp(
                start=start, end=end, fraction=fraction
            )
            batched = RotationQuaternion.from_array(
                array=slerp_batch(
                    start=start.as_array()[np.newaxis, :],
                    end=end.as_array()[np.newaxis, :],
                    fractions=np.array([fraction]),
                )[0]
            )
            assert scalar.is_same_rotation(other=batched, tolerance_radians=1e-9), (
                f"scalar and batch SLERP disagree at angle {angle:.3e} rad, "
                f"fraction {fraction}"
            )


def test_slerp_takes_the_shorter_arc_across_the_double_cover() -> None:
    start = RotationQuaternion.identity()
    end = RotationQuaternion.from_rotation_vector(
        rotation_vector=np.array([0.0, 0.0, 3.0])
    )
    negated_end = RotationQuaternion(w=-end.w, x=-end.x, y=-end.y, z=-end.z)
    for fraction in (0.25, 0.5, 0.75):
        assert RotationQuaternion.slerp(
            start=start, end=end, fraction=fraction
        ).is_same_rotation(
            other=RotationQuaternion.slerp(
                start=start, end=negated_end, fraction=fraction
            ),
            tolerance_radians=1e-9,
        )


def test_pre_and_post_multiplication_apply_the_constant_on_the_right_side() -> None:
    rotations = _random_rotations(seed=20, count=40)
    constant = RotationQuaternion.from_rotation_vector(
        rotation_vector=np.array([0.4, -0.2, 1.3])
    )
    pre = pre_multiply_by_constant(
        quaternions=_as_array(rotations), constant=constant.as_array()
    )
    post = post_multiply_by_constant(
        quaternions=_as_array(rotations), constant=constant.as_array()
    )
    for index, rotation in enumerate(rotations):
        assert RotationQuaternion.from_array(array=pre[index]).is_same_rotation(
            other=constant * rotation, tolerance_radians=1e-9
        )
        assert RotationQuaternion.from_array(array=post[index]).is_same_rotation(
            other=rotation * constant, tolerance_radians=1e-9
        )


# ── trajectory operations ─────────────────────────────────────────────


def test_normalize_quaternion_array_returns_unit_rows_without_mutating_its_input() -> None:
    raw = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0]])
    original = raw.copy()
    normalized = normalize_quaternion_array(quaternions=raw)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(raw, original)


def test_slerp_resample_reproduces_the_source_at_its_own_timestamps() -> None:
    rotations = _random_rotations(seed=21, count=10)
    timestamps = np.arange(len(rotations), dtype=np.float64) / 30.0
    resampled = slerp_resample(
        quaternions=_as_array(rotations),
        original_timestamps=timestamps,
        target_timestamps=timestamps,
    )
    for index, rotation in enumerate(rotations):
        assert RotationQuaternion.from_array(array=resampled[index]).is_same_rotation(
            other=rotation, tolerance_radians=1e-8
        )


def test_slerp_resample_clamps_outside_the_source_range() -> None:
    rotations = _random_rotations(seed=22, count=4)
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    resampled = slerp_resample(
        quaternions=_as_array(rotations),
        original_timestamps=timestamps,
        target_timestamps=np.array([-10.0, 100.0]),
    )
    assert RotationQuaternion.from_array(array=resampled[0]).is_same_rotation(
        other=rotations[0], tolerance_radians=1e-8
    )
    assert RotationQuaternion.from_array(array=resampled[1]).is_same_rotation(
        other=rotations[-1], tolerance_radians=1e-8
    )


def test_slerp_resample_halfway_matches_a_direct_slerp() -> None:
    rotations = _random_rotations(seed=23, count=2)
    resampled = slerp_resample(
        quaternions=_as_array(rotations),
        original_timestamps=np.array([0.0, 1.0]),
        target_timestamps=np.array([0.5]),
    )
    expected = RotationQuaternion.slerp(
        start=rotations[0], end=rotations[1], fraction=0.5
    )
    assert RotationQuaternion.from_array(array=resampled[0]).is_same_rotation(
        other=expected, tolerance_radians=1e-8
    )


def test_angular_velocity_of_a_constant_spin_is_that_spin() -> None:
    """Rotating at a fixed rate about a fixed axis must report exactly that rate."""
    axis = np.array([0.0, 0.0, 1.0])
    rate_radians_per_second = 1.5
    timestamps = np.linspace(0.0, 1.0, 41)
    trajectory = np.stack(
        [
            RotationQuaternion.from_rotation_vector(
                rotation_vector=axis * rate_radians_per_second * time
            ).as_array()
            for time in timestamps
        ],
        axis=0,
    )
    omega_world, omega_body = compute_angular_velocity(
        quaternions=trajectory, timestamps=timestamps
    )
    expected = axis * rate_radians_per_second
    np.testing.assert_allclose(omega_world, np.broadcast_to(expected, omega_world.shape), atol=1e-9)
    # Spinning about its own z, the body frame sees the same vector.
    np.testing.assert_allclose(omega_body, np.broadcast_to(expected, omega_body.shape), atol=1e-9)


def test_angular_velocity_of_a_still_trajectory_is_zero() -> None:
    still = np.broadcast_to(RotationQuaternion.identity().as_array(), (5, 4)).copy()
    omega_world, omega_body = compute_angular_velocity(
        quaternions=still, timestamps=np.arange(5, dtype=np.float64)
    )
    np.testing.assert_allclose(omega_world, 0.0, atol=1e-15)
    np.testing.assert_allclose(omega_body, 0.0, atol=1e-15)


# ── what the module refuses ───────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_array",
    [
        np.zeros((3,)),
        np.zeros((3, 3)),
        np.zeros((2, 4, 1)),
    ],
)
def test_batched_functions_reject_arrays_that_are_not_quaternions(bad_array) -> None:
    with pytest.raises(ValueError, match="quaternion array"):
        conjugate_quaternion_array(quaternions=bad_array)


def test_normalize_rejects_a_near_zero_row() -> None:
    with pytest.raises(ValueError, match="Near-zero quaternion"):
        normalize_quaternion_array(
            quaternions=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        )


def test_slerp_rejects_a_fraction_outside_the_unit_interval() -> None:
    identity = RotationQuaternion.identity()
    with pytest.raises(ValueError, match=r"fraction must be in \[0, 1\]"):
        RotationQuaternion.slerp(start=identity, end=identity, fraction=1.5)
    with pytest.raises(ValueError, match=r"fractions must all be in \[0, 1\]"):
        slerp_batch(
            start=identity.as_array()[np.newaxis, :],
            end=identity.as_array()[np.newaxis, :],
            fractions=np.array([-0.1]),
        )


def test_trajectory_functions_reject_non_increasing_timestamps() -> None:
    trajectory = np.broadcast_to(
        RotationQuaternion.identity().as_array(), (3, 4)
    ).copy()
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_angular_velocity(
            quaternions=trajectory, timestamps=np.array([0.0, 1.0, 1.0])
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        slerp_resample(
            quaternions=trajectory,
            original_timestamps=np.array([0.0, 2.0, 1.0]),
            target_timestamps=np.array([0.5]),
        )


def test_the_constructor_still_refuses_a_non_unit_quaternion() -> None:
    with pytest.raises(ValueError, match="must be unit length"):
        RotationQuaternion(w=1.0, x=1.0, y=0.0, z=0.0)
