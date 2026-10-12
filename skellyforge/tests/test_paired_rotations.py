"""Paired rotations preserve the original scalar math without N-by-N work."""

import numpy as np
import pytest

from skellyforge.core.math.geometry.rotation_quaternion import (
    RotationQuaternion,
    rotate_vector_pairs,
)


def test_pairs_and_scalar_match_original_cross_products():
    rng = np.random.default_rng(824)
    quaternions = rng.normal(size=(1000, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    vectors = rng.normal(size=(1000, 3)) * 2000
    expected = []
    actual = []
    for q, v in zip(quaternions, vectors, strict=True):
        cross = np.cross(q[1:], v)
        expected.append(v + 2.0 * (q[0] * cross + np.cross(q[1:], cross)))
        actual.append(RotationQuaternion(*map(float, q)).rotate_vector(vector=v))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        rotate_vector_pairs(quaternions=quaternions, vectors=vectors), expected
    )
    np.testing.assert_array_equal(
        rotate_vector_pairs(quaternions=-quaternions, vectors=vectors), expected
    )


def test_empty_pairs():
    assert rotate_vector_pairs(
        quaternions=np.empty((0, 4)), vectors=np.empty((0, 3))
    ).shape == (0, 3)


@pytest.mark.parametrize(
    "q,v",
    [
        (np.zeros((2, 4)), np.zeros((1, 3))),
        (np.zeros(4), np.zeros(3)),
        (np.zeros((2, 3)), np.zeros((2, 3))),
    ],
)
def test_reject_mismatched_pairs(q, v):
    with pytest.raises(ValueError):
        rotate_vector_pairs(quaternions=q, vectors=v)
