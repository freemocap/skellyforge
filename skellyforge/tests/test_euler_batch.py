"""Batch Euler conversion must preserve all twelve conventions and lock branches."""

from itertools import permutations
import numpy as np
import pytest

from skellyforge.core.math.kinematics.euler_sequence import (
    compose_euler_angles,
    decompose_euler_angles,
    decompose_euler_batch,
)

SEQUENCES = ["".join(p) for p in permutations("xyz")] + [
    a + b + a for a in "xyz" for b in "xyz" if a != b
]


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_batch_matches_scalar_at_random_rotations_and_both_locks(sequence):
    rng = np.random.default_rng(872)
    triples = list(rng.uniform(-np.pi, np.pi, (100, 3)))
    for middle in (
        [0.0, np.pi] if sequence[0] == sequence[2] else [-np.pi / 2, np.pi / 2]
    ):
        for delta in (0.0, -1e-7, 1e-7, -1e-4, 1e-4):
            triples.append(np.array([0.37, middle + delta, -0.83]))
    quaternions = [
        compose_euler_angles(sequence=sequence, angles=row) for row in triples
    ]
    expected = np.asarray(
        [decompose_euler_angles(quaternion=q, sequence=sequence) for q in quaternions]
    )
    actual = decompose_euler_batch(
        quaternions=np.asarray([q.as_array() for q in quaternions]), sequence=sequence
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_empty_batch():
    assert decompose_euler_batch(
        quaternions=np.empty((0, 4)), sequence="xyz"
    ).shape == (0, 3)
