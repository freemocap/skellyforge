"""Batched direction math keeps deterministic parallel/antiparallel choices."""

import numpy as np
from skellyforge.core.math.geometry.spatial_vectors import UnitVector
from skellyforge.core.math.kinematics.coordinate_frame_ops import (
    rotation_between_vectors,
    rotations_between_vector_pairs,
)


def test_direction_pairs_match_scalar():
    rng = np.random.default_rng(923)
    source = rng.normal(size=(100, 3))
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    target = rng.normal(size=(100, 3))
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    source = np.concatenate([source, np.eye(3), np.eye(3)])
    target = np.concatenate([target, np.eye(3), -np.eye(3)])
    actual = rotations_between_vector_pairs(
        from_directions=UnitVector(source), to_directions=UnitVector(target)
    )
    for left, right, rotation in zip(source, target, actual, strict=True):
        expected = rotation_between_vectors(
            from_direction=UnitVector(left), to_direction=UnitVector(right)
        )
        np.testing.assert_allclose(
            rotation.as_array(), expected.as_array(), rtol=0, atol=1e-14
        )


def test_empty_direction_pairs():
    # UnitVector's public constructor requires a nonempty norm reduction.
    # Derived empty selections can use its existing prevalidated boundary.
    empty = UnitVector.from_prevalidated_array(array=np.empty((0, 3)))
    assert (
        rotations_between_vector_pairs(from_directions=empty, to_directions=empty) == []
    )
