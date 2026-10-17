"""Small frame-construction helpers: the shortest-arc rotation between two directions.

A two-landmark segment (an arm, a leg) yields only a direction, not a full orientation: two
points fix the way the segment points but leave roll about that axis free. This module turns
that direction into a rotation - the shortest arc from the segment's authored primary axis to
the observed direction. Resolving the roll it leaves free is `roll_resolution.py`'s job.
"""

from __future__ import annotations

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_VECTOR_NORM
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import UnitVector
from skellyforge.type_overloads import FloatArray


def rotation_between_vectors(
    *, from_direction: UnitVector, to_direction: UnitVector
) -> RotationQuaternion:
    """The shortest-arc rotation mapping `from_direction` onto `to_direction`.

    The angle comes from `atan2(|cross|, dot)` rather than `arccos(dot)`. Both are exact in
    infinite precision; only `atan2` stays accurate near 0 and near pi, where `arccos`
    loses digits to its own vertical tangent - and near-parallel is the common case here,
    since a segment usually points close to where it pointed last frame.

    Args:
        from_direction: the unit direction to rotate from.
        to_direction: the unit direction to rotate onto.

    Returns:
        The rotation whose application to `from_direction` yields `to_direction`.
    """
    from_array = from_direction.array
    to_array = to_direction.array
    cross = np.cross(from_array, to_array)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.dot(from_array, to_array))

    if cross_norm < MINIMUM_VECTOR_NORM:
        if dot < 0.0:
            # Anti-parallel: any half turn about a perpendicular axis works, so pick a
            # deterministic one rather than leaving it to rounding noise in `cross`.
            return RotationQuaternion.from_rotation_vector(
                rotation_vector=_default_perpendicular(direction=from_array) * np.pi
            )
        return RotationQuaternion.identity()

    axis = cross / cross_norm
    angle = float(np.arctan2(cross_norm, dot))
    return RotationQuaternion.from_rotation_vector(rotation_vector=axis * angle)


def direction_pair_quaternions(*, from_directions: UnitVector, to_directions: UnitVector) -> FloatArray:
    """Shortest arcs as an array, including deterministic parallel/antiparallel cases."""
    from skellyforge.core.math.geometry.rotation_quaternion import rotation_vectors_to_quaternions
    source, target = from_directions.array, to_directions.array
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError('Direction arrays must have matching (N, 3) shapes')
    crosses = np.cross(source, target)
    norms = np.linalg.norm(crosses, axis=1)
    dots = np.sum(source * target, axis=1)
    ordinary = norms >= MINIMUM_VECTOR_NORM
    vectors = np.zeros_like(crosses)
    vectors[ordinary] = (crosses[ordinary] / norms[ordinary, None]
        * np.arctan2(norms[ordinary], dots[ordinary])[:, None])
    opposite = ~ordinary & (dots < 0)
    axes = source[opposite]
    references = np.zeros_like(axes)
    references[:, 0] = 1.
    use_y = np.abs(axes[:, 0]) > .9
    references[use_y] = [0., 1., 0.]
    perpendicular = references - np.sum(references * axes, axis=1)[:, None] * axes
    vectors[opposite] = perpendicular / np.linalg.norm(perpendicular, axis=1)[:, None] * np.pi
    return rotation_vectors_to_quaternions(vectors=vectors)


def rotations_between_vector_pairs(*, from_directions: UnitVector, to_directions: UnitVector) -> list[RotationQuaternion]:
    """Shortest-arc rotations for matching rows; retain scalar singular branches."""
    source, target = from_directions.array, to_directions.array
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError('Direction arrays must have matching (N, 3) shapes')
    crosses = np.cross(source, target)
    norms = np.linalg.norm(crosses, axis=1)
    dots = np.sum(source * target, axis=1)
    ordinary = norms >= MINIMUM_VECTOR_NORM
    vectors = np.zeros_like(crosses)
    vectors[ordinary] = (crosses[ordinary] / norms[ordinary, None]
                         * np.arctan2(norms[ordinary], dots[ordinary])[:, None])
    # The exponential map still owns quaternion construction and normalization.
    # Batch the geometric cross/dot/norm work without changing that boundary.
    return [RotationQuaternion.from_rotation_vector(rotation_vector=vector) if regular
            else rotation_between_vectors(
                from_direction=UnitVector.from_prevalidated_array(array=left),
                to_direction=UnitVector.from_prevalidated_array(array=right))
            for vector, regular, left, right in zip(vectors, ordinary, source, target, strict=True)]


def primary_axis_unit(*, axis: SpatialAxis) -> FloatArray:
    """The signed axis as a unit vector in the segment's local frame."""
    unit = np.zeros(3, dtype=np.float64)
    unit[axis.index] = float(axis.sign)
    return unit


def default_perpendicular(*, direction: FloatArray) -> FloatArray:
    """A deterministic unit vector perpendicular to `direction`.

    Deterministic matters: this is what a caller falls back to when the direction it wanted
    to build a frame around carries no usable roll reference, and a fallback that varied
    between calls would make the resulting frame jitter.
    """
    return _default_perpendicular(direction=direction)


def _default_perpendicular(*, direction: FloatArray) -> FloatArray:
    """Gram-Schmidt one of two fixed references against `direction`, whichever is safer."""
    reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(direction[0])) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    perpendicular = reference - float(np.dot(reference, direction)) * direction
    return perpendicular / np.linalg.norm(perpendicular)
