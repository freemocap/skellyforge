"""Small frame-construction helpers: the shortest-arc rotation between two directions.

A two-landmark segment (an arm, a leg) yields only a direction, not a full orientation: two
points fix the way the segment points but leave roll about that axis free. This module turns
that direction into a rotation - the shortest arc from the segment's authored primary axis to
the observed direction - whose roll is then resolved downstream by a continuous parallel-transport convention.
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
    """The shortest-arc rotation mapping from_direction onto to_direction.

    Args:
        from_direction: the unit direction to rotate from.
        to_direction: the unit direction to rotate onto.

    Returns:
        The rotation whose application to from_direction yields to_direction.
    """
    from_array = from_direction.array
    to_array = to_direction.array
    dot = float(np.clip(np.dot(from_array, to_array), -1.0, 1.0))
    cross = np.cross(from_array, to_array)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < MINIMUM_VECTOR_NORM:
        if dot < 0.0:
            # Anti-parallel: any 180-degree rotation about a perpendicular axis works.
            return RotationQuaternion.from_rotation_vector(
                _default_perpendicular(from_array) * np.pi
            )
        return RotationQuaternion.identity()
    axis = cross / cross_norm
    angle = float(np.arccos(dot))
    return RotationQuaternion.from_rotation_vector(axis * angle)


def primary_axis_unit(axis: SpatialAxis) -> FloatArray:
    """The axis as a unit vector in the segment's local frame."""
    unit = np.zeros(3, dtype=np.float64)
    unit[axis.index] = float(axis.sign)
    return unit


def _default_perpendicular(direction: FloatArray) -> FloatArray:
    """A deterministic unit vector perpendicular to direction (for the 180-degree case)."""
    reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(direction[0])) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    perpendicular = reference - float(np.dot(reference, direction)) * direction
    return perpendicular / np.linalg.norm(perpendicular)
