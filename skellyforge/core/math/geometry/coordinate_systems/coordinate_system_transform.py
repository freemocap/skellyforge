"""Converting positions, rotations, and quaternions between coordinate-system conventions.

A convention is a pure axis-relabelling, so conversion is a single orthogonal matrix with
no translation term. The matrix has determinant +1 when the two conventions share
handedness, and -1 when converting across a handedness flip (into or out of a left-handed
system like Unreal or Unity).

A second-order tensor such as an inertia tensor transforms by the similarity
M @ tensor @ M.T; compose it from conversion_matrix when the inertia work lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skellyforge.core.math.geometry.coordinate_systems.coordinate_system_convention import (
    CoordinateSystemConvention,
)
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Displacement, Point
from skellyforge.type_overloads import FloatArray


def conversion_matrix(
    *, from_convention: CoordinateSystemConvention, to_convention: CoordinateSystemConvention
) -> FloatArray:
    """The 3-by-3 matrix M such that M applied to from-convention coordinates yields to-convention coordinates."""
    return to_convention.from_canonical_matrix @ from_convention.to_canonical_matrix


@dataclass(frozen=True, slots=True, eq=False)
class CoordinateSystemTransform:
    """A bound conversion from one convention to another, reusable across many values.

    Precomputes the matrix once at construction so a per-frame loop pays a single matmul
    rather than rebuilding the matrix each call.
    """

    from_convention: CoordinateSystemConvention
    to_convention: CoordinateSystemConvention
    matrix: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matrix",
            conversion_matrix(
                from_convention=self.from_convention, to_convention=self.to_convention
            ),
        )

    def convert_point(self, *, point: Point) -> Point:
        """A location expressed in the destination convention. Handles any batch shape."""
        return Point.from_prevalidated_array(
            array=np.einsum("ij,...j->...i", self.matrix, point.array)
        )

    def convert_displacement(self, *, displacement: Displacement) -> Displacement:
        """A displacement expressed in the destination convention. Handles any batch shape."""
        return Displacement.from_prevalidated_array(
            array=np.einsum("ij,...j->...i", self.matrix, displacement.array)
        )

    def convert_rotation_matrix(self, *, rotation_matrix: FloatArray) -> FloatArray:
        """A 3-by-3 rotation expressed in the destination convention (similarity transform)."""
        rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
        if rotation_matrix.shape != (3, 3):
            raise ValueError(f"rotation matrix must be (3, 3), got {rotation_matrix.shape}")
        return self.matrix @ rotation_matrix @ self.matrix.T

    def convert_quaternion(self, *, quaternion: RotationQuaternion) -> RotationQuaternion:
        """A rotation expressed in the destination convention.

        Routes through the rotation-matrix form, which stays correct across a handedness
        flip (a reflection cannot be carried by quaternion conjugation alone).
        """
        return RotationQuaternion.from_rotation_matrix(
            matrix=self.convert_rotation_matrix(rotation_matrix=quaternion.to_rotation_matrix())
        )
