"""A coordinate-system convention: which way its +X/+Y/+Z axes point.

A convention is pure axis-labelling - it has no origin and no unit choice, only "which
anatomical direction is +X", and so on. Handedness is derived from that labelling and is
not a separate input. A left-handed convention (Unreal, Unity) is represented exactly like
a right-handed one; its basis matrix simply has determinant -1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.coordinate_systems.anatomical_direction import (
    AnatomicalDirection,
)
from skellyforge.core.math.geometry.numeric_tolerances import ORTHONORMALITY_TOLERANCE
from skellyforge.core.math.geometry.orthonormal_basis.handedness import Handedness
from skellyforge.type_overloads import FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class CoordinateSystemConvention:
    """Which anatomical direction each of a coordinate system's positive axes points along.

    Attributes:
        name: unique name, e.g. "blender".
        description: human-readable prose describing the convention.
        x_direction: the anatomical direction +X points along.
        y_direction: the anatomical direction +Y points along.
        z_direction: the anatomical direction +Z points along.
    """

    name: str
    description: str
    x_direction: AnatomicalDirection
    y_direction: AnatomicalDirection
    z_direction: AnatomicalDirection

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a coordinate system convention needs a name")
        gram = self.to_canonical_matrix.T @ self.to_canonical_matrix
        worst_off_diagonal = float(np.abs(gram - np.eye(3)).max())
        if worst_off_diagonal > ORTHONORMALITY_TOLERANCE:
            raise ValueError(
                f"coordinate system {self.name!r}: +X, +Y, and +Z must point along three "
                f"mutually perpendicular directions - got "
                f"{self.x_direction.value}, {self.y_direction.value}, {self.z_direction.value}"
            )

    @property
    def to_canonical_matrix(self) -> FloatArray:
        """The 3-by-3 matrix whose columns are this convention's x, y, z axes.

        The columns are expressed in the canonical (Blender) frame, so multiplying this
        matrix by coordinates in this convention yields canonical coordinates. Orthogonal,
        with determinant +1 (right-handed) or -1 (left-handed).
        """
        return np.column_stack(
            [
                self.x_direction.unit_vector,
                self.y_direction.unit_vector,
                self.z_direction.unit_vector,
            ]
        )

    @property
    def from_canonical_matrix(self) -> FloatArray:
        """The 3-by-3 matrix mapping canonical coordinates into this convention.

        It is the transpose of to_canonical_matrix, and its inverse (the matrix is
        orthogonal), so multiplying it by canonical coordinates yields coordinates in this
        convention.
        """
        return self.to_canonical_matrix.T

    @property
    def handedness(self) -> Handedness:
        """Whether this convention is right- or left-handed, derived from its axes."""
        determinant = float(np.linalg.det(self.to_canonical_matrix))
        return Handedness.RIGHT_HANDED if determinant > 0.0 else Handedness.LEFT_HANDED

    def __str__(self) -> str:
        handedness_label = (
            "right" if self.handedness is Handedness.RIGHT_HANDED else "left"
        )
        return (
            f"{self.name}: +X {self.x_direction.value}, +Y {self.y_direction.value}, "
            f"+Z {self.z_direction.value} ({handedness_label}-handed)"
        )
