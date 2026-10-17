"""Rigid transforms: a rotation and a translation, and what they do to points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Displacement, Point

@dataclass(frozen=True, slots=True, eq=False)
class Transform:
    """A rigid transform: rotate about the origin, then translate.

    Attributes:
        rotation: the rotation applied first, about the frame origin.
        translation: the displacement applied after rotating.
    """

    rotation: RotationQuaternion
    translation: Displacement

    def __post_init__(self) -> None:
        if self.translation.batch_shape != ():
            raise ValueError(
                "A Transform holds a single translation - got a batched Displacement "
                f"of shape {self.translation.array.shape}"
            )

    @classmethod
    def identity(cls) -> Transform:
        """The transform that leaves every point where it is."""
        return cls(
            rotation=RotationQuaternion.identity(),
            translation=Displacement.from_xyz(x=0.0, y=0.0, z=0.0),
        )

    def apply(self, *, points: Point) -> Point:
        """Rotate `points` about the origin, then translate them.

        Args:
            points: `(..., 3)` locations, batched over any leading dimensions.

        Returns:
            The transformed `(..., 3)` locations.
        """
        rotated = np.einsum(
            "ij,...j->...i", self.rotation.to_rotation_matrix(), points.array
        )
        return Point.from_prevalidated_array(array=rotated + self.translation.array)

    def inverse(self) -> Transform:
        """The transform that undoes this one."""
        inverse_rotation = self.rotation.inverse()
        undone_translation = np.einsum(
            "ij,j->i", inverse_rotation.to_rotation_matrix(), -self.translation.array
        )
        return Transform(
            rotation=inverse_rotation,
            translation=Displacement.from_prevalidated_array(array=undone_translation),
        )
