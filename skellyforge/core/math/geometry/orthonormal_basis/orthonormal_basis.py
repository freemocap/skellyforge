"""An orthonormal (x, y, z) triad with an origin, and the transforms it defines.

The triad is stored as three `UnitVector` axes rather than a raw matrix, so each axis is
unit length by construction; `__post_init__` then checks the two properties that unit
length alone does not give you - mutual orthogonality and the expected handedness. An
`OrthonormalBasis` that exists is therefore orthonormal, and nothing downstream needs to
re-check it.

The leading dimensions are free, so one static frame is shape `(3,)` per axis and a
motion capture trajectory is `(num_frames, 3)`. Every calculation is fully vectorized
over those leading dimensions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import ORTHONORMALITY_TOLERANCE
from skellyforge.core.math.geometry.orthonormal_basis.handedness import Handedness
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.math.geometry.spatial_vectors import Displacement, Point, UnitVector
from skellyforge.type_overloads import FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class OrthonormalBasis:
    """An orthonormal basis with an origin, expressed in world coordinates.

    Attributes:
        origin: location of the frame's origin in world coordinates.
        x_axis, y_axis, z_axis: the frame's unit axes, in world coordinates.
        handedness: chirality of the triad. A left-handed basis has determinant -1, so
            its matrix is a reflection rather than a pure rotation.
    """

    origin: Point
    x_axis: UnitVector
    y_axis: UnitVector
    z_axis: UnitVector
    handedness: Handedness

    def __post_init__(self) -> None:
        pairwise_dot_products = np.stack(
            arrays=[
                self.x_axis.dot(other=self.y_axis),
                self.y_axis.dot(other=self.z_axis),
                self.z_axis.dot(other=self.x_axis),
            ],
            axis=-1,
        )
        worst_dot_product = float(np.abs(pairwise_dot_products).max())
        if worst_dot_product > ORTHONORMALITY_TOLERANCE:
            raise ValueError(
                f"Basis vectors are not mutually orthogonal - worst |v·w| = "
                f"{worst_dot_product:.3e} > {ORTHONORMALITY_TOLERANCE:.1e}"
            )

        # Scalar triple product: equals the determinant of an orthonormal basis matrix.
        determinants = self.x_axis.dot(other=self.y_axis.cross(other=self.z_axis))
        expected_determinant = self.handedness.expected_determinant
        if np.any(np.abs(determinants - expected_determinant) > ORTHONORMALITY_TOLERANCE):
            raise ValueError(
                f"Basis handedness does not match {self.handedness.name} - determinant "
                f"ranges over [{float(determinants.min()):.6f}, "
                f"{float(determinants.max()):.6f}], expected {expected_determinant:+.1f}"
            )

    @classmethod
    def from_prevalidated_axes(
        cls,
        *,
        origin: Point,
        x_axis: UnitVector,
        y_axis: UnitVector,
        z_axis: UnitVector,
        handedness: Handedness,
    ) -> OrthonormalBasis:
        """Assemble a basis whose orthogonality and handedness are ALREADY established.

        Bypasses `__post_init__`, so the caller is asserting the invariant. Reserved for
        slicing a validated batched basis, where every element was checked together.
        """
        instance = object.__new__(cls)
        for field_name, value in (
            ("origin", origin),
            ("x_axis", x_axis),
            ("y_axis", y_axis),
            ("z_axis", z_axis),
            ("handedness", handedness),
        ):
            object.__setattr__(instance, field_name, value)
        return instance

    def at_batch_index(self, *, index: int) -> OrthonormalBasis:
        """One element of a batched basis, selected along the leading axis.

        Slicing a validated batch cannot produce an invalid element, so this skips
        re-checking orthogonality per slice - which is what makes solving many segments in
        one batched call and splitting the result afterwards actually cheaper than solving
        them one at a time.
        """
        return OrthonormalBasis.from_prevalidated_axes(
            origin=Point.from_prevalidated_array(array=self.origin.array[index]),
            x_axis=UnitVector.from_prevalidated_array(array=self.x_axis.array[index]),
            y_axis=UnitVector.from_prevalidated_array(array=self.y_axis.array[index]),
            z_axis=UnitVector.from_prevalidated_array(array=self.z_axis.array[index]),
            handedness=self.handedness,
        )

    def axis_along(self, *, axis: SpatialAxis) -> UnitVector:
        """The unit axis named by `axis`, flipped when the axis is a negative direction."""
        axis_by_index = (self.x_axis, self.y_axis, self.z_axis)
        direction = axis_by_index[axis.index]
        return direction if axis.sign > 0 else -direction

    @property
    def local_from_world_matrix(self) -> FloatArray:
        """`(..., 3, 3)` orthogonal matrix mapping world-frame vectors into local ones.

        Its rows are the local axes expressed in world coordinates.
        """
        return np.stack(
            arrays=[self.x_axis.array, self.y_axis.array, self.z_axis.array], axis=-2
        )

    @property
    def world_from_local_matrix(self) -> FloatArray:
        """`(..., 3, 3)` orthogonal matrix mapping local-frame vectors into world ones."""
        return np.swapaxes(self.local_from_world_matrix, -1, -2)

    def transform_points_to_local(self, *, points: Point) -> Point:
        """Express world-coordinate points in this frame.

        Args:
            points: `(..., num_points, 3)` world locations, with leading dimensions
                broadcastable against this basis's leading dimensions.

        Returns:
            The same locations expressed in the local frame.
        """
        # World point minus frame origin is the displacement of that point from the
        # origin; rotating that displacement into the local axes gives its local coordinates.
        displacements_from_origin = points - self.origin.expanded_at(axis=-2)
        local_coordinates = np.einsum(
            "...ij,...pj->...pi", self.local_from_world_matrix, displacements_from_origin.array
        )
        return Point.from_prevalidated_array(array=local_coordinates)

    def transform_points_to_world(self, *, points: Point) -> Point:
        """Express local-frame points in world coordinates.

        Args:
            points: `(..., num_points, 3)` local locations.

        Returns:
            The same locations expressed in the world frame.
        """
        # A local coordinate IS a displacement from the frame origin, so rotate it out of
        # the local axes and then walk that displacement from the origin.
        rotated = np.einsum(
            "...ij,...pj->...pi", self.world_from_local_matrix, points.array
        )
        return self.origin.expanded_at(axis=-2) + Displacement.from_prevalidated_array(
            array=rotated
        )

    def transform_named_points_to_local(
        self, *, points: Mapping[str, Point]
    ) -> dict[str, Point]:
        """Express a mapping of named world-coordinate points in this frame."""
        names = list(points.keys())
        stacked = Point.from_prevalidated_array(
            array=np.stack(arrays=[points[name].array for name in names], axis=-2)
        )
        local_points = self.transform_points_to_local(points=stacked)
        return {
            name: Point.from_prevalidated_array(array=local_points.array[..., index, :])
            for index, name in enumerate(names)
        }
