"""Closed-form rigid alignment of point sets (Kabsch).

A segment with many landmarks - the skull, the pelvis - is a rigid body: every pair of
landmarks keeps a fixed distance. Given the reference local positions and the observed world
positions, the Kabsch algorithm recovers the single rigid transform (rotation, then
translation) that best maps the reference onto the observed points in the least-squares
sense. The closed form is exact, needs no iteration, and is what "rigidify all the points"
means for these segments.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Displacement, Point
from skellyforge.core.math.geometry.transform_math import Transform
from skellyforge.type_overloads import LandmarkNameString

MINIMUM_POINTS_FOR_RIGID_FIT: int = 3


def align_point_sets_kabsch(*, reference: Point, observed: Point) -> Transform:
    """The rigid transform mapping reference points onto observed points.

    Minimizes the sum of squared distances between the transformed reference points and the
    observed points, in closed form (no iteration). The result is a pure rotation plus a
    translation - a reflection is explicitly rejected, because a rigid body cannot mirror.

    Args:
        reference: (n, 3) points in the reference (local) frame.
        observed: (n, 3) matching points in the observed (world) frame.

    Returns:
        The transform that, applied to reference, lands nearest the observed points.

    Raises:
        ValueError: the two point sets have different shapes, there are fewer than three
            points, or the points are collinear (so no full rotation is recoverable).
    """
    reference_array = reference.array
    observed_array = observed.array
    if reference_array.shape != observed_array.shape:
        raise ValueError(
            f"reference and observed must have the same shape - got "
            f"{reference_array.shape} and {observed_array.shape}"
        )
    if reference_array.ndim != 2 or reference_array.shape[1] != 3:
        raise ValueError(
            f"points must have shape (n, 3) - got {reference_array.shape}"
        )
    if reference_array.shape[0] < MINIMUM_POINTS_FOR_RIGID_FIT:
        raise ValueError(
            f"a rigid fit needs at least {MINIMUM_POINTS_FOR_RIGID_FIT} points - got "
            f"{reference_array.shape[0]}"
        )

    reference_centroid = reference_array.mean(axis=0)
    observed_centroid = observed_array.mean(axis=0)
    reference_centered = reference_array - reference_centroid
    observed_centered = observed_array - observed_centroid

    cross_covariance = reference_centered.T @ observed_centered
    left, singular_values, right_transpose = np.linalg.svd(cross_covariance)
    if singular_values[1] <= np.finfo(np.float64).eps * singular_values[0]:
        raise ValueError(
            "points are collinear - no full rotation is recoverable from them"
        )

    determinant = float(np.linalg.det(right_transpose.T @ left.T))
    reflection_sign = 1.0 if determinant >= 0.0 else -1.0
    rotation_matrix = right_transpose.T @ np.diag([1.0, 1.0, reflection_sign]) @ left.T

    translation = observed_centroid - rotation_matrix @ reference_centroid

    return Transform(
        rotation=RotationQuaternion.from_rotation_matrix(rotation_matrix),
        translation=Displacement.from_prevalidated_array(array=translation),
    )


@dataclass(frozen=True, slots=True, eq=False)
class RigidPointSet:
    """A named set of reference points that move together as one rigid body.

    Attributes:
        point_names: the landmark names, in the same order as the reference positions.
        reference_positions: the (n, 3) reference (local) positions of those landmarks.
    """

    point_names: tuple[LandmarkNameString, ...]
    reference_positions: Point

    def __post_init__(self) -> None:
        if not self.point_names:
            raise ValueError("a rigid point set needs at least one point")
        if self.reference_positions.batch_shape != (len(self.point_names),):
            raise ValueError(
                f"reference_positions must have one row per point name - got shape "
                f"{self.reference_positions.array.shape} for {len(self.point_names)} names"
            )

    def fit_pose(self, *, observed: Mapping[LandmarkNameString, Point]) -> Transform:
        """The rigid transform placing this set onto the observed points.

        Landmarks absent from observed are treated as missing and left out of the fit, so a
        partially occluded segment still solves as long as three non-collinear landmarks are
        visible.

        Args:
            observed: observed world positions, keyed by landmark name.

        Returns:
            The rigid transform mapping the reference positions onto the observed ones.

        Raises:
            ValueError: fewer than three of this set's landmarks are observed.
        """
        observed_names = [name for name in self.point_names if name in observed]
        if len(observed_names) < MINIMUM_POINTS_FOR_RIGID_FIT:
            raise ValueError(
                f"need at least {MINIMUM_POINTS_FOR_RIGID_FIT} observed landmarks to fit, "
                f"got {observed_names} of {self.point_names}"
            )
        name_to_index = {name: index for index, name in enumerate(self.point_names)}
        reference_arrays = np.stack(
            [self.reference_positions.array[name_to_index[name]] for name in observed_names],
            axis=0,
        )
        observed_arrays = np.stack(
            [observed[name].array for name in observed_names], axis=0
        )
        return align_point_sets_kabsch(
            reference=Point.from_prevalidated_array(array=reference_arrays),
            observed=Point.from_prevalidated_array(array=observed_arrays),
        )
