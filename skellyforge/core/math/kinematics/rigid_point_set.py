"""Closed-form similarity alignment of point sets (Umeyama).

A segment with many landmarks - the skull, the pelvis - is a rigid body: every pair of
landmarks keeps a fixed distance. Given the reference local positions and the observed world
positions, the Umeyama algorithm recovers the single similarity transform (uniform scale,
then rotation, then translation) that best maps the reference onto the observed points in
the least-squares sense. The closed form is exact, needs no iteration, and is what
"rigidify all the points" means for these segments.

The scale is not an optional refinement, it is the reason this is a SIMILARITY and not a
rigid motion. The authored template is dimensionless - a landmark's local position is a
fraction of body height, not a length - so the map from a segment's own frame into the world
carries a size as well as a pose. Solving without it does not merely lose the size: the
translation absorbs the mismatch (`observed_centroid - rotation @ reference_centroid` puts
the origin on the observed centroid when the reference is a thousand times smaller), so the
segment's origin lands in the wrong place. The scale is recovered from the same SVD as the
rotation, at the cost of one trace.

What the scale MEANS, therefore, is millimetres per unit body height - this segment's own
estimate of how big the subject is. `core.skeleton.pose.body_scale_fitting` is what pools
those per-segment estimates into one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import (
    MINIMUM_RELATIVE_SINGULAR_VALUE,
    MINIMUM_VECTOR_NORM,
)
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Displacement, Point
from skellyforge.core.math.geometry.transform_math import Transform
from skellyforge.type_overloads import LandmarkNameString

MINIMUM_POINTS_FOR_RIGID_FIT: int = 3


@dataclass(frozen=True, slots=True, eq=False)
class SimilarityFit:
    """A similarity transform, split into its scale and its rigid part.

    The two are kept apart rather than folded together because they answer different
    questions and are consumed by different layers. The `transform` places the segment -
    its translation IS the segment's world origin, since a segment's own origin sits at the
    zero of its frame - and is what hydration returns. The `scale` sizes it, and is what the
    body-scale fit pools across segments.

    Attributes:
        transform: the rotation and translation. Applying it to the local origin gives the
            segment's world origin, with the scale already accounted for.
        scale: the uniform scale mapping reference units onto observed units. For the
            standard human's dimensionless template that is millimetres per unit body
            height.
    """

    transform: Transform
    scale: float

    def apply(self, *, points: Point) -> Point:
        """Scale `points` about the reference origin, rotate them, then translate."""
        return self.transform.apply(
            points=Point.from_prevalidated_array(array=self.scale * points.array)
        )


def align_point_sets_similarity(*, reference: Point, observed: Point) -> SimilarityFit:
    """The similarity transform mapping reference points onto observed points.

    Minimizes the sum of squared distances between the transformed reference points and the
    observed points, in closed form (no iteration). The rotation is a pure rotation - a
    reflection is explicitly rejected, because a body cannot mirror - and the scale is a
    single positive number, because a body cannot be stretched along one axis alone.

    Args:
        reference: (n, 3) points in the reference (local) frame.
        observed: (n, 3) matching points in the observed (world) frame.

    Returns:
        The fit that, applied to reference, lands nearest the observed points.

    Raises:
        ValueError: the two point sets have different shapes, there are fewer than three
            points, the reference points are collinear (so no full rotation is recoverable)
            or coincident (so no scale is), or the recovered scale is not positive.
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
    # A relative threshold, not machine epsilon: a point set that is collinear to a
    # part in 10^12 yields a roll that is amplified rounding error, not a measurement.
    if singular_values[1] <= MINIMUM_RELATIVE_SINGULAR_VALUE * singular_values[0]:
        raise ValueError(
            "points are collinear - no full rotation is recoverable from them "
            f"(second singular value is {singular_values[1]:.3e} against a largest of "
            f"{singular_values[0]:.3e}, a ratio below "
            f"{MINIMUM_RELATIVE_SINGULAR_VALUE:.1e})"
        )

    determinant = float(np.linalg.det(right_transpose.T @ left.T))
    reflection_sign = 1.0 if determinant >= 0.0 else -1.0
    rotation_matrix = right_transpose.T @ np.diag([1.0, 1.0, reflection_sign]) @ left.T

    # Umeyama's scale: the singular values are the aligned covariance between the two
    # point sets, so their (reflection-signed) sum over the reference's own spread is how
    # much bigger the observed set is. The reflected axis subtracts, matching the rotation.
    reference_spread = float(np.sum(reference_centered**2))
    if reference_spread < MINIMUM_VECTOR_NORM:
        raise ValueError(
            "reference points are coincident - they have no spread for a scale to be "
            f"measured against (summed squared deviation {reference_spread:.3e} < "
            f"{MINIMUM_VECTOR_NORM:.1e})"
        )
    signed_singular_values = singular_values * np.array([1.0, 1.0, reflection_sign])
    scale = float(np.sum(signed_singular_values) / reference_spread)
    # A reflected configuration subtracts its smallest singular value, and a set mirrored
    # enough can drive the sum to zero or below. There is no such body, so the fit refuses
    # to report one rather than handing back a segment of negative size.
    if not scale > 0.0:
        raise ValueError(
            f"recovered a non-positive scale ({scale}) - the observed points do not "
            "resemble the reference set under any similarity transform"
        )

    translation = observed_centroid - scale * (rotation_matrix @ reference_centroid)

    return SimilarityFit(
        transform=Transform(
            rotation=RotationQuaternion.from_rotation_matrix(matrix=rotation_matrix),
            translation=Displacement.from_prevalidated_array(array=translation),
        ),
        scale=scale,
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

    def fit_pose(self, *, observed: Mapping[LandmarkNameString, Point]) -> SimilarityFit:
        """The similarity transform placing this set onto the observed points.

        Landmarks absent from observed are treated as missing and left out of the fit, so a
        partially occluded segment still solves as long as three non-collinear landmarks are
        visible.

        Args:
            observed: observed world positions, keyed by landmark name.

        Returns:
            The fit mapping the reference positions onto the observed ones.

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
        return align_point_sets_similarity(
            reference=Point.from_prevalidated_array(array=reference_arrays),
            observed=Point.from_prevalidated_array(array=observed_arrays),
        )
