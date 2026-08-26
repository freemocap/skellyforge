"""Hydrate a skeleton: recover each segment's pose from observed landmark positions.

This is the closed-form, single-frame core of Phase 3. A segment whose own landmarks span
a plane is treated as one rigid body and fit with Umeyama over every one of them that is
observed; a segment that only ever pins a line - the limbs, the spine - yields a direction,
turned into an orientation by the shortest-arc rotation, with roll left free for a
downstream roll convention to resolve. The input is a mapping of landmark name to world
position, which freemocap fills from tracker keypoints - this module never sees keypoints.

Which of the two a segment gets is a STATIC property of the segment, answered by
`RigidBodySegment.supports_rigid_fit`, not a runtime accident. If a segment that can be
rigid-fit is handed degenerate observations, that is an error and it is raised, not
quietly downgraded to a direction fit.

Both closed forms also recover a SCALE, because the authored template is dimensionless: a
local position is a fraction of body height, so placing a segment in a world measured in
millimetres means answering how big it is as well as where and which way. The rigid fit
reads that off every observed landmark at once; the direction fit reads it off the one
distance it has. Neither pools it across segments - that is `body_scale_fitting`'s job.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_VECTOR_NORM
from skellyforge.core.math.geometry.orthonormal_basis.calculate_orthonormal_basis import (
    direction_along,
)
from skellyforge.core.math.geometry.spatial_vectors import Point, UnitVector
from skellyforge.core.math.kinematics.coordinate_frame_ops import rotation_between_vectors
from skellyforge.core.math.kinematics.rigid_point_set import (
    MINIMUM_POINTS_FOR_RIGID_FIT,
)
from skellyforge.core.skeleton.components.rigid_body_segment import RigidBodySegment
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import (
    PoseSolution,
    SegmentPose,
    SkeletonPose,
)
from skellyforge.type_overloads import LandmarkNameString


class MissingLandmarkObservations(ValueError):
    """A segment cannot hydrate because too few of its landmarks are observed.

    Distinct from a degenerate-observation error: a segment whose landmarks ARE
    observed but collinear (so a rigid fit or direction is undefined) still fails
    loudly, while a segment whose landmarks are simply not tracked is a missing
    observation and may be skipped by a streaming caller.
    """


class DegenerateObservations(ValueError):
    """A segment's observed landmarks are degenerate, so no full rotation is recoverable.

    Distinct from a missing-observation error: the landmarks ARE tracked, but they
    collapse onto a line (e.g. a hand seen edge-on), so a rigid fit is undefined. A
    streaming caller may skip this segment the same way it skips a missing one.
    """


def hydrate_segment(
    *, segment: RigidBodySegment, observed: Mapping[LandmarkNameString, Point]
) -> SegmentPose:
    """Recover one segment's pose from observed landmark positions.

    Args:
        segment: the segment to hydrate.
        observed: observed landmark positions, keyed by landmark name.

    Returns:
        The segment's pose, carrying which closed form produced it.

    Raises:
        ValueError: the segment has neither enough observed landmarks for the rigid fit it
            supports nor both its origin and primary observed; or its observed landmarks
            are degenerate; or its primary landmark sits on its own origin, so there is no
            local direction to rotate.
    """
    owned_names = tuple(segment.landmarks.keys())
    observed_owned = tuple(name for name in owned_names if name in observed)

    if segment.supports_rigid_fit and len(observed_owned) >= MINIMUM_POINTS_FOR_RIGID_FIT:
        try:
            fit = segment.rigid_point_set.fit_pose(observed=observed)
        except ValueError as error:
            raise DegenerateObservations(
                f"segment {segment.name!r}: observed landmarks are degenerate "
                f"(collinear/coplanar) - {error}"
            ) from error
        return SegmentPose(
            segment_name=segment.name,
            # The segment's own origin is the zero of its frame, so the fit's translation
            # IS its world origin - and only once the fit carries a scale, because an
            # unscaled fit hides the size mismatch in exactly this translation.
            origin=fit.apply(points=Point.from_xyz(x=0.0, y=0.0, z=0.0)),
            orientation=fit.transform.rotation,
            body_scale_estimate=fit.scale,
            solved_by=PoseSolution.RIGID_FIT,
        )

    origin_name = segment.frame_definition.origin_point_name
    primary_name = segment.frame_definition.primary_point_name
    if origin_name in observed and primary_name in observed:
        # One displacement answers both of this branch's questions - which way the segment
        # runs, and how big it is - so it is measured once. Two measurements could disagree
        # about whether the pair is degenerate; one cannot.
        displacement = observed[primary_name] - observed[origin_name]
        observed_length = float(displacement.norm())
        # Two landmarks that land on each other give neither a direction nor a size. Named
        # as its own failure rather than left to the normalization below, so a streaming
        # caller can skip this segment the same way it skips a collinear one.
        if observed_length < MINIMUM_VECTOR_NORM:
            raise DegenerateObservations(
                f"segment {segment.name!r}: its observed origin {origin_name!r} and "
                f"primary {primary_name!r} are {observed_length:.3e} apart, which is not a "
                f"usable distance (needs >= {MINIMUM_VECTOR_NORM:.1e}) - the segment has "
                "neither a direction nor a size"
            )
        return SegmentPose(
            segment_name=segment.name,
            origin=observed[origin_name],
            orientation=rotation_between_vectors(
                from_direction=_local_primary_direction(segment=segment),
                to_direction=direction_along(
                    axis=segment.frame_definition.primary_axis,
                    displacement=displacement,
                    description=(
                        f"segment {segment.name!r}'s primary axis (`{origin_name}` -> "
                        f"`{primary_name}`)"
                    ),
                ),
            ),
            # `segment.length` is the authored origin-to-primary distance as a fraction of
            # body height, nonzero by construction - a primary landmark sitting on its own
            # origin is refused at load - so this ratio is world units per unit body
            # height: this segment's own reading of how big the subject is.
            body_scale_estimate=observed_length / segment.length,
            solved_by=PoseSolution.DIRECTION,
        )

    raise MissingLandmarkObservations(
        f"segment {segment.name!r}: cannot hydrate - it needs either at least "
        f"{MINIMUM_POINTS_FOR_RIGID_FIT} observed landmarks (it supports a rigid fit: "
        f"{segment.supports_rigid_fit}) or both its origin {origin_name!r} and its "
        f"primary {primary_name!r} observed; observed owned landmarks: {list(observed_owned)}"
    )


def _local_primary_direction(*, segment: RigidBodySegment) -> UnitVector:
    """The segment's local PRIMARY AXIS direction, in its own frame.

    The primary landmark's local position is the displacement from the origin, because a
    segment's origin is the zero of its own frame - but the frame's primary axis is that
    displacement signed by the declared axis, which is what `calculate_direction` returns
    on the world side. Both sides have to apply the sign or a segment whose primary axis is
    a NEGATIVE_* one hydrates a half turn out, silently and only on that segment.

    A primary landmark sitting on the origin would make the direction undefined, so it is
    refused by name rather than divided into a NaN.
    """
    primary_local = segment.landmarks[
        segment.frame_definition.primary_point_name
    ].local_position.array
    norm = float(np.linalg.norm(primary_local))
    if norm < MINIMUM_VECTOR_NORM:
        raise ValueError(
            f"segment {segment.name!r}: its primary landmark "
            f"{segment.frame_definition.primary_point_name!r} sits on the segment's own "
            f"origin (local norm {norm:.3e} < {MINIMUM_VECTOR_NORM:.1e}), so the segment "
            "has no local direction and its length is zero"
        )
    signed = float(segment.frame_definition.primary_axis.sign) * primary_local / norm
    return UnitVector.from_prevalidated_array(array=signed)


def hydrate_skeleton(
    *,
    skeleton: SkeletonDefinition,
    observed: Mapping[LandmarkNameString, Point],
    require_all: bool = True,
) -> SkeletonPose:
    """Recover every segment's pose from observed landmark positions.

    Args:
        skeleton: the skeleton whose segments to hydrate.
        observed: observed landmark positions, keyed by landmark name.
        require_all: when True (default) a segment whose landmarks are missing or
            degenerate raises :class:`MissingLandmarkObservations` /\
            :class:`DegenerateObservations`; when False such segments are omitted from
            the returned pose, so a streaming tracker with partial or degenerate
            observations (an occluded hand, a face out of frame, a hand seen edge-on)
            hydrates the segments it can and drops the rest.

    Returns:
        The skeleton's pose, keyed by segment name. Every segment is present when
        ``require_all``; otherwise only the hydratable ones.
    """
    segment_poses = {}
    for segment in skeleton.segments.values():
        try:
            segment_poses[segment.name] = hydrate_segment(
                segment=segment, observed=observed
            )
        except (MissingLandmarkObservations, DegenerateObservations):
            if require_all:
                raise
    return SkeletonPose(segment_poses=segment_poses)
