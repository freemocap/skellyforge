"""Hydrate a skeleton: recover each segment's pose from observed landmark positions.

This is the closed-form, single-frame core of Phase 3. A segment whose own landmarks span
a plane is treated as one rigid body and fit with Kabsch over every one of them that is
observed; a segment that only ever pins a line - the limbs, the spine - yields a direction,
turned into an orientation by the shortest-arc rotation, with roll left free for a
downstream roll convention to resolve. The input is a mapping of landmark name to world
position, which freemocap fills from tracker keypoints - this module never sees keypoints.

Which of the two a segment gets is a STATIC property of the segment, answered by
`RigidBodySegment.supports_rigid_fit`, not a runtime accident. If a segment that can be
rigid-fit is handed degenerate observations, that is an error and it is raised, not
quietly downgraded to a direction fit.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_VECTOR_NORM
from skellyforge.core.math.geometry.spatial_vectors import Point, UnitVector
from skellyforge.core.math.kinematics.coordinate_frame_ops import rotation_between_vectors
from skellyforge.core.math.kinematics.rigid_point_set import (
    MINIMUM_POINTS_FOR_RIGID_FIT,
    RigidPointSet,
)
from skellyforge.core.skeleton_parts.rigid_body_segment import RigidBodySegment
from skellyforge.core.skeleton_parts.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton_parts.skeleton_pose import (
    PoseSolution,
    SegmentPose,
    SkeletonPose,
)
from skellyforge.type_overloads import LandmarkNameString


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
        reference_positions = Point.from_prevalidated_array(
            array=np.stack(
                [segment.landmarks[name].local_position.array for name in owned_names],
                axis=0,
            )
        )
        point_set = RigidPointSet(
            point_names=owned_names, reference_positions=reference_positions
        )
        transform = point_set.fit_pose(observed=observed)
        return SegmentPose(
            segment_name=segment.name,
            origin=transform.apply(points=Point.from_xyz(x=0.0, y=0.0, z=0.0)),
            orientation=transform.rotation,
            solved_by=PoseSolution.RIGID_FIT,
        )

    origin_name = segment.frame_definition.origin_point_name
    primary_name = segment.frame_definition.primary_point_name
    if origin_name in observed and primary_name in observed:
        return SegmentPose(
            segment_name=segment.name,
            origin=observed[origin_name],
            orientation=rotation_between_vectors(
                from_direction=_local_primary_direction(segment=segment),
                to_direction=segment.calculate_direction(points=observed),
            ),
            solved_by=PoseSolution.DIRECTION,
        )

    raise ValueError(
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
    *, skeleton: SkeletonDefinition, observed: Mapping[LandmarkNameString, Point]
) -> SkeletonPose:
    """Recover every segment's pose from observed landmark positions.

    Args:
        skeleton: the skeleton whose segments to hydrate.
        observed: observed landmark positions, keyed by landmark name.

    Returns:
        The skeleton's pose, keyed by segment name.
    """
    return SkeletonPose(
        segment_poses={
            segment.name: hydrate_segment(segment=segment, observed=observed)
            for segment in skeleton.segments.values()
        }
    )
