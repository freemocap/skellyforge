"""Hydrate a skeleton: recover each segment's pose from observed landmark positions.

This is the closed-form, single-frame core of Phase 3. A segment with three or more observed
landmarks is treated as one rigid body and fit with Kabsch over all of them; a segment with
only its origin and primary observed yields just a direction, turned into an orientation by
the shortest-arc rotation. The input is a mapping of landmark name to world position, which
freemocap fills from tracker keypoints - this module never sees keypoints.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from skellyforge.core.math.geometry.spatial_vectors import Point, UnitVector
from skellyforge.core.math.kinematics.coordinate_frame_ops import rotation_between_vectors
from skellyforge.core.math.kinematics.rigid_point_set import (
    MINIMUM_POINTS_FOR_RIGID_FIT,
    RigidPointSet,
)
from skellyforge.core.skeleton_parts.rigid_body_segment import RigidBodySegment
from skellyforge.core.skeleton_parts.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton_parts.skeleton_pose import SegmentPose, SkeletonPose
from skellyforge.type_overloads import LandmarkNameString


def hydrate_segment(
    *, segment: RigidBodySegment, observed: Mapping[LandmarkNameString, Point]
) -> SegmentPose:
    """Recover one segment's pose from observed landmark positions.

    A segment with three or more observed landmarks is rigid-fit with Kabsch over all of
    them; otherwise its origin-to-primary direction is recovered, with roll left free.

    Args:
        segment: the segment to hydrate.
        observed: observed landmark positions, keyed by landmark name.

    Returns:
        The segment's pose.

    Raises:
        ValueError: the segment has neither three observed landmarks nor both its origin and
            primary observed, so no pose can be recovered.
    """
    owned_names = tuple(segment.landmarks.keys())
    observed_owned = [name for name in owned_names if name in observed]

    if len(observed_owned) >= MINIMUM_POINTS_FOR_RIGID_FIT:
        reference_positions = Point.from_prevalidated_array(
            array=np.stack(
                [segment.landmarks[name].local_position.array for name in owned_names],
                axis=0,
            )
        )
        point_set = RigidPointSet(
            point_names=owned_names, reference_positions=reference_positions
        )
        try:
            transform = point_set.fit_pose(observed=observed)
        except ValueError:
            # Collinear landmarks (a straight spine, say) cannot fix the roll; fall
            # through to the origin-to-primary direction below instead.
            pass
        else:
            return SegmentPose(
                segment_name=segment.name,
                origin=transform.apply(points=Point.from_xyz(x=0.0, y=0.0, z=0.0)),
                orientation=transform.rotation,
            )

    origin_name = segment.frame_definition.origin_point_name
    primary_name = segment.frame_definition.primary_point_name
    if origin_name in observed and primary_name in observed:
        world_direction = segment.calculate_direction(points=observed)
        primary_local = segment.landmarks[primary_name].local_position.array
        local_primary = UnitVector.from_prevalidated_array(
            array=primary_local / np.linalg.norm(primary_local)
        )
        return SegmentPose(
            segment_name=segment.name,
            origin=observed[origin_name],
            orientation=rotation_between_vectors(
                from_direction=local_primary, to_direction=world_direction
            ),
        )

    raise ValueError(
        f"segment {segment.name!r}: cannot hydrate - needs at least three observed "
        f"landmarks or both origin and primary observed; observed owned landmarks: "
        f"{observed_owned}"
    )


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
