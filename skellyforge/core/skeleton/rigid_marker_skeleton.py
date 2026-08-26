"""Build a skeleton for a rigid marked object: one segment, many markers.

The simple end of the skeleton spectrum. A calibration board, a fiducial-tagged prop, a
rigid cluster on a limb - anything whose markers keep fixed distances from each other and
whose whole pose is one rigid transform. There are no joints, no chains, and no rest pose
to author: the markers are where they are, and the segment's rest pose is identity.

Hydration needs nothing new. A segment with three or more non-collinear landmarks already
rigid-fits (Umeyama) over every marker observed this frame, recovering the object's pose
AND its scale in one closed form - the same code path the skull and pelvis take.

This module knows nothing about charuco, aruco, or any particular object: it takes names
and numbers. That keeps the boundary rule intact (skellyforge never imports skellytracker)
and means the next marked object reuses it as-is rather than growing a second builder.

Positions are in the object's own REFERENCE UNIT, whatever the caller normalizes against -
`1.0` = square length for a charuco board, exactly as `1.0` = body height for the standard
human. The model-scale fit then reports that unit's measured size in millimetres, which for
a board is directly comparable to the square length the user entered at calibration.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_VECTOR_NORM
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.components.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton.components.landmark_grouping import (
    LandmarkConnectionGroup,
    LandmarkGroup,
)
from skellyforge.core.skeleton.components.rigid_body_segment import RigidBodySegment
from skellyforge.core.skeleton.loading.reference_frame_building import (
    build_reference_frame_definition,
)
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.type_overloads import (
    FloatArray,
    LandmarkNameString,
    RigidBodySegmentName,
    SkeletonNameString,
)


def build_rigid_marker_skeleton(
    *,
    name: SkeletonNameString,
    segment_name: RigidBodySegmentName,
    marker_positions: Mapping[LandmarkNameString, FloatArray],
    origin_marker_name: LandmarkNameString,
    primary_marker_name: LandmarkNameString,
    secondary_marker_name: LandmarkNameString,
    marker_definition: str = "a marker on a rigid object",
    landmark_groups: Mapping[str, LandmarkGroup] | None = None,
    landmark_connections: Mapping[str, LandmarkConnectionGroup] | None = None,
    derived_quantities: frozenset[str] = frozenset(),
    coordinate_system: str = "blender",
) -> SkeletonDefinition:
    """One rigid object's markers as a one-segment skeleton.

    The three named markers fix the object's local frame the same way a component YAML's
    `reference_geometry` does - and through the same builder, so a programmatic skeleton
    and an authored one cannot drift into two answers. They are REQUIRED rather than picked
    automatically: many triads of markers would produce a valid frame, so choosing one is a
    guess, not a default, and a guess would silently change the object's local axes the day
    a marker is added.

    Positions are recentred on the origin marker, because a segment's own origin sits at
    the zero of its frame. The caller may therefore pass positions in whatever frame the
    object's geometry is naturally expressed in.

    Args:
        name: the skeleton's name.
        segment_name: the single segment's name (snake_case).
        marker_positions: every marker's `(3,)` position, in the object's reference unit.
        origin_marker_name: the marker that is the segment's origin.
        primary_marker_name: the marker the segment's `+x` axis points exactly at.
        secondary_marker_name: the marker its `+y` axis points approximately at.
        marker_definition: the anatomical-definition prose every marker gets. Markers on a
            board have no anatomy, but a landmark must say what it is.
        landmark_groups: named sets of markers, keyed by group name.
        landmark_connections: named sets of marker edges, keyed by group name.
        derived_quantities: the computed properties this object opts into. Most marked
            objects want none - they have no mass model and their roll is measured, not
            conventional - but nothing here forbids one that does.
        coordinate_system: the convention the positions are expressed in.

    Returns:
        A `SkeletonDefinition` of one segment, ready to hydrate.

    Raises:
        ValueError: fewer than three markers; a named frame marker is not among them; the
            frame markers are collinear (so no frame is recoverable); or a resulting
            segment is not fully specified.
    """
    landmark_groups = landmark_groups or {}
    landmark_connections = landmark_connections or {}

    if len(marker_positions) < MINIMUM_MARKERS_FOR_A_RIGID_FRAME:
        raise ValueError(
            f"skeleton {name!r}: a rigid marker object needs at least "
            f"{MINIMUM_MARKERS_FOR_A_RIGID_FRAME} markers to have a frame at all - got "
            f"{sorted(marker_positions)}"
        )
    frame_marker_names = (origin_marker_name, primary_marker_name, secondary_marker_name)
    missing = [
        marker_name
        for marker_name in frame_marker_names
        if marker_name not in marker_positions
    ]
    if missing:
        raise ValueError(
            f"skeleton {name!r}: the markers naming its frame are not among its markers - "
            f"{missing}. Known markers: {sorted(marker_positions)}"
        )
    if len(set(frame_marker_names)) != len(frame_marker_names):
        raise ValueError(
            f"skeleton {name!r}: the origin, primary and secondary markers must be three "
            f"different markers - got {frame_marker_names}"
        )

    origin_position = np.asarray(marker_positions[origin_marker_name], dtype=np.float64)
    _raise_unless_frame_markers_span_a_plane(
        name=name,
        origin_position=origin_position,
        primary_position=np.asarray(
            marker_positions[primary_marker_name], dtype=np.float64
        ),
        secondary_position=np.asarray(
            marker_positions[secondary_marker_name], dtype=np.float64
        ),
    )

    landmarks = {
        marker_name: AnatomicalLandmark(
            name=marker_name,
            anatomical_definition=marker_definition,
            # Recentred: the origin marker sits at the zero of the segment's own frame,
            # which `RigidBodySegment` enforces and hydration relies on.
            local_position=Point.from_prevalidated_array(
                array=np.asarray(position, dtype=np.float64) - origin_position
            ),
            segment=segment_name,
        )
        for marker_name, position in marker_positions.items()
    }
    frame_definition = build_reference_frame_definition(
        segment_name=segment_name,
        reference_geometry={
            "origin": origin_marker_name,
            "x_axis": {"landmark": primary_marker_name, "type": "exact"},
            "y_axis": {"landmark": secondary_marker_name, "type": "approximate"},
        },
    )
    segment = RigidBodySegment(
        name=segment_name, landmarks=landmarks, frame_definition=frame_definition
    )
    if not segment.supports_rigid_fit:
        raise ValueError(
            f"skeleton {name!r}: segment {segment_name!r} does not support a rigid fit - "
            "its markers are collinear, so no pose is recoverable from them however many "
            "are observed"
        )
    return SkeletonDefinition(
        name=name,
        landmarks=landmarks,
        segments={segment_name: segment},
        landmark_groups=dict(landmark_groups),
        landmark_connections=dict(landmark_connections),
        derived_quantities=derived_quantities,
        coordinate_system=coordinate_system,
    )


MINIMUM_MARKERS_FOR_A_RIGID_FRAME: int = 3


def _raise_unless_frame_markers_span_a_plane(
    *,
    name: SkeletonNameString,
    origin_position: FloatArray,
    primary_position: FloatArray,
    secondary_position: FloatArray,
) -> None:
    """Refuse a frame whose three markers lie on one line.

    Checked here, on the authored geometry, rather than left to the first frame that tries
    to hydrate: a collinear triad is a mistake in the object's definition, and it should
    fail when the object is built, naming the object.
    """
    primary_direction = primary_position - origin_position
    secondary_direction = secondary_position - origin_position
    spanned_area = float(np.linalg.norm(np.cross(primary_direction, secondary_direction)))
    if spanned_area < MINIMUM_VECTOR_NORM:
        raise ValueError(
            f"skeleton {name!r}: its origin, primary and secondary markers are collinear "
            f"(they span an area of {spanned_area:.3e}), so they define no frame. Pick "
            "three markers that do not lie on one line."
        )
