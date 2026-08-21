"""A fully specified rigid-body segment: a local reference frame solved from 3+ landmarks.

The segment owns a `ReferenceFrameDefinition` naming which of its landmarks sit on which
signed axes, so solving its frame is exactly `calculate_orthonormal_basis`. Its length is
derived from its landmarks' rest positions.

For streaming, prefer `calculate_bases_for_segments` over calling the solver per segment.
The solver's cost is dominated by fixed per-call overhead rather than by batch size, so
solving twenty segments in one batched call is roughly twenty times cheaper than twenty
separate calls - see that function's docstring for measured numbers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.orthonormal_basis.calculate_orthonormal_basis import (
    calculate_orthonormal_basis,
)
from skellyforge.core.math.geometry.orthonormal_basis.handedness import Handedness
from skellyforge.core.math.geometry.orthonormal_basis.orthonormal_basis import OrthonormalBasis
from skellyforge.core.math.geometry.orthonormal_basis.reference_frame_definition import (
    ReferenceFrameDefinition,
)
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton_parts.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton_parts.naming import (
    raise_unless_aliases_are_valid,
    raise_unless_snake_case_segment_name,
)
from skellyforge.type_overloads import LandmarkNameString, RigidBodySegmentName

MINIMUM_LANDMARKS_FOR_A_FULL_FRAME: int = 3


@dataclass(frozen=True, slots=True, eq=False)
class RigidBodySegment:
    """One rigid body, whose local frame is fully determined by its landmarks.

    Attributes:
        name: snake_case segment name, optionally suffixed `.L` or `.R`.
        landmarks: this segment's landmarks, keyed by name. Needs at least three, and
            must contain every landmark the frame definition names. Typed as a `Mapping`
            because the segment stores it without copying - mutating it afterwards is not
            part of the contract.
        frame_definition: which landmarks lie on which signed axes. The segment's origin
            is `frame_definition.origin_point_name`, so there is no separate origin field
            to fall out of sync with it.
        aliases: other names this segment answers to.
    """

    name: RigidBodySegmentName
    landmarks: Mapping[LandmarkNameString, AnatomicalLandmark]
    frame_definition: ReferenceFrameDefinition
    aliases: tuple[RigidBodySegmentName, ...] = ()

    def __post_init__(self) -> None:
        raise_unless_snake_case_segment_name(name=self.name)
        raise_unless_aliases_are_valid(name=self.name, aliases=self.aliases)
        if len(self.landmarks) < MINIMUM_LANDMARKS_FOR_A_FULL_FRAME:
            raise ValueError(
                f"segment {self.name!r}: a fully specified segment needs at least "
                f"{MINIMUM_LANDMARKS_FOR_A_FULL_FRAME} landmarks - got "
                f"{len(self.landmarks)}. Use UnderspecifiedRigidBodySegment for two."
            )
        mismatched_keys = [
            key for key, landmark in self.landmarks.items() if key != landmark.name
        ]
        if mismatched_keys:
            raise ValueError(
                f"segment {self.name!r}: landmarks must be keyed by their own name - "
                f"mismatched keys {mismatched_keys}"
            )
        claimed_by: dict[LandmarkNameString, LandmarkNameString] = {}
        for landmark in self.landmarks.values():
            for known_name in landmark.all_names:
                colliding_landmark = claimed_by.get(known_name)
                if colliding_landmark is not None:
                    raise ValueError(
                        f"segment {self.name!r}: the name {known_name!r} is claimed by "
                        f"both landmark {colliding_landmark!r} and {landmark.name!r}"
                    )
                claimed_by[known_name] = landmark.name

        missing_names = [
            name for name in self.frame_definition_landmark_names if name not in self.landmarks
        ]
        if missing_names:
            raise ValueError(
                f"segment {self.name!r}: frame definition names landmarks {missing_names} "
                f"that this segment does not have - it has {sorted(self.landmarks)}"
            )

    @property
    def all_names(self) -> tuple[RigidBodySegmentName, ...]:
        """Every name this segment answers to, canonical name first."""
        return (self.name, *self.aliases)

    @property
    def frame_definition_landmark_names(
        self,
    ) -> tuple[LandmarkNameString, LandmarkNameString, LandmarkNameString]:
        """The origin, primary, and secondary landmark names, in that order."""
        return (
            self.frame_definition.origin_point_name,
            self.frame_definition.primary_point_name,
            self.frame_definition.secondary_point_name,
        )

    @property
    def length(self) -> float:
        """Origin-to-primary distance, from the landmarks' rest positions."""
        origin_name, primary_name, _ = self.frame_definition_landmark_names
        origin = self.landmarks[origin_name].local_position
        primary = self.landmarks[primary_name].local_position
        return float((primary - origin).norm())

    def __str__(self) -> str:
        origin_name, primary_name, secondary_name = self.frame_definition_landmark_names
        return (
            f"{self.name} ({len(self.landmarks)} landmarks, length {self.length:.4g}): "
            f"origin {origin_name}, {self.frame_definition.primary_axis} -> {primary_name}, "
            f"{self.frame_definition.secondary_axis} ~ {secondary_name}"
        )


def calculate_bases_for_segments(
    *,
    segments: Sequence[RigidBodySegment],
    points: Mapping[str, Point],
) -> dict[RigidBodySegmentName, OrthonormalBasis]:
    """Solve many segments' frames at once, grouping those that share an axis convention.

    `calculate_orthonormal_basis` costs about 100 microseconds per CALL and about 0.23
    microseconds per FRAME once batched, so its cost is almost entirely fixed overhead.
    Solving segments one at a time pays that overhead per segment; stacking every segment
    that shares a (primary axis, secondary axis, handedness) convention into one batched
    call pays it once per convention. A skeleton whose segments share one convention
    therefore solves in roughly the time a single segment used to take.

    The batch axis here is SEGMENTS rather than time, and it composes with time: each
    segment's points may themselves be a single frame, a rolling window, or a whole take,
    as long as every segment's points share a shape.

    Args:
        segments: the segments to solve. Names must be unique.
        points: observed world locations keyed by landmark name, covering every landmark
            the segments' frame definitions name.

    Returns:
        One basis per segment, keyed by segment name.
    """
    segment_names = [segment.name for segment in segments]
    if len(set(segment_names)) != len(segment_names):
        duplicates = sorted({name for name in segment_names if segment_names.count(name) > 1})
        raise ValueError(f"segment names must be unique - repeated: {duplicates}")

    segments_by_convention: dict[
        tuple[SpatialAxis, SpatialAxis, Handedness], list[RigidBodySegment]
    ] = {}
    for segment in segments:
        convention = (
            segment.frame_definition.primary_axis,
            segment.frame_definition.secondary_axis,
            segment.frame_definition.handedness,
        )
        segments_by_convention.setdefault(convention, []).append(segment)

    bases_by_segment_name: dict[RigidBodySegmentName, OrthonormalBasis] = {}
    for (primary_axis, secondary_axis, handedness), grouped_segments in (
        segments_by_convention.items()
    ):
        stacked_points = {
            role: Point.from_prevalidated_array(
                array=np.stack(
                    arrays=[
                        points[segment.frame_definition_landmark_names[position]].array
                        for segment in grouped_segments
                    ],
                    axis=0,
                )
            )
            for position, role in enumerate(("origin", "primary", "secondary"))
        }
        stacked_definition = ReferenceFrameDefinition(
            origin_point_name="origin",
            primary_axis=primary_axis,
            primary_point_name="primary",
            secondary_axis=secondary_axis,
            secondary_point_name="secondary",
            handedness=handedness,
        )
        batched_basis = calculate_orthonormal_basis(
            points=stacked_points, definition=stacked_definition
        )
        for index, segment in enumerate(grouped_segments):
            bases_by_segment_name[segment.name] = batched_basis.at_batch_index(index=index)

    return bases_by_segment_name
