"""A rigid-body segment: landmarks plus the reference frame definition they realize.

A segment is FULLY SPECIFIED when its frame definition names a secondary axis and point
AND that point is one of its landmarks - three non-collinear points being exactly what a
Gram-Schmidt frame needs. It is UNDERSPECIFIED otherwise: two points fix the direction
the bone runs in, but roll about that direction is unconstrained, so the segment yields a
direction and refuses to invent a triad.

There is one class for both because there is one thing here. A segment gains its third
landmark, or its definition gains its secondary axis, and the same object starts
answering `calculate_basis` - which is how bones actually behave while a skeleton is
being calibrated.

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
    direction_along,
)
from skellyforge.core.math.geometry.orthonormal_basis.handedness import Handedness
from skellyforge.core.math.geometry.orthonormal_basis.orthonormal_basis import OrthonormalBasis
from skellyforge.core.math.geometry.orthonormal_basis.reference_frame_definition import (
    ReferenceFrameDefinition,
)
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.math.geometry.spatial_vectors import Point, UnitVector
from skellyforge.core.skeleton_parts.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton_parts.naming import (
    raise_unless_aliases_are_valid,
    raise_unless_snake_case_segment_name,
)
from skellyforge.type_overloads import LandmarkNameString, RigidBodySegmentName


@dataclass(frozen=True, slots=True, eq=False)
class RigidBodySegment:
    """One rigid body: its landmarks and the frame definition they realize.

    Attributes:
        name: snake_case segment name, optionally suffixed `.L` or `.R`.
        landmarks: this segment's landmarks, keyed by name. Must contain the definition's
            origin and primary landmarks; whether it also contains the secondary one is
            what makes the segment fully specified. Typed as a `Mapping` because the
            segment stores it without copying - mutating it afterwards is not part of the
            contract.
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

        missing_required = [
            name
            for name in (
                self.frame_definition.origin_point_name,
                self.frame_definition.primary_point_name,
            )
            if name not in self.landmarks
        ]
        if missing_required:
            raise ValueError(
                f"segment {self.name!r}: frame definition names landmarks "
                f"{missing_required} that this segment does not have - it has "
                f"{sorted(self.landmarks)}"
            )

    @property
    def all_names(self) -> tuple[RigidBodySegmentName, ...]:
        """Every name this segment answers to, canonical name first."""
        return (self.name, *self.aliases)

    @property
    def is_fully_specified(self) -> bool:
        """Whether this segment can produce a full orthonormal triad.

        True when the frame definition names a secondary axis and point AND that landmark
        is one of this segment's. Either half being absent leaves roll unconstrained.
        """
        secondary_point_name = self.frame_definition.secondary_point_name
        return secondary_point_name is not None and secondary_point_name in self.landmarks

    @property
    def landmark_names(self) -> tuple[LandmarkNameString, ...]:
        """The landmark names this segment needs observed positions for.

        Origin and primary always; the secondary one too once it is available, since that
        is exactly when it starts being used.
        """
        return tuple(
            name for name in self.frame_definition.point_names if name in self.landmarks
        )

    @property
    def length(self) -> float:
        """Origin-to-primary distance, from the landmarks' rest positions."""
        origin = self.landmarks[self.frame_definition.origin_point_name].local_position
        primary = self.landmarks[self.frame_definition.primary_point_name].local_position
        return float((primary - origin).norm())

    def calculate_direction(self, *, points: Mapping[str, Point]) -> UnitVector:
        """The observed origin-to-primary direction, signed to match the primary axis.

        Available whether or not the segment is fully specified, because two points are
        all a direction needs. Vectorized over the leading dimensions of `points`, so one
        frame, a rolling window, and a whole take all go through this unchanged.

        Args:
            points: observed world locations keyed by landmark name.

        Returns:
            The unit direction the segment currently points in.
        """
        origin_name = self.frame_definition.origin_point_name
        primary_name = self.frame_definition.primary_point_name
        return direction_along(
            axis=self.frame_definition.primary_axis,
            displacement=points[primary_name] - points[origin_name],
            description=(
                f"segment {self.name!r}'s primary axis (`{origin_name}` -> "
                f"`{primary_name}`)"
            ),
        )

    def calculate_basis(self, *, points: Mapping[str, Point]) -> OrthonormalBasis:
        """The observed orthonormal frame of this segment.

        Args:
            points: observed world locations keyed by landmark name.

        Returns:
            The segment's `OrthonormalBasis` in the world frame.

        Raises:
            ValueError: the segment is underspecified, so there is no triad to build.
        """
        self.raise_unless_fully_specified()
        return calculate_orthonormal_basis(points=points, definition=self.frame_definition)

    def raise_unless_fully_specified(self) -> None:
        """Raise unless this segment has everything a full triad needs."""
        if self.is_fully_specified:
            return
        secondary_point_name = self.frame_definition.secondary_point_name
        reason = (
            "its frame definition names no secondary axis"
            if secondary_point_name is None
            else f"its secondary landmark {secondary_point_name!r} is not one of its "
            f"landmarks {sorted(self.landmarks)}"
        )
        raise ValueError(
            f"segment {self.name!r} is underspecified - {reason}. Two points fix the "
            f"{self.frame_definition.primary_axis.name} direction but leave roll about "
            "it free; use `calculate_direction` and resolve roll separately."
        )

    def __str__(self) -> str:
        specification = "fully specified" if self.is_fully_specified else "underspecified"
        return (
            f"{self.name} ({len(self.landmarks)} landmarks, {specification}, "
            f"length {self.length:.4g}): {self.frame_definition}"
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
        segments: the segments to solve. Names must be unique and every segment must be
            fully specified.
        points: observed world locations keyed by landmark name, covering every landmark
            the segments' frame definitions name.

    Returns:
        One basis per segment, keyed by segment name.

    Raises:
        ValueError: segment names repeat, or any segment is underspecified.
    """
    segment_names = [segment.name for segment in segments]
    if len(set(segment_names)) != len(segment_names):
        duplicates = sorted({name for name in segment_names if segment_names.count(name) > 1})
        raise ValueError(f"segment names must be unique - repeated: {duplicates}")

    # One pass: validate, read each definition's point names once, and group by convention.
    # `point_names` and the fully-specified check both cost more than the array indexing
    # they feed, so neither belongs inside the per-role stacking comprehension below.
    grouped_by_convention: dict[
        tuple[SpatialAxis, SpatialAxis, Handedness],
        tuple[list[RigidBodySegment], list[tuple[str, ...]]],
    ] = {}
    for segment in segments:
        segment.raise_unless_fully_specified()
        definition = segment.frame_definition
        convention = (
            definition.primary_axis,
            definition.secondary_axis,
            definition.handedness,
        )
        grouped_segments, grouped_point_names = grouped_by_convention.setdefault(
            convention, ([], [])
        )
        grouped_segments.append(segment)
        grouped_point_names.append(definition.point_names)

    bases_by_segment_name: dict[RigidBodySegmentName, OrthonormalBasis] = {}
    for (primary_axis, secondary_axis, handedness), (
        grouped_segments,
        grouped_point_names,
    ) in grouped_by_convention.items():
        stacked_points = {
            role: Point.from_prevalidated_array(
                array=np.stack(
                    arrays=[
                        points[segment_point_names[position]].array
                        for segment_point_names in grouped_point_names
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
