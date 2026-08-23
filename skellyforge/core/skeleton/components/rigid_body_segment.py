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

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import (
    MINIMUM_RELATIVE_SINGULAR_VALUE,
    MINIMUM_VECTOR_NORM,
)
from skellyforge.core.math.geometry.orthonormal_basis.calculate_orthonormal_basis import (
    calculate_orthonormal_basis,
    direction_along,
)
from skellyforge.core.math.geometry.orthonormal_basis.orthonormal_basis import OrthonormalBasis
from skellyforge.core.math.geometry.orthonormal_basis.reference_frame_definition import (
    ReferenceFrameDefinition,
)
from skellyforge.core.math.geometry.spatial_vectors import Point, UnitVector
from skellyforge.core.skeleton.components.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.math.kinematics.rigid_point_set import (
    MINIMUM_POINTS_FOR_RIGID_FIT,
)
from skellyforge.core.skeleton.components.naming import (
    raise_unless_aliases_are_valid,
    raise_unless_snake_case_segment_name,
)
from skellyforge.type_overloads import LandmarkNameString, RigidBodySegmentName


@dataclass(frozen=True, slots=True, eq=False)
class RigidBodySegment:
    """One rigid body: its landmarks and the frame definition they realize.

    Attributes:
        name: snake_case segment name. Sidedness is a `left_`/`right_` PREFIX that
            the YAML loader adds; there is no `.L`/`.R` suffix convention here.
        landmarks: this segment's landmarks, keyed by name. Must contain the definition's
            primary landmark; the origin may instead be a shared landmark owned by the
            parent segment (the joint a linkage is built on). Whether it also contains
            the secondary one is what makes the segment fully specified. Typed as a
            `Mapping` because the
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

        if self.frame_definition.primary_point_name not in self.landmarks:
            raise ValueError(
                f"segment {self.name!r}: frame definition names its primary landmark "
                f"{self.frame_definition.primary_point_name!r}, which this segment does "
                f"not have - it has {sorted(self.landmarks)}"
            )

        # A segment's origin IS the zero of its own frame, by definition. When the origin
        # landmark belongs to this segment rather than to its parent, that identity is a
        # claim about the authored coordinates, and everything downstream relies on it:
        # `length` measures the primary's magnitude, and hydration rotates local positions
        # about a local origin it assumes is at zero. An authoring slip here would produce
        # silently wrong lengths and orientations, so it is checked rather than trusted.
        own_origin = self.landmarks.get(self.frame_definition.origin_point_name)
        if own_origin is not None:
            offset = float(np.linalg.norm(own_origin.local_position.array))
            if offset > MINIMUM_VECTOR_NORM:
                raise ValueError(
                    f"segment {self.name!r}: its own origin landmark "
                    f"{own_origin.name!r} must sit at [0, 0, 0] in this segment's frame, "
                    f"because that is what being the origin means - got "
                    f"{own_origin.local_position.array.tolist()} "
                    f"({offset:.4g} away). Either move it to the origin, or make the "
                    "origin a landmark owned by this segment's parent."
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
    def supports_rigid_fit(self) -> bool:
        """Whether this segment's own landmarks span a plane rather than just a line.

        This is a STATIC property of the authored geometry, and it is what decides which
        closed form hydrates the segment. A segment whose landmarks are collinear - a
        two-landmark limb, or a straight spine link - can never yield a full orientation
        from a rigid fit, however many frames are observed, because the roll about the
        line they share is not in the data. Deciding the branch here, once, keeps the
        decision out of a per-frame `try`/`except` and stops a genuine runtime failure
        from being mistaken for a collinear segment.
        """
        if len(self.landmarks) < MINIMUM_POINTS_FOR_RIGID_FIT:
            return False
        local_positions = np.stack(
            arrays=[landmark.local_position.array for landmark in self.landmarks.values()],
            axis=0,
        )
        centered = local_positions - local_positions.mean(axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        if singular_values[0] <= 0.0:
            return False
        return bool(
            singular_values[1] > MINIMUM_RELATIVE_SINGULAR_VALUE * singular_values[0]
        )

    @property
    def landmark_names(self) -> tuple[LandmarkNameString, ...]:
        """The landmark names this segment needs observed positions for.

        Origin and primary always; the secondary one too once it is available, since that
        is exactly when it starts being used.
        """
        names = [
            self.frame_definition.origin_point_name,
            self.frame_definition.primary_point_name,
        ]
        secondary_point_name = self.frame_definition.secondary_point_name
        if secondary_point_name is not None and secondary_point_name in self.landmarks:
            names.append(secondary_point_name)
        return tuple(names)

    @property
    def length(self) -> float:
        """Origin-to-primary distance, from the landmarks' rest positions.

        The origin sits at `[0, 0, 0]` in this segment's frame whichever segment owns the
        landmark - enforced above when this segment owns it, and true by construction when
        the parent does, since a shared joint's rest position lives in the parent's frame -
        so the length is the primary's magnitude, with no special case either way.
        """
        primary = self.landmarks[self.frame_definition.primary_point_name].local_position
        return float(np.linalg.norm(primary.array))

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
