"""Reference geometry: the T-pose each segment's live pose is measured against.

Built from the composed segments + per-subject measured lengths. One build
serves both the orientation solver (identity == T-pose) and the stream
schema's rest pose. Right-side segments mirror by negating Y and REBUILDING
frames right-handed (SF-AL A3) — a basis is never reflected.

Each axis carries an optional rest_direction (a world-space unit vector at
the T-pose): the exact axis's direction is the toward-child (primary)
direction; the approximate axis's direction is its twist reference. A
rest_direction of None defaults to the axis row's positive unit vector
(the identity rest orientation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from skellyforge.kinematics.coordinate_frame_ops import (
    assemble_named_basis,
    axis_index_and_sign,
)
from skellyforge.skellymodels.standard_human.segment_definition import (
    AxisDefinition,
    ParentAttachment,
    SegmentDefinition,
)

_COLLINEARITY_DOT = 0.9998  # cos(1°) — stiffer than the solver's ~5° gate


def _mirror(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """Negate Y — the standard mirror across the sagittal (XZ) plane."""
    return np.array([vec[0], -vec[1], vec[2]], dtype=np.float64)


def _axis_unit_vector(axis: str) -> NDArray[np.float64]:
    """The POSITIVE unit vector of a signed axis's row (e.g. "y"/"-y" -> +Y)."""
    idx, _ = axis_index_and_sign(axis)
    return np.eye(3, dtype=np.float64)[idx]


def _axis_rest_direction(axis: AxisDefinition) -> NDArray[np.float64]:
    """The axis's authored rest direction, or its row's positive unit vector."""
    if axis.rest_direction is not None:
        return np.asarray(axis.rest_direction, dtype=np.float64)
    return _axis_unit_vector(axis.axis)


@dataclass(frozen=True)
class SegmentReferenceGeometry:
    """One segment's T-pose geometry: where it sits, how it's oriented, how long."""

    origin: NDArray[np.float64]  # (3,) standard mm — the transform origin
    basis: NDArray[np.float64]   # (3,3) rows [x̂, ŷ, ẑ] of the rest frame
    length: float                # standard mm


@dataclass(frozen=True)
class ReferenceGeometry:
    """The whole human's T-pose: per-segment geometry + rest landmark positions."""

    segments: dict[str, SegmentReferenceGeometry]
    landmarks: dict[str, NDArray[np.float64]]

    @classmethod
    def from_segments(
        cls,
        segments: list[SegmentDefinition],
        measured_lengths: dict[str, float],
    ) -> "ReferenceGeometry":
        """Build the T-pose reference from composed segments + measured lengths.

        Origins accumulate through the tree in authoring order (the root at the
        origin — the reference pose is a schematic identity frame of orientations ×
        lengths; ORIGIN attachments place the child at the parent's origin, so
        e.g. the rest hip joints coincide with hips_center: no widths are declared).
        """
        origins: dict[str, NDArray[np.float64]] = {}
        rest_dirs: dict[str, NDArray[np.float64]] = {}
        landmarks: dict[str, NDArray[np.float64]] = {}

        # pass 1: rest directions of the EXACT axis (right side mirrored)
        for segment in segments:
            direction = _axis_rest_direction(segment.exact_axis)
            if segment.name.startswith("right_"):
                direction = _mirror(direction)
            rest_dirs[segment.name] = direction

        missing = [s.name for s in segments if s.name not in measured_lengths]
        if missing:
            raise ValueError(
                "measured_lengths is missing segments: "
                + ", ".join(sorted(missing))
            )

        # pass 2: origins + rest landmark positions
        for segment in segments:
            length = measured_lengths[segment.name]
            if segment.parent is None:
                origin = np.zeros(3, dtype=np.float64)
            elif segment.parent_attachment is ParentAttachment.DISTAL:
                origin = (
                    origins[segment.parent]
                    + rest_dirs[segment.parent] * measured_lengths[segment.parent]
                )
            else:  # ORIGIN — branch from the parent's origin
                # Name agreement: if this segment's origin landmark was already
                # positioned by an earlier declaration (e.g. the middle finger's
                # mcp, which is the hand's exact-axis endpoint), that position is
                # authoritative. Otherwise the branch point is the parent's origin.
                origin = landmarks.get(segment.origin_landmark, origins[segment.parent])
            origins[segment.name] = origin
            landmarks[segment.origin_landmark] = origin.copy()
            landmarks[segment.exact_axis.target_landmark] = (
                origin + rest_dirs[segment.name] * length
            )

        # pass 3: bases (approximate axes need the completed landmark map)
        geometries: dict[str, SegmentReferenceGeometry] = {}
        for segment in segments:
            basis = cls._rest_basis(segment, origins, rest_dirs, landmarks)
            geometries[segment.name] = SegmentReferenceGeometry(
                origin=origins[segment.name],
                basis=basis,
                length=measured_lengths[segment.name],
            )

        # nose is an off-chain landmark: several driven segments (head, neck,
        # and the three face bones) name it as their exact axis, but it has no
        # single standard rest position — the three face bones point different
        # ways from the head origin and cannot share one schematic point. The
        # tracker (or a live-pose fixture) supplies it per frame; the reference
        # pose leaves it out so the face bones solve only when it is present.
        landmarks.pop("nose", None)

        return cls(segments=geometries, landmarks=landmarks)

    @staticmethod
    def _rest_basis(
        segment: SegmentDefinition,
        origins: dict[str, NDArray[np.float64]],
        rest_dirs: dict[str, NDArray[np.float64]],
        landmarks: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """The segment's rest orthonormal frame (rows [x̂, ŷ, ẑ]).

        The EXACT axis's named row is the exact rest direction (signed by the
        axis name); the APPROXIMATE axis's named row is the approximate rest
        direction (Gram-Schmidt'd, signed); the remaining row is the
        right-handed cross product.
        """
        exact = segment.exact_axis
        approx = segment.approximate_axis
        exact_idx, exact_sign = axis_index_and_sign(exact.axis)

        exact_dir = exact_sign * rest_dirs[segment.name]

        if approx is None:
            # twist-less segment: a deterministic orthogonal placeholder for the
            # second vector, carried by the damped minimal-roll tier.
            second_dir = _default_perpendicular(exact_dir)
            second_idx = (exact_idx + 1) % 3
            return assemble_named_basis(
                {exact_idx: exact_dir, second_idx: second_dir}
            )

        approx_idx, approx_sign = axis_index_and_sign(approx.axis)
        approx_dir = approx_sign * ReferenceGeometry._rest_approximate_direction(
            segment, approx, origins, rest_dirs, landmarks
        )

        # Gram-Schmidt the approximate direction against the exact direction
        approx_orth = approx_dir - float(np.dot(approx_dir, exact_dir)) * exact_dir
        approx_norm = float(np.linalg.norm(approx_orth))
        if approx_norm < 1e-10:
            raise ValueError(
                f"segment {segment.name!r}: rest approximate axis is collinear with "
                f"its exact axis — author an override"
            )
        approx_orth = approx_orth / approx_norm

        return assemble_named_basis(
            {exact_idx: exact_dir, approx_idx: approx_orth}
        )

    @staticmethod
    def _rest_approximate_direction(
        segment: SegmentDefinition,
        approx: AxisDefinition,
        origins: dict[str, NDArray[np.float64]],
        rest_dirs: dict[str, NDArray[np.float64]],
        landmarks: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """The approximate axis's RAW rest direction (unsigned): the authored
        rest_direction if present, else the target landmark's rest position,
        else the default perpendicular. Fails loudly if collinear with the
        exact axis."""
        exact_dir = rest_dirs[segment.name]
        candidate: NDArray[np.float64] | None = None

        if approx.rest_direction is not None:
            candidate = np.asarray(approx.rest_direction, dtype=np.float64)
            if segment.name.startswith("right_"):
                candidate = _mirror(candidate)

        if candidate is None and approx.target_landmark in landmarks:
            vec = landmarks[approx.target_landmark] - origins[segment.name]
            norm = float(np.linalg.norm(vec))
            if norm > 1e-10:
                vec = vec / norm
                if abs(float(np.dot(exact_dir, vec))) <= _COLLINEARITY_DOT:
                    candidate = vec

        if candidate is None:
            candidate = _default_perpendicular(exact_dir)

        if abs(float(np.dot(exact_dir, candidate))) > _COLLINEARITY_DOT:
            raise ValueError(
                f"segment {segment.name!r}: rest approximate axis is collinear with "
                f"its exact axis — author an override"
            )
        return candidate


def _default_perpendicular(exact_dir: NDArray[np.float64]) -> NDArray[np.float64]:
    """A deterministic unit direction orthogonal to exact_dir.

    Orthogonalizes a standard reference (+X unless the exact direction is
    near +X, then +Y) against exact_dir, so the result is never collinear
    with it.
    """
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(exact_dir, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    perp = ref - float(np.dot(ref, exact_dir)) * exact_dir
    norm = float(np.linalg.norm(perp))
    if norm < 1e-10:
        perp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        norm = 1.0
    return perp / norm
