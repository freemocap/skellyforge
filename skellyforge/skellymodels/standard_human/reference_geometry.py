"""Reference geometry: the T-pose each segment's live pose is measured against.

Built from the composed segments + per-subject measured lengths. One build
serves both the orientation solver (identity == T-pose) and the stream
schema's rest pose. Right-side segments mirror by negating Y and REBUILDING
frames right-handed (SF-AL A3) — a basis is never reflected.

``rest_rotation`` is extrinsic XYZ euler, radians: R = Rx·Ry·Rz. Each segment's
rest frame derives the basis vector NAMED by its EXACT axis as
``R · unit(axis_name)`` — so a body segment (exact axis on y) rests with
``R · ŷ`` toward its child bone, and a face bone (exact axis on z) rests with
``R · ẑ`` as its gaze direction. Every authored value is single-axis, so the
euler convention matters only here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from skellyforge.kinematics.coordinate_frame_ops import (
    _AXIS_TO_INDEX,
    assemble_named_basis,
)
from skellyforge.skellymodels.standard_human.segment_definition import (
    AxisDefinition,
    ParentAttachment,
    SegmentDefinition,
)

_COLLINEARITY_DOT = 0.9998  # cos(1°) — stiffer than the solver's ~5° gate

# Rest approximate DIRECTIONS (world vectors at the T-pose) for segments whose
# approximate axis references a point with no schematic rest position (or one
# coinciding with the origin): the nose (head), the heel (foot), small_toe
# (toes), and the hip joints (hips, whose lateral pair coincides with the origin
# — no widths are declared). Authored for the LEFT side; mirroring flips Y for
# the right.
_TWIST_OVERRIDES: dict[str, NDArray[np.float64]] = {
    "hips": np.array([1.0, 0.0, 0.0]),   # the lateral pair coincides with the origin
    "head": np.array([1.0, 0.0, 0.0]),   # nose — anterior (+X), the gaze direction
    "foot": np.array([0.0, 0.0, -1.0]),  # heel — down-back → −Z after Gram-Schmidt
    "toes": np.array([0.0, 1.0, 0.0]),   # small toe — lateral (left side, +Y)
}


def _unprefixed(name: str) -> str:
    for prefix in ("left_", "right_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _mirror(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """Negate Y — the standard mirror across the sagittal (XZ) plane."""
    return np.array([vec[0], -vec[1], vec[2]], dtype=np.float64)


def _rest_axis_direction(
    rest_rotation: tuple[float, float, float], axis_name: str
) -> NDArray[np.float64]:
    """The rest direction for a NAMED basis vector: ``R · unit(axis_name)``.

    The segment's rest frame's named vector equals this direction — a body
    segment's ``ŷ`` points toward its child bone, a face bone's ``ẑ`` is its
    gaze.
    """
    rx, ry, rz = rest_rotation
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    unit = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0])}[axis_name]
    return rx_m @ ry_m @ rz_m @ unit


@dataclass(frozen=True)
class SegmentReferenceGeometry:
    """One segment's T-pose geometry: where it sits, how it's oriented, how long."""

    origin: NDArray[np.float64]  # (3,) standard mm — the transform origin
    basis: NDArray[np.float64]   # (3,3) rows [x̂, ŷ, ẑ] of the rest frame
    length: float                # standard mm


@dataclass(frozen=True)
class ReferenceGeometry:
    """The whole human's T-pose: per-segment geometry + rest keypoint positions."""

    segments: dict[str, SegmentReferenceGeometry]
    keypoints: dict[str, NDArray[np.float64]]


def build_reference_geometry(
    segments: list[SegmentDefinition],
    measured_lengths: dict[str, float],
) -> ReferenceGeometry:
    """Build the T-pose reference from composed segments + measured lengths.

    Origins accumulate through the tree in authoring order (the root at the
    origin — the reference pose is a schematic identity frame of orientations ×
    lengths; ORIGIN attachments place the child at the parent's origin, so
    e.g. the rest hip joints coincide with hips_center: no widths are declared).
    """
    by_name = {s.name: s for s in segments}
    origins: dict[str, NDArray[np.float64]] = {}
    rest_dirs: dict[str, NDArray[np.float64]] = {}
    keypoints: dict[str, NDArray[np.float64]] = {}

    # pass 1: rest directions of the EXACT axis (right side mirrored)
    for segment in segments:
        exact = segment.exact_axis
        direction = _rest_axis_direction(segment.rest_rotation, exact.axis)
        if segment.name.startswith("right_"):
            direction = _mirror(direction)
        rest_dirs[segment.name] = direction

    missing = [s.name for s in segments if s.name not in measured_lengths]
    if missing:
        raise ValueError(
            "measured_lengths is missing segments: "
            + ", ".join(sorted(missing))
        )

    # pass 2: origins + rest keypoint positions
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
            # Name agreement: if this segment's origin keypoint was already
            # positioned by an earlier declaration (e.g. the middle finger's
            # mcp, which is the hand's exact-axis endpoint), that position is
            # authoritative. Otherwise the branch point is the parent's origin
            # (e.g. a hip joint, the spine) — the reference pose is schematic;
            # no wrist→mcp fan geometry is declared.
            origin = keypoints.get(segment.origin_keypoint, origins[segment.parent])
        origins[segment.name] = origin
        keypoints[segment.origin_keypoint] = origin.copy()
        keypoints[segment.exact_axis.target_keypoint] = (
            origin + rest_dirs[segment.name] * length
        )

    # pass 3: bases (approximate axes need the completed keypoint map)
    geometries: dict[str, SegmentReferenceGeometry] = {}
    for segment in segments:
        basis = _rest_basis(segment, origins, rest_dirs, keypoints)
        geometries[segment.name] = SegmentReferenceGeometry(
            origin=origins[segment.name],
            basis=basis,
            length=measured_lengths[segment.name],
        )

    # ``nose`` is an off-chain keypoint: several driven segments (head, neck,
    # and the three face bones) name it as their exact axis, but it has no
    # single standard rest position — the three face bones point different
    # ways from the head origin and cannot share one schematic point. The
    # tracker (or a live-pose fixture) supplies it per frame; the reference
    # pose leaves it out so the face bones solve only when it is present.
    keypoints.pop("nose", None)

    return ReferenceGeometry(segments=geometries, keypoints=keypoints)


def _rest_basis(
    segment: SegmentDefinition,
    origins: dict[str, NDArray[np.float64]],
    rest_dirs: dict[str, NDArray[np.float64]],
    keypoints: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """The segment's rest orthonormal frame (rows [x̂, ŷ, ẑ]).

    The EXACT axis's named row is the exact rest direction (``rest_dirs``); the
    APPROXIMATE axis's named row is the approximate rest direction
    (Gram-Schmidt'd); the remaining row is the right-handed cross product. The
    named-row placement means each rest basis vector aligns with its authored
    axis name.
    """
    exact = segment.exact_axis
    approx = segment.approximate_axis
    exact_idx = _AXIS_TO_INDEX[exact.axis]

    exact_dir = rest_dirs[segment.name]

    if approx is None:
        # twist-less segment: a deterministic orthogonal placeholder for the
        # second vector, carried by the damped minimal-roll tier. It occupies
        # the next basis row after the exact axis (cyclic), matching what the
        # solver's damped tier carries.
        second_dir = _default_perpendicular(exact_dir)
        second_idx = (exact_idx + 1) % 3  # next row in cyclic basis order
        return assemble_named_basis(
            {exact_idx: exact_dir, second_idx: second_dir}
        )

    approx_idx = _AXIS_TO_INDEX[approx.axis]
    approx_dir = _rest_approximate_direction(
        segment, approx, origins, rest_dirs, keypoints
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


def _rest_approximate_direction(
    segment: SegmentDefinition,
    approx: "AxisDefinition",
    origins: dict[str, NDArray[np.float64]],
    rest_dirs: dict[str, NDArray[np.float64]],
    keypoints: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """The approximate axis at rest, as the rest frame's approximate (named) vector.

    The authored override table is AUTHORITATIVE where one exists (see module
    docstring); otherwise the APPROXIMATE axis declaration's ``target_keypoint``
    rest position (``origin → target_keypoint``); else the default perpendicular.
    Fails loudly if the result is still collinear with the exact axis.
    """
    exact_dir = rest_dirs[segment.name]
    candidate: NDArray[np.float64] | None = None

    override = _TWIST_OVERRIDES.get(_unprefixed(segment.name))
    if override is not None:
        candidate = override.copy()
        if segment.name.startswith("right_"):
            candidate = _mirror(candidate)

    if candidate is None:
        if approx.target_keypoint in keypoints:
            vec = keypoints[approx.target_keypoint] - origins[segment.name]
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
    """A deterministic unit direction orthogonal to *exact_dir*.

    Orthogonalizes a standard reference (``+X`` unless the exact direction is
    near ``+X``, then ``+Y``) against *exact_dir*, so the result is never
    collinear with it.
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
