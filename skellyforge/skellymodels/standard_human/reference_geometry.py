"""Reference geometry: the T-pose each segment's live pose is measured against.

Built from the composed segments + per-subject measured lengths. One build
serves both the orientation solver (identity == T-pose) and the stream
schema's rest pose. Right-side segments mirror by negating Y and REBUILDING
frames right-handed (SF-AL A3) — a basis is never reflected.

``rest_rotation`` is extrinsic XYZ euler, radians: R = Rx·Ry·Rz applied to the
segment's rest +Z axis. Every authored value is single-axis, so this is the
only place the convention matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from skellyforge.kinematics.coordinate_frame_ops import build_orthonormal_basis
from skellyforge.skellymodels.standard_human.segment_definition import (
    AxisDefinition,
    AxisKind,
    SegmentDefinition,
)

_COLLINEARITY_DOT = 0.9998  # cos(1°) — stiffer than the solver's ~5° gate

# Rest approximate axes for segments whose approximate axis references a point
# of its own rigid set that has no rest position (or coincides with the origin)
# at the T-pose: the nose (head), the heel (foot), small_toe (toes), and the
# hip joints (hips, whose lateral pair coincide with the origin — no widths are
# declared). An authored direction supplies the rest value. Authored for the
# LEFT side; mirroring flips Y for the right.
_TWIST_OVERRIDES: dict[str, NDArray[np.float64]] = {
    "hips": np.array([1.0, 0.0, 0.0]),        # right_hip coincides with the origin
    "head": np.array([1.0, 0.0, 0.0]),        # nose — anterior
    "foot": np.array([0.0, 0.0, 1.0]),        # up — the foot points +X, so its
                                             # roll/pitch reference is vertical
    "toes": np.array([0.0, 1.0, 0.0]),        # small toe — lateral (left side)
}


def _exact_axis(segment: SegmentDefinition) -> AxisDefinition:
    """The segment's exact axis declaration (the defining direction)."""
    return next(a for a in segment.axes if a.kind is AxisKind.EXACT)


def _approximate_axis(segment: SegmentDefinition) -> AxisDefinition | None:
    """The segment's approximate axis declaration, or ``None`` if twist-less."""
    return next(
        (a for a in segment.axes if a.kind is AxisKind.APPROXIMATE), None
    )


def _unprefixed(name: str) -> str:
    for prefix in ("left_", "right_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _mirror(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """Negate Y — the canonical mirror across the sagittal (XZ) plane."""
    return np.array([vec[0], -vec[1], vec[2]], dtype=np.float64)


def _rest_direction(rest_rotation: tuple[float, float, float]) -> NDArray[np.float64]:
    """The segment's rest long axis: extrinsic XYZ euler applied to +Z."""
    rx, ry, rz = rest_rotation
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rx_m @ ry_m @ rz_m @ np.array([0.0, 0.0, 1.0])


def _build_basis(
    long_axis: NDArray[np.float64], approximate: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Rows [long, approximate orthonormalized, third] — right-handed.

    The Gram-Schmidt + cross-product construction is shared with the live
    solver via ``build_orthonormal_basis`` — only the rest-specific direction
    resolution (``_rest_approximate_axis``) differs between the two.
    """
    return build_orthonormal_basis(long_axis, approximate)


@dataclass(frozen=True)
class SegmentReferenceGeometry:
    """One segment's T-pose geometry: where it sits, how it's oriented, how long."""

    origin: NDArray[np.float64]  # (3,) canonical mm — the transform origin
    basis: NDArray[np.float64]   # (3,3) rows: [long axis, approximate axis, third]
    length: float                # canonical mm


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

    # pass 1: rest directions (right side mirrored)
    for segment in segments:
        direction = _rest_direction(segment.rest_rotation)
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
        elif segment.parent_attachment.value == "distal":
            origin = (
                origins[segment.parent]
                + rest_dirs[segment.parent] * measured_lengths[segment.parent]
            )
        else:  # ORIGIN — branch from the parent's origin
            # Name agreement: if this segment's origin keypoint was already
            # positioned by an earlier declaration (e.g. the middle finger's
            # mcp, which is the hand's long-axis endpoint), that position is
            # authoritative. Otherwise the branch point is the parent's origin
            # (e.g. a hip joint, the spine) — the reference pose is schematic;
            # no wrist→mcp fan geometry is declared.
            origin = keypoints.get(segment.origin_keypoint, origins[segment.parent])
        origins[segment.name] = origin
        keypoints[segment.origin_keypoint] = origin.copy()
        keypoints[_exact_axis(segment).target_keypoint] = (
            origin + rest_dirs[segment.name] * length
        )

    # pass 3: bases (approximate axes need the completed keypoint map)
    geometries: dict[str, SegmentReferenceGeometry] = {}
    for segment in segments:
        approximate = _rest_approximate_axis(
            segment, origins, rest_dirs, keypoints
        )
        basis = _build_basis(rest_dirs[segment.name], approximate)
        geometries[segment.name] = SegmentReferenceGeometry(
            origin=origins[segment.name],
            basis=basis,
            length=measured_lengths[segment.name],
        )

    # ``nose`` is an off-chain keypoint: several driven segments (head, neck,
    # and the three face bones) name it as their long axis, but it has no
    # single canonical rest position — the three face bones point different
    # ways from the head origin and cannot share one schematic point. The
    # tracker (or a live-pose fixture) supplies it per frame; the reference
    # pose leaves it out so the face bones solve only when it is present.
    keypoints.pop("nose", None)

    return ReferenceGeometry(segments=geometries, keypoints=keypoints)


def _rest_approximate_axis(
    segment: SegmentDefinition,
    origins: dict[str, NDArray[np.float64]],
    rest_dirs: dict[str, NDArray[np.float64]],
    keypoints: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """The approximate axis at rest: the authored override table where one
    exists (AUTHORITATIVE — see below), else the APPROXIMATE axis declaration's
    ``target_keypoint`` rest position (``origin → target_keypoint``) where it
    exists and is off-axis, else the default perpendicular. Fail loudly if the
    result is still collinear.
    """
    candidate: NDArray[np.float64] | None = None
    # An authored override is the rest twist wherever the schematic geometry
    # can't be trusted, and it takes PRECEDENCE over the target-position branch.
    # Overrides exist precisely for approximate targets that are off-chain or
    # degenerate at the T-pose: `hips`' lateral pair coincides with the origin,
    # and `foot`/`toes`' twist keypoints (heel/small_toe) have no schematic rest
    # position — their target branch already yields nothing. `head` is the
    # dangerous case: its target `nose` is off-chain, YET written (last-writer-
    # wins) by the five other segments that share it as their exact-axis target,
    # so it holds a non-degenerate but meaningless position that would otherwise
    # pre-empt the authored anterior override and roll the head's frame off
    # forward. Override-first keeps `head` anterior; `hips`/`foot`/`toes` are
    # unchanged (their target branch never resolved a value anyway).
    override = _TWIST_OVERRIDES.get(_unprefixed(segment.name))
    if override is not None:
        candidate = override.copy()
        if segment.name.startswith("right_"):
            candidate = _mirror(candidate)

    # No override: derive the rest twist from the approximate target's rest
    # position (origin → target_keypoint) when it exists and is off-axis.
    if candidate is None:
        approx = _approximate_axis(segment)
        if approx is not None and approx.target_keypoint in keypoints:
            vec = keypoints[approx.target_keypoint] - origins[segment.name]
            norm = float(np.linalg.norm(vec))
            if norm > 1e-10:
                vec = vec / norm
                if abs(float(np.dot(rest_dirs[segment.name], vec))) <= _COLLINEARITY_DOT:
                    candidate = vec

    if candidate is None:
        # twist-less segments: a deterministic placeholder direction orthogonal
        # to the long axis, carried by the damped minimal-roll tier. Picked from
        # a canonical reference so it is never collinear with any long axis.
        candidate = _default_perpendicular(rest_dirs[segment.name])

    if abs(float(np.dot(rest_dirs[segment.name], candidate))) > _COLLINEARITY_DOT:
        raise ValueError(
            f"segment {segment.name!r}: rest approximate axis is collinear with "
            f"its long axis — author an override"
        )
    return candidate


def _default_perpendicular(long_dir: NDArray[np.float64]) -> NDArray[np.float64]:
    """A deterministic unit direction orthogonal to *long_dir*.

    Orthogonalizes a canonical reference (``+X`` unless the long axis is near
    ``+X``, then ``+Y``) against *long_dir*, so the result is never collinear
    with it.
    """
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(long_dir, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    perp = ref - float(np.dot(ref, long_dir)) * long_dir
    norm = float(np.linalg.norm(perp))
    if norm < 1e-10:
        # long_dir is ±X and ±Y simultaneously is impossible given the branch,
        # but keep a hard fallback for numerical safety.
        perp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        norm = 1.0
    return perp / norm
