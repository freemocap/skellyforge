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

from skellyforge.skellymodels.standard_human.segment_definition import (
    SegmentDefinition,
)

_COLLINEARITY_DOT = 0.9998  # cos(1°) — stiffer than the solver's ~5° gate

# Rest approximate axes for segments whose twist reference cannot be derived
# from rest keypoint positions: either the twist keypoint is collinear with
# the long axis at the T-pose (upper_arm ← wrist, upper_leg ← ankle), or the
# twist keypoint is off every chain and has no rest position (nose, heel,
# small_toe). Authored for the LEFT side; mirroring flips Y for the right.
_TWIST_OVERRIDES: dict[str, NDArray[np.float64]] = {
    "hips": np.array([1.0, 0.0, 0.0]),        # right_hip coincides with the origin
    "spine": np.array([1.0, 0.0, 0.0]),       # right_hip coincides with the origin
    "neck": np.array([1.0, 0.0, 0.0]),        # nose — anterior
    "head": np.array([1.0, 0.0, 0.0]),        # nose — anterior
    "upper_arm": np.array([0.0, 0.0, 1.0]),   # elbow flexion axis
    "upper_leg": np.array([0.0, 1.0, 0.0]),   # knee flexion axis
    "foot": np.array([0.0, 0.0, 1.0]),        # up — the foot points +X, so its
                                             # roll/pitch reference is vertical
    "toes": np.array([0.0, 1.0, 0.0]),        # small toe — lateral (left side)
}
_DEFAULT_APPROXIMATE = np.array([0.0, 0.0, 1.0])  # twist-less segments


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
    """Rows [long, approximate orthonormalized, third] — right-handed."""
    approx_orth = approximate - np.dot(approximate, long_axis) * long_axis
    norm = float(np.linalg.norm(approx_orth))
    if norm < 1e-10:
        raise ValueError("approximate axis is collinear with the long axis")
    approx_orth = approx_orth / norm
    third = np.cross(long_axis, approx_orth)
    third = third / np.linalg.norm(third)
    return np.stack([long_axis, approx_orth, third])


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
        keypoints[segment.long_axis_keypoint] = (
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
    """The twist reference at rest: derived from the twist keypoint's rest
    position where it exists and is off-axis, else the override table, else
    the default perpendicular. Fail loudly if the result is still collinear.
    """
    candidate: NDArray[np.float64] | None = None
    if segment.twist_keypoint is not None and segment.twist_keypoint in keypoints:
        vec = keypoints[segment.twist_keypoint] - origins[segment.name]
        norm = float(np.linalg.norm(vec))
        if norm > 1e-10:
            vec = vec / norm
            if abs(float(np.dot(rest_dirs[segment.name], vec))) <= _COLLINEARITY_DOT:
                candidate = vec

    if candidate is None:
        override = _TWIST_OVERRIDES.get(_unprefixed(segment.name))
        if override is not None:
            candidate = override.copy()
            if segment.name.startswith("right_"):
                candidate = _mirror(candidate)
    if candidate is None:
        candidate = _DEFAULT_APPROXIMATE.copy()

    if abs(float(np.dot(rest_dirs[segment.name], candidate))) > _COLLINEARITY_DOT:
        raise ValueError(
            f"segment {segment.name!r}: rest approximate axis is collinear with "
            f"its long axis — author an override"
        )
    return candidate
