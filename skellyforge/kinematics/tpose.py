"""Build the T-pose reference geometry from a loaded HumanSkeleton.

Origins accumulate through the tree (the root at the world origin); each child's
world origin is its origin_landmark's rest position expressed in the parent's
frame. Each landmark's rest-world position is its rest_position in its
reference_frame segment's frame, composed by that frame's transform.

The rest basis is built from the segment's axes, in order: the FIRST axis is the
primary direction (the hard seed), the SECOND (if present) is the twist direction
(the roll reference, Gram-Schmidt'd against the primary). The remaining row fills
via the right-handed cross product.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from skellyforge.kinematics.coordinate_frame_ops import (
    assemble_named_basis,
    axis_index_and_sign,
)
from skellyforge.skellymodels.standard_human.human_skeleton import HumanSkeleton
from skellyforge.skellymodels.standard_human.rigid_body_segment import (
    AxisDefinition,
    RigidBodySegment,
)
from skellyforge.skellymodels.standard_human.standard_human_tpose import (
    SegmentTposeGeometry,
    StandardHumanTPose,
)


def _axis_rest_direction(axis: AxisDefinition) -> NDArray[np.float64]:
    """The axis's authored rest direction, or its row's positive unit vector."""
    if axis.rest_direction is not None:
        return np.asarray(axis.rest_direction, dtype=np.float64)
    idx, _ = axis_index_and_sign(axis.axis)
    return np.eye(3, dtype=np.float64)[idx]


def _default_perpendicular(direction: NDArray[np.float64]) -> NDArray[np.float64]:
    """A deterministic unit direction orthogonal to *direction*."""
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(direction, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    perp = ref - float(np.dot(ref, direction)) * direction
    norm = float(np.linalg.norm(perp))
    if norm < 1e-10:
        perp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        norm = 1.0
    return perp / norm


def _build_rest_basis(segment: RigidBodySegment) -> NDArray[np.float64]:
    """The segment's rest orthonormal frame (rows [x-hat, y-hat, z-hat]).

    The FIRST axis is the primary direction (hard seed); the SECOND (if present)
    is the twist direction (roll reference, Gram-Schmidt'd against the primary).
    """
    primary = segment.axes[0]
    primary_idx, primary_sign = axis_index_and_sign(primary.axis)
    primary_dir = primary_sign * _axis_rest_direction(primary)

    if len(segment.axes) == 1:
        second_dir = _default_perpendicular(primary_dir)
        second_idx = (primary_idx + 1) % 3
        return assemble_named_basis({primary_idx: primary_dir, second_idx: second_dir})

    twist = segment.axes[1]
    twist_idx, twist_sign = axis_index_and_sign(twist.axis)
    twist_dir = twist_sign * _axis_rest_direction(twist)

    twist_orth = twist_dir - float(np.dot(twist_dir, primary_dir)) * primary_dir
    norm = float(np.linalg.norm(twist_orth))
    if norm < 1e-10:
        raise ValueError(
            f"segment {segment.name!r}: rest twist direction is collinear with its "
            f"primary direction — author a distinct direction"
        )
    twist_orth = twist_orth / norm

    return assemble_named_basis({primary_idx: primary_dir, twist_idx: twist_orth})


def build_standard_human_tpose(skeleton: HumanSkeleton) -> StandardHumanTPose:
    """Build the T-pose reference geometry from a loaded HumanSkeleton."""
    children: dict[str, list[RigidBodySegment]] = defaultdict(list)
    roots: list[RigidBodySegment] = []
    for segment in skeleton.segments:
        if segment.parent is None:
            roots.append(segment)
        else:
            children[segment.parent.name].append(segment)

    geometries: dict[str, SegmentTposeGeometry] = {}

    def build(
        segment: RigidBodySegment,
        parent_origin: NDArray[np.float64],
        parent_basis: NDArray[np.float64],
    ) -> None:
        if segment.parent is None:
            origin = np.zeros(3, dtype=np.float64)
        else:
            offset = np.asarray(segment.origin_landmark.rest_position, dtype=np.float64)
            origin = parent_origin + parent_basis.T @ offset
        basis = _build_rest_basis(segment)
        geometries[segment.name] = SegmentTposeGeometry(
            origin=origin, basis=basis, length=segment.length
        )
        for child in children.get(segment.name, []):
            build(child, origin, basis)

    for root in roots:
        build(root, np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64))

    landmarks: dict[str, NDArray[np.float64]] = {}
    for segment in skeleton.segments:
        for lm in segment.landmarks:
            if lm.name in landmarks:
                continue
            geometry = geometries[lm.reference_frame]
            offset = np.asarray(lm.rest_position, dtype=np.float64)
            landmarks[lm.name] = geometry.origin + geometry.basis.T @ offset

    return StandardHumanTPose(segments=geometries, landmarks=landmarks)
