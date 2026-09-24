"""Forward kinematic synthesis: joint angles -> whole-body poses.

The chain layer's forward math. Given a per-joint relative rotation for every
joint of the skeleton (plus where the root sits), walk the joint tree once and
produce every segment's world orientation and origin, and every landmark's
world position. This is the rest pose's forward pass generalized from one
authored pose to any authored motion - and it is the `F` in the closure test:
angles -> landmarks -> hydration -> recovered angles must round-trip.

The companion inverse direction (landmarks -> poses) is NOT here - that is
hydration (`core/skeleton/pose/hydration.py`), which measures; this module
only synthesizes. Synthesized output is exact by construction: closed-form
composition, no iteration, no repair.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.model_scale_fitting import ModelScaleFit
from skellyforge.core.skeleton.pose.rest_pose import build_rest_pose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.type_overloads import (
    FloatArray,
    LandmarkNameString,
    RigidBodySegmentName,
)


def synthesize_pose(
    *,
    skeleton: SkeletonDefinition,
    joint_relative_orientations: Mapping[str, RotationQuaternion],
    root_world_orientation: RotationQuaternion | None = None,
    root_origin: Point | None = None,
    segment_scales: Mapping[str, float] | None = None,
    segment_names: frozenset[str] | None = None,
) -> tuple[
    dict[RigidBodySegmentName, RotationQuaternion],
    dict[RigidBodySegmentName, Point],
    dict[LandmarkNameString, Point],
]:
    """Walk the joint tree once, producing every segment and landmark in world.

    Args:
        skeleton: the skeleton whose joints define the tree.
        joint_relative_orientations: one rotation per joint NAME, expressed in
            the parent segment's frame - exactly what a joint's hydrated face
            reports, so synthesis and measurement speak the same language. The
            root segment's entry (keyed by its own name) sets its world
            orientation; defaults to identity.
        root_world_orientation: alternative way to set the root's world
            orientation, overriding the map's root entry when given.
        root_origin: where the root segment sits in world. Defaults to the
            world origin.
        segment_scales: positive world units per template unit, covering the
            selected segments. Omit for the original unit-template behavior.
        segment_names: optional selection including the root and every ancestor.

    Returns:
        ``(world_orientations, world_origins, landmark_positions)`` keyed by
        segment / landmark name - the same shape ``build_rest_pose`` returns,
        of which this is the time-varying generalization.

    Raises:
        ValueError: the selection lacks its root or an ancestor, or its scales
            are missing, non-finite, or non-positive.
        KeyError: a joint has no entry in `joint_relative_orientations` -
            synthesis refuses to invent rotations.
    """
    parents: dict[RigidBodySegmentName, RigidBodySegmentName | None] = {
        segment.name: None for segment in skeleton.segments.values()
    }
    connect_ats: dict[RigidBodySegmentName, str] = {}
    orientations_by_segment: dict[RigidBodySegmentName, RotationQuaternion] = {}

    for joint in skeleton.joints.values():
        parents[joint.child.name] = joint.parent.name
    roots = [name for name, parent in parents.items() if parent is None]
    if len(roots) != 1:
        raise ValueError("Synthesis requires exactly one root")
    root_name = roots[0]
    selected = set(parents) if segment_names is None else set(segment_names)
    if not selected or not selected.issubset(parents) or root_name not in selected:
        raise ValueError("Selection must contain the root and only known segments")
    if any(
        parents[name] is not None and parents[name] not in selected for name in selected
    ):
        raise ValueError("Selection must include every ancestor")
    parents = {name: parent for name, parent in parents.items() if name in selected}
    if segment_scales is not None:
        if not selected.issubset(segment_scales) or any(
            not np.isfinite(segment_scales[name]) or segment_scales[name] <= 0
            for name in selected
        ):
            raise ValueError(
                "Selected segment scales must be present, finite and positive"
            )
    missing = [
        joint.name
        for joint in skeleton.joints.values()
        if joint.child.name in selected
        and joint.name not in joint_relative_orientations
    ]
    if missing:
        raise KeyError(
            f"synthesis needs a relative orientation for every joint - missing "
            f"{sorted(missing)}"
        )

    for joint in skeleton.joints.values():
        if joint.child.name not in selected:
            continue
        parents[joint.child.name] = joint.parent.name
        connect_ats[joint.child.name] = joint.connect_at.name
        orientations_by_segment[joint.child.name] = joint_relative_orientations[
            joint.name
        ]

    orientations_by_segment[root_name] = (
        root_world_orientation
        if root_world_orientation is not None
        else joint_relative_orientations.get(root_name) or RotationQuaternion.identity()
    )
    connect_ats[root_name] = skeleton.segments[
        root_name
    ].frame_definition.origin_point_name

    world_orientations, world_origins, landmark_positions = build_rest_pose(
        skeleton=skeleton,
        parents=parents,
        connect_ats=connect_ats,
        orientations=orientations_by_segment,
        segment_scales=segment_scales,
    )

    if root_origin is not None:
        translation = root_origin.array
        world_origins = {
            name: Point.from_prevalidated_array(array=origin.array + translation)
            for name, origin in world_origins.items()
        }
        landmark_positions = {
            name: Point.from_prevalidated_array(array=point.array + translation)
            for name, point in landmark_positions.items()
        }

    return world_orientations, world_origins, landmark_positions


def synthesize_fitted_pose(
    *,
    skeleton: SkeletonDefinition,
    fit: ModelScaleFit,
    segment_relative_orientations: Mapping[str, RotationQuaternion],
    root_world_orientation: RotationQuaternion,
    root_origin: Point,
    segment_names: frozenset[str] | None = None,
) -> tuple[dict[str, RotationQuaternion], dict[str, Point], dict[str, Point]]:
    """Produce one connected pose at fitted dimensions, without refitting motion.

    Local rotations are absolute parent-relative rotations, already including rest
    orientation. Keys are selected non-root SEGMENT names (as in saved local
    rotations), not joint names. Root motion is always explicit. Freeze the fit and
    selection across frames for a fixed-dimension animation; all positions remain
    in the fit/root's shared units. Missing selected rotations fail: no gap filling,
    IK, rest-pose fallback, or temporal resampling is performed here.

    Connected origins are synthesized from parent-owned attachments, so they need
    not coincide with independently measured segment origins. The input model,
    fit, rotations, and root are not modified. Full measured poses stay separate.
    """
    selected = set(skeleton.segments) if segment_names is None else set(segment_names)
    if set(fit.segment_scales) != set(skeleton.segments):
        raise ValueError("Fit must cover the skeleton's exact segment set")
    for name, segment in skeleton.segments.items():
        if not np.isclose(
            fit.segment_lengths[name],
            segment.length * fit.segment_scales[name],
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(f"Fit length and template geometry disagree for {name!r}")
    if root_origin.array.shape != (3,) or not np.isfinite(root_origin.array).all():
        raise ValueError("Root origin must be one finite 3D point")
    required = {
        joint.child.name
        for joint in skeleton.joints.values()
        if joint.child.name in selected
    }
    if set(segment_relative_orientations) != required:
        raise ValueError("Rotations must cover exactly the selected non-root segments")
    return synthesize_pose(
        skeleton=skeleton,
        joint_relative_orientations={
            joint.name: segment_relative_orientations[joint.child.name]
            for joint in skeleton.joints.values()
            if joint.child.name in selected
        },
        root_world_orientation=root_world_orientation,
        root_origin=root_origin,
        segment_scales=fit.segment_scales,
        segment_names=segment_names,
    )


def synthesize_from_euler(
    *,
    skeleton: SkeletonDefinition,
    joint_euler_angles: Mapping[str, FloatArray],
    root_world_orientation: RotationQuaternion | None = None,
    root_origin: Point | None = None,
) -> tuple[
    dict[RigidBodySegmentName, RotationQuaternion],
    dict[RigidBodySegmentName, Point],
    dict[LandmarkNameString, Point],
]:
    """`synthesize_pose` with per-joint euler triples, composed via each joint's
    own convention.

    This is the authored-input boundary: humans and animation clips speak
    angles, so they are converted to quaternions immediately, here, and every
    computation after this call is quaternion-only.
    """
    joint_relative_orientations = {}
    for joint_name, triple in joint_euler_angles.items():
        from skellyforge.core.math.kinematics.euler_sequence import (
            compose_euler_angles,
        )

        convention = skeleton.joints[joint_name].convention
        joint_relative_orientations[joint_name] = compose_euler_angles(
            sequence=convention.sequence,
            angles=np.asarray(triple, dtype=np.float64)
            + np.asarray(convention.zero_offsets),
        )
    return synthesize_pose(
        skeleton=skeleton,
        joint_relative_orientations=joint_relative_orientations,
        root_world_orientation=root_world_orientation,
        root_origin=root_origin,
    )
