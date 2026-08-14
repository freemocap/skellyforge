"""Kinematics engine — rigid-body math layer for the standard human model.

This package provides the mathematical core that the orientation solver
(SF-SH-4) and the standard-human frame aggregator call per-frame. Everything
here is hot-path safe: dataclasses for scalar ops, pure numpy functions
for vectorized batch ops. No Pydantic in the hot path, no serialization,
no wire-format awareness, no imports from skellytracker or freemocap.

Package structure:
    quaternion_math.py          — RotationQuaternion dataclass + vectorized ops
    coordinate_frame_ops.py     — Runtime basis construction
    rigid_body_kinematics.py    — Aggregate model + vectorized kinematics
    orientation_solver.py       — Per-bone orientation from live landmarks
    skeleton_rigidifier.py      — Forward-pass skeleton rigidifier
    online_segment_lengths.py   — Rolling-window median segment-length estimator
    segment_lengths.py          — Segment-length measurement + diagnostics
    inertial/                   — Anthropometric BSIP + composite inertia
"""

from skellyforge.kinematics.coordinate_frame_ops import (
    align_point_sets_kabsch,
    build_segment_frame,
    compute_rotation_from_live_basis,
    rotation_between_vectors,
)
from skellyforge.kinematics.online_segment_lengths import (
    SegmentLengthEstimator,
)
from skellyforge.kinematics.orientation_solver import (
    FrameOrientationResult,
    solve_frame_orientations,
)
from skellyforge.kinematics.quaternion_math import (
    RotationQuaternion,
    compose_with_constant,
    compute_angular_velocity,
    conjugate_quaternion_array,
    hamilton_product,
    normalize_quaternion_array,
    quaternion_to_axis_angle,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    rotate_vector_batch,
    rotate_vectors_batch,
    slerp_batch,
    slerp_resample,
)
from skellyforge.kinematics.rigid_point_set import (
    RigidPointTemplate,
    embed_distance_matrix,
    fit_template_to_observed,
)
from skellyforge.kinematics.rigid_body_kinematics import (
    RigidBodyKinematics,
    compute_angular_acceleration,
    compute_keypoint_world_positions,
    compute_linear_acceleration,
    compute_linear_velocity,
)
from skellyforge.kinematics.segment_lengths import (
    DEFAULT_THRESHOLDS,
    LIMB_SEGMENTS,
    HumanShapeThresholds,
    SegmentDef,
    SegmentLengthReport,
    SegmentStats,
    StreamingSegmentLengthMonitor,
    build_segment_length_report,
    canonical_bone_length_ratios,
    equivalence_violations,
    measure_segment_lengths,
    report_from_segment_lengths,
)
from skellyforge.kinematics.skeleton_rigidifier import (
    TreeRigidifier,
)

__all__ = [
    # quaternion_math
    "RotationQuaternion",
    "compose_with_constant",
    "compute_angular_velocity",
    "conjugate_quaternion_array",
    "hamilton_product",
    "normalize_quaternion_array",
    "quaternion_to_axis_angle",
    "quaternion_to_euler",
    "quaternion_to_rotation_matrix",
    "rotate_vector_batch",
    "rotate_vectors_batch",
    "slerp_batch",
    "slerp_resample",
    # coordinate_frame_ops
    "align_point_sets_kabsch",
    "build_segment_frame",
    "compute_rotation_from_live_basis",
    "rotation_between_vectors",
    # rigid_point_set
    "RigidPointTemplate",
    "embed_distance_matrix",
    "fit_template_to_observed",
    # rigid_body_kinematics
    "RigidBodyKinematics",
    "compute_angular_acceleration",
    "compute_keypoint_world_positions",
    "compute_linear_acceleration",
    "compute_linear_velocity",
    # skeleton_rigidifier
    "TreeRigidifier",
    # online_segment_lengths
    "SegmentLengthEstimator",
    # segment_lengths
    "SegmentDef",
    "SegmentStats",
    "SegmentLengthReport",
    "HumanShapeThresholds",
    "DEFAULT_THRESHOLDS",
    "LIMB_SEGMENTS",
    "StreamingSegmentLengthMonitor",
    "measure_segment_lengths",
    "report_from_segment_lengths",
    "build_segment_length_report",
    "canonical_bone_length_ratios",
    "equivalence_violations",
    # orientation_solver
    "FrameOrientationResult",
    "solve_frame_orientations",
]
