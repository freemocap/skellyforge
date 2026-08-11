"""Kinematics engine — rigid-body math layer for the standard human model.

This package provides the mathematical core that the orientation solver
(SF-SH-4) and the canonical frame aggregator call per-frame. Everything
here is hot-path safe: dataclasses with ``__slots__`` for scalar ops,
pure numpy functions for vectorized batch ops. No Pydantic, no
serialization, no wire-format awareness.

Package structure:
    quaternion_math.py          — Quaternion dataclass + all vectorized
                                  quaternion operations (single home)
    coordinate_frame_ops.py     — Runtime basis construction from live
                                  landmarks vs reference geometry
    rigid_body_kinematics.py    — Aggregate model: pose arrays -> derived
                                  kinematics + module-level vectorized fns
"""

from skellyforge.kinematics.coordinate_frame_ops import (
    align_point_sets_kabsch,
    build_orthonormal_basis,
    compute_live_bone_basis,
    compute_rotation_from_live_basis,
    rotation_between_vectors,
)
from skellyforge.kinematics.quaternion_math import (
    Quaternion,
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
from skellyforge.kinematics.rigid_body_kinematics import (
    RigidBodyKinematics,
    compute_angular_acceleration,
    compute_keypoint_world_positions,
    compute_linear_acceleration,
    compute_linear_velocity,
)

__all__ = [
    # quaternion_math
    "Quaternion",
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
    "build_orthonormal_basis",
    "compute_live_bone_basis",
    "compute_rotation_from_live_basis",
    "rotation_between_vectors",
    # rigid_body_kinematics
    "RigidBodyKinematics",
    "compute_angular_acceleration",
    "compute_keypoint_world_positions",
    "compute_linear_acceleration",
    "compute_linear_velocity",
]
