"""Rigid-body hydration math: closed-form solvers on observed landmark positions.

Phase 3 (hydration) turns observed landmark positions into per-segment poses. This package
holds the pure math: the rigid fit (Kabsch) that recovers a segment's rotation and
translation from landmarks whose pairwise distances are fixed, and the shortest-arc rotation
that turns a two-landmark direction into an orientation (whose free roll is resolved
downstream by a continuous parallel-transport convention).
"""

from skellyforge.core.math.kinematics.coordinate_frame_ops import (
    primary_axis_unit,
    rotation_between_vectors,
)
from skellyforge.core.math.kinematics.rigid_point_set import (
    RigidPointSet,
    align_point_sets_kabsch,
)

__all__ = [
    "RigidPointSet",
    "align_point_sets_kabsch",
    "primary_axis_unit",
    "rotation_between_vectors",
]
