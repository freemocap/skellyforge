"""Rigid-body hydration math: closed-form solvers on observed landmark positions.

Phase 3 (hydration) turns observed landmark positions into per-segment poses. This package
holds the pure math: the rigid fit (Kabsch) that recovers a segment's rotation and
translation from landmarks whose pairwise distances are fixed, and the shortest-arc rotation
that turns a two-landmark direction into an orientation. The parallel-transport convention
that resolves the roll that orientation leaves free lives in `skeleton_parts`, because it
needs a skeleton and a pose rather than only vectors.
"""

from skellyforge.core.math.kinematics.coordinate_frame_ops import (
    default_perpendicular,
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
    "default_perpendicular",
    "primary_axis_unit",
    "rotation_between_vectors",
]
