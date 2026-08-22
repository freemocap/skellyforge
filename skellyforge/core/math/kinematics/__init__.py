"""Rigid-body hydration math: closed-form solvers on observed landmark positions.

Phase 3 (hydration) turns observed landmark positions into per-segment poses. This package
holds the pure math: the rigid fit that recovers a segment's rotation and translation from a
set of landmarks whose pairwise distances are fixed (the skull, the pelvis), plus - as the
phase lands - segment length estimation and the critically-damped orientation filter.
"""

from skellyforge.core.math.kinematics.rigid_point_set import (
    RigidPointSet,
    align_point_sets_kabsch,
)

__all__ = ["RigidPointSet", "align_point_sets_kabsch"]
