"""Joint rotations and angles over frame chunks with named axes and provenance."""

from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import UNIT_QUATERNION_TOLERANCE
from skellyforge.core.math.geometry.rotation_quaternion import (
    conjugate_quaternion_array, hamilton_product, normalize_quaternion_array,
)
from skellyforge.core.math.kinematics.euler_sequence import decompose_euler_batch
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import PoseSolution


@dataclass(frozen=True)
class JointAngleBatch:
    """Arrays (F,J,4), (F,J,3), (F,J), and two (F,J) provenance arrays.

    Absent joint values are NaN, absent provenance is None. Present provenance
    contains PoseSolution members, preserving measured versus convention-carried
    orientations. Angles are radians in the prepared model's named conventions.
    """

    relative_orientations: np.ndarray
    angles: np.ndarray
    joint_present: np.ndarray
    parent_solutions: np.ndarray
    child_solutions: np.ndarray


@dataclass(frozen=True)
class JointBatchGeometry:
    """Prepare model indices/conventions once, then calculate frame chunks."""

    segment_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    angle_names: tuple[tuple[str, str, str], ...]
    parent_indices: np.ndarray
    child_indices: np.ndarray
    zero_offsets: np.ndarray
    sequence_groups: tuple[tuple[str, np.ndarray], ...]

    @classmethod
    def from_skeleton(cls, *, skeleton: SkeletonDefinition) -> "JointBatchGeometry":
        segments = tuple(skeleton.segments)
        index = {name: i for i, name in enumerate(segments)}
        joints = tuple(skeleton.joints.values())
        groups = {}
        for i, joint in enumerate(joints):
            groups.setdefault(joint.convention.sequence, []).append(i)
        def frozen(values, dtype):
            array = np.asarray(values, dtype=dtype)
            array.setflags(write=False)
            return array
        return cls(segments, tuple(j.name for j in joints),
            tuple(j.convention.angle_names for j in joints),
            frozen([index[j.parent.name] for j in joints], np.int64),
            frozen([index[j.child.name] for j in joints], np.int64),
            frozen([j.convention.zero_offsets for j in joints], np.float64).reshape((-1, 3)),
            tuple((sequence, frozen(indices, np.int64)) for sequence, indices in groups.items()))

    def calculate(self, *, world_orientations: np.ndarray, segment_present: np.ndarray,
                  segment_solutions: np.ndarray) -> JointAngleBatch:
        """Batch independent frame/joint pairs; never bridge missing segments.

        Input shapes are (F,S,4), boolean (F,S), and object (F,S) PoseSolution.
        Inputs are not mutated. No time ordering is assumed or changed.
        """
        rotations = np.asarray(world_orientations, dtype=np.float64)
        present = np.asarray(segment_present)
        solutions = np.asarray(segment_solutions, dtype=object)
        if rotations.ndim != 3 or rotations.shape[1:] != (len(self.segment_names), 4):
            raise ValueError("Orientations must have shape (frames, segments, 4)")
        if present.dtype != np.bool_ or present.shape != rotations.shape[:2] or solutions.shape != present.shape:
            raise ValueError("Presence and solutions must match (frames, segments)")
        selected = rotations[present]
        if not np.isfinite(selected).all() or np.any(np.abs(np.linalg.norm(selected, axis=1) - 1) > UNIT_QUATERNION_TOLERANCE):
            raise ValueError("Present orientations must be finite unit quaternions")
        if not np.isin(solutions[present], list(PoseSolution)).all():
            raise ValueError("Present segments must declare PoseSolution provenance")
        joint_present = present[:, self.parent_indices] & present[:, self.child_indices]
        parent = rotations[:, self.parent_indices][joint_present]
        child = rotations[:, self.child_indices][joint_present]
        relative = np.full((*joint_present.shape, 4), np.nan)
        relative[joint_present] = normalize_quaternion_array(quaternions=hamilton_product(
            left=conjugate_quaternion_array(quaternions=parent), right=child))
        angles = np.full((*joint_present.shape, 3), np.nan)
        # At most twelve Euler conventions; every frame and joint in a group is
        # evaluated together, including the original singularity branches.
        for sequence, indices in self.sequence_groups:
            valid = joint_present[:, indices]
            values = np.full((*valid.shape, 3), np.nan)
            values[valid] = decompose_euler_batch(quaternions=relative[:, indices][valid], sequence=sequence)
            angles[:, indices] = values + self.zero_offsets[indices]
        return JointAngleBatch(relative, angles, joint_present,
            np.where(joint_present, solutions[:, self.parent_indices], None),
            np.where(joint_present, solutions[:, self.child_indices], None))
