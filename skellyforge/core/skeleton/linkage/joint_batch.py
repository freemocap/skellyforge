"""Joint rotations and angles over frame chunks with named axes and provenance."""

from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import UNIT_QUATERNION_TOLERANCE
from skellyforge.core.math.geometry.rotation_quaternion import (
    conjugate_quaternion_array, hamilton_product, normalize_quaternion_array,
)
from skellyforge.core.math.kinematics.euler_sequence import decompose_euler_batch
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.pose.pose_tensor import PoseTensor, SolutionCode, encode_solutions, decode_solutions

EULER_COMPONENTS = ("first_angle", "middle_angle", "last_angle")
QUATERNION_COMPONENTS = ("w", "x", "y", "z")


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
class JointTensor:
    """Named sample/joint arrays with numeric parent/child solution codes.

    Construction validates borrowed arrays. JointBatchGeometry returns new arrays
    independent of its pose input, with angles in radians in authored conventions.
    """
    joint_names: tuple[str, ...]
    angle_names: tuple[tuple[str, ...], ...]
    relative_orientations: np.ndarray
    angles: np.ndarray
    joint_present: np.ndarray
    parent_codes: np.ndarray
    child_codes: np.ndarray

    def __post_init__(self):
        if (not isinstance(self.joint_names, tuple) or len(set(self.joint_names)) != len(self.joint_names)
                or any(not isinstance(name, str) or not name for name in self.joint_names)):
            raise ValueError("Joint identities must be unique nonempty names in a tuple")
        if not isinstance(self.angle_names, tuple) or len(self.angle_names) != len(self.joint_names):
            raise ValueError("Joint angle declarations must match joint identities")
        for names in self.angle_names:
            if (not isinstance(names, tuple) or not names or len(set(names)) != len(names)
                    or any(not isinstance(name, str) or not name for name in names)):
                raise ValueError("Each joint must declare unique nonempty angle names")
        widths = {len(names) for names in self.angle_names}
        if len(widths) > 1:
            raise ValueError("Joint tensor requires one declared angle component width")
        angle_width = next(iter(widths), len(EULER_COMPONENTS))
        presence = self.joint_present
        if presence.dtype != bool or presence.ndim != 2 or presence.shape[1] != len(self.joint_names):
            raise ValueError("Joint presence must be boolean on sample/joint axes")
        for values, width in ((self.relative_orientations, len(QUATERNION_COMPONENTS)), (self.angles, angle_width)):
            if values.dtype != np.float64 or values.shape != (*presence.shape, width):
                raise ValueError("Joint values must be float64 with declared component axes")
            if not np.isfinite(values[presence]).all() or not np.isnan(values[~presence]).all():
                raise ValueError("Present joint values must be finite and unavailable values NaN")
        if np.any(np.abs(np.linalg.norm(self.relative_orientations[presence], axis=-1) - 1) > UNIT_QUATERNION_TOLERANCE):
            raise ValueError("Joint rotations must be unit wxyz quaternions")
        for codes in (self.parent_codes, self.child_codes):
            if (codes.dtype != np.uint8 or codes.shape != presence.shape
                    or np.any(codes > SolutionCode.TRANSPORTED_ROLL)
                    or np.any((codes != SolutionCode.MISSING) != presence)):
                raise ValueError("Joint solution codes must describe exactly the present joints")


@dataclass(frozen=True)
class JointBatchGeometry:
    """Prepare model indices/conventions once, then calculate frame chunks."""

    segment_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    angle_names: tuple[tuple[str, ...], ...]
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
            frozen([j.convention.zero_offsets for j in joints], np.float64).reshape((-1, len(EULER_COMPONENTS))),
            tuple((sequence, frozen(indices, np.int64)) for sequence, indices in groups.items()))

    def calculate(self, *, world_orientations: np.ndarray, segment_present: np.ndarray,
                  segment_solutions: np.ndarray) -> JointAngleBatch:
        """Adapt existing enum-provenance callers to the numerical joint solver."""
        result = self._calculate_arrays(world_orientations=world_orientations,
            segment_present=segment_present, solution_codes=encode_solutions(segment_solutions))
        return JointAngleBatch(result.relative_orientations, result.angles, result.joint_present,
                               decode_solutions(result.parent_codes), decode_solutions(result.child_codes))

    def calculate_tensor(self, *, poses: PoseTensor) -> JointTensor:
        if poses.segment_names != self.segment_names:
            raise ValueError("Pose segment identities must match prepared joint geometry")
        return self._calculate_arrays(world_orientations=poses.orientations,
            segment_present=poses.segment_present, solution_codes=poses.solution_codes)

    def _calculate_arrays(self, *, world_orientations: np.ndarray, segment_present: np.ndarray,
                          solution_codes: np.ndarray) -> JointTensor:
        """Batch independent frame/joint pairs; never bridge missing segments.

        Input shapes are (F,S,4), boolean (F,S), and uint8 (F,S) solution codes.
        Inputs are not mutated. No time ordering is assumed or changed.
        """
        rotations = np.asarray(world_orientations, dtype=np.float64)
        present = np.asarray(segment_present)
        solutions = np.asarray(solution_codes)
        if rotations.ndim != 3 or rotations.shape[1:] != (len(self.segment_names), len(QUATERNION_COMPONENTS)):
            raise ValueError("Orientations must have shape (frames, segments, 4)")
        if present.dtype != np.bool_ or present.shape != rotations.shape[:2] or solutions.shape != present.shape:
            raise ValueError("Presence and solutions must match (frames, segments)")
        selected = rotations[present]
        if not np.isfinite(selected).all() or np.any(np.abs(np.linalg.norm(selected, axis=1) - 1) > UNIT_QUATERNION_TOLERANCE):
            raise ValueError("Present orientations must be finite unit quaternions")
        if solutions.dtype != np.uint8 or not np.isin(solutions[present], list(SolutionCode)[1:]).all():
            raise ValueError("Present segments must declare PoseSolution provenance")
        joint_present = present[:, self.parent_indices] & present[:, self.child_indices]
        parent = rotations[:, self.parent_indices][joint_present]
        child = rotations[:, self.child_indices][joint_present]
        relative = np.full((*joint_present.shape, len(QUATERNION_COMPONENTS)), np.nan)
        relative[joint_present] = normalize_quaternion_array(quaternions=hamilton_product(
            left=conjugate_quaternion_array(quaternions=parent), right=child))
        angle_width = len(self.angle_names[0]) if self.angle_names else len(EULER_COMPONENTS)
        angles = np.full((*joint_present.shape, angle_width), np.nan)
        # At most twelve Euler conventions; every frame and joint in a group is
        # evaluated together, including the original singularity branches.
        for sequence, indices in self.sequence_groups:
            valid = joint_present[:, indices]
            values = np.full((*valid.shape, angle_width), np.nan)
            values[valid] = decompose_euler_batch(quaternions=relative[:, indices][valid], sequence=sequence)
            angles[:, indices] = values + self.zero_offsets[indices]
        return JointTensor(self.joint_names, self.angle_names, relative, angles, joint_present,
            np.where(joint_present, solutions[:, self.parent_indices], np.uint8(0)),
            np.where(joint_present, solutions[:, self.child_indices], np.uint8(0)))
