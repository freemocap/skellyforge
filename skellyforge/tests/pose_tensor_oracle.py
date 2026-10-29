"""Pre-tensor scalar roll algorithm retained solely as an independent test oracle."""
import numpy as np

from skellyforge.core.math.geometry.numeric_tolerances import MINIMUM_VECTOR_NORM
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import UnitVector
from skellyforge.core.math.kinematics.coordinate_frame_ops import rotation_between_vectors
from skellyforge.core.skeleton.chain.twist_backfill import apply_terminal_twist_backfills
from skellyforge.core.skeleton.pose.roll_resolution import SegmentRollReference
from skellyforge.core.skeleton.pose.pose_tensor import PoseTensor
from skellyforge.core.skeleton.skeleton_pose import PoseSolution, SkeletonPose


def scalar_roll_oracle(*, skeleton, rest, poses, reset_before):
    parents = {joint.child.name: joint.parent.name for joint in skeleton.joints.values()}
    axes = {name: SegmentRollReference.for_segment(skeleton=skeleton, segment_name=name) for name in skeleton.segments}
    previous, results = {}, []
    for sample in range(poses.sample_count):
        if reset_before[sample]:
            previous.clear()
        pose = poses.to_pose(sample)
        resolved = {}

        def resolve(name):
            if name in resolved:
                return resolved[name]
            source = pose.segment_poses[name]
            parent = parents.get(name)
            reference = previous.get(name, rest.segment_orientations[name])
            if parent in pose.segment_poses:
                parent_rotation = resolve(parent).orientation
                if name not in previous:
                    reference = parent_rotation * rest.relative_orientations[name]
                elif parent in previous:
                    reference = parent_rotation * previous[parent].inverse() * reference
            result = source
            if source.solved_by is PoseSolution.DIRECTION:
                before = reference.rotate_vector(vector=axes[name].primary_local)
                after = source.orientation.rotate_vector(vector=axes[name].primary_local)
                if np.linalg.norm(np.cross(before, after)) < MINIMUM_VECTOR_NORM and np.dot(before, after) < 0:
                    swing = RotationQuaternion.from_rotation_vector(
                        rotation_vector=reference.rotate_vector(vector=axes[name].secondary_local) * np.pi)
                else:
                    swing = rotation_between_vectors(from_direction=UnitVector.from_array(values=before),
                                                     to_direction=UnitVector.from_array(values=after))
                result = source.with_orientation(orientation=swing * reference, solved_by=PoseSolution.TRANSPORTED_ROLL)
            resolved[name] = result
            return result

        for name in pose.segment_poses:
            resolve(name)
        result = apply_terminal_twist_backfills(skeleton=skeleton, pose=SkeletonPose(segment_poses=resolved),
                                               rest_relative_orientations=rest.relative_orientations)
        if result.segment_poses:
            previous = {name: segment.orientation for name, segment in result.segment_poses.items()}
        results.append(PoseTensor.from_pose(pose=result, segment_names=poses.segment_names))
    return np.concatenate([item.orientations for item in results]), np.concatenate([item.solution_codes for item in results])
