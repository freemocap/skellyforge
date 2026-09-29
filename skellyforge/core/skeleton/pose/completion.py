"""Complete modeled geometry from a partial measured pose, without inventing evidence."""
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import SkeletonPose, SegmentPose
from .rest_pose import RestPose


@dataclass(frozen=True)
class CompletedPose:
    """Connected model output, deliberately not a measured SkeletonPose.

    Inferred segments use parent-relative rest orientation. All landmarks are
    generated from segment transforms and fixed supplied dimensions. This result
    must not be used as new keypoint evidence or as a scale measurement.
    """
    segment_rotations_world: Mapping[str, RotationQuaternion]
    segment_origins: Mapping[str, Point]
    landmarks: Mapping[str, Point]
    inferred_segments: frozenset[str]


def complete_pose(*, skeleton: SkeletonDefinition, rest: RestPose,
                  observed: SkeletonPose, segment_scales: Mapping[str, float],
                  root_seed: SegmentPose | None = None) -> CompletedPose | None:
    """Connect all segments, retaining measured orientations and filling the rest.

    Hydration stays a measurement operation. Call this after roll resolution and
    dimension estimation. Child origins are reconstructed from parent attachments,
    rather than independently measured positions. Missing-root placement requires
    an explicit seed from the caller; it is never placed at an arbitrary origin.
    No measurable segment and no explicit root seed returns no skeleton.
    """
    names = set(skeleton.segments)
    if set(segment_scales) != names or any(not np.isfinite(s) or s <= 0 for s in segment_scales.values()):
        raise ValueError('Completion requires finite positive dimensions for every segment')
    if not set(observed.segment_poses).issubset(names):
        raise ValueError('Observed pose contains unknown segments')
    if set(rest.parents) != names or set(rest.relative_orientations) != names:
        raise ValueError('Rest pose must cover the complete skeleton')
    root = rest.root_segment_name
    if root_seed is not None and root_seed.segment_name != root:
        raise ValueError('Root seed must belong to the skeleton root')
    if not observed.segment_poses and root_seed is None:
        return None
    anchor = observed.segment_poses.get(root, root_seed)
    if anchor is None:
        raise ValueError('Partial pose has no root: supply an explicit root seed')
    rotations, origins = {}, {}
    visiting = set()

    def resolve(name):
        if name in rotations:
            return
        if name in visiting:
            raise ValueError('Rest topology contains a cycle')
        visiting.add(name)
        measured = observed.segment_poses.get(name)
        if name == root:
            rotations[name], origins[name] = anchor.orientation, anchor.origin
        else:
            parent = rest.parents[name]
            if parent is None or parent not in names:
                raise ValueError('Completion requires a single connected root')
            resolve(parent)
            rotations[name] = measured.orientation if measured is not None else rotations[parent] * rest.relative_orientations[name]
            attachment = skeleton.landmarks[rest.connect_ats[name]]
            if attachment.segment != parent:
                raise ValueError('Connection landmark must belong to the parent segment')
            origins[name] = Point.from_array(values=origins[parent].array + rotations[parent].rotate_vector(
                vector=attachment.local_position.array * segment_scales[parent]))
        visiting.remove(name)

    for name in skeleton.segments:
        resolve(name)
    landmarks = {name: Point.from_array(values=origins[landmark.segment].array +
        rotations[landmark.segment].rotate_vector(vector=landmark.local_position.array * segment_scales[landmark.segment]))
        for name, landmark in skeleton.landmarks.items()}
    return CompletedPose(rotations, origins, landmarks, frozenset(names - set(observed.segment_poses)))
