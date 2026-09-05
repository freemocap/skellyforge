"""Portable authored skeleton values, excluding derived indices and numeric caches."""

from dataclasses import dataclass

from skellyforge.core.math.geometry.orthonormal_basis.handedness import Handedness
from skellyforge.core.math.geometry.orthonormal_basis.reference_frame_definition import (
    ReferenceFrameDefinition,
)
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain.kinematic_chain import KinematicChain
from skellyforge.core.skeleton.components.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton.components.landmark_grouping import (
    LandmarkGroup,
    LandmarkConnectionGroup,
)
from skellyforge.core.skeleton.components.rigid_body_segment import RigidBodySegment
from skellyforge.core.skeleton.linkage.joint_definition import (
    JointDefinition,
    EulerConvention,
)
from skellyforge.core.skeleton.pose.rest_pose import RestPose, build_rest_pose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition


@dataclass(frozen=True, slots=True)
class LandmarkSnapshot:
    name: str
    definition: str
    position: tuple[float, float, float]
    segment: str
    aliases: tuple[str, ...]

    @classmethod
    def capture(cls, landmark: AnatomicalLandmark) -> "LandmarkSnapshot":
        x, y, z = landmark.local_position.array
        return cls(
            name=landmark.name,
            definition=landmark.anatomical_definition,
            position=(float(x), float(y), float(z)),
            segment=landmark.segment,
            aliases=landmark.aliases,
        )

    def restore(self) -> AnatomicalLandmark:
        x, y, z = self.position
        return AnatomicalLandmark(
            name=self.name,
            anatomical_definition=self.definition,
            local_position=Point.from_xyz(x=x, y=y, z=z),
            segment=self.segment,
            aliases=self.aliases,
        )


@dataclass(frozen=True, slots=True)
class FrameSnapshot:
    """Signed axes are stored as their domain (coordinate index, sign) pairs."""
    origin: str
    primary_point: str
    primary_axis: tuple[int, int]
    secondary_point: str | None
    secondary_axis: tuple[int, int] | None
    handedness: Handedness

    @classmethod
    def capture(cls, frame: ReferenceFrameDefinition) -> "FrameSnapshot":
        return cls(
            origin=frame.origin_point_name,
            primary_point=frame.primary_point_name,
            primary_axis=frame.primary_axis.value,
            secondary_point=frame.secondary_point_name,
            secondary_axis=frame.secondary_axis.value
            if frame.secondary_axis is not None
            else None,
            handedness=frame.handedness,
        )

    def restore(self) -> ReferenceFrameDefinition:
        return ReferenceFrameDefinition(
            origin_point_name=self.origin,
            primary_point_name=self.primary_point,
            primary_axis=SpatialAxis(self.primary_axis),
            secondary_point_name=self.secondary_point,
            secondary_axis=SpatialAxis(self.secondary_axis)
            if self.secondary_axis is not None
            else None,
            handedness=self.handedness,
        )


@dataclass(frozen=True, slots=True)
class SegmentSnapshot:
    name: str
    landmarks: tuple[str, ...]
    frame: FrameSnapshot
    aliases: tuple[str, ...]
    anatomical_segment: str | None


@dataclass(frozen=True, slots=True)
class JointSnapshot:
    name: str
    parent: str
    child: str
    connect_at: str
    joint_type: str
    convention: EulerConvention


@dataclass(frozen=True, slots=True)
class ChainSnapshot:
    name: str
    segments: tuple[str, ...]
    joints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkeletonSnapshot:
    name: str
    landmarks: tuple[LandmarkSnapshot, ...]
    segments: tuple[SegmentSnapshot, ...]
    joints: tuple[JointSnapshot, ...]
    chains: tuple[ChainSnapshot, ...]
    landmark_groups: tuple[LandmarkGroup, ...]
    landmark_connections: tuple[LandmarkConnectionGroup, ...]
    derived_quantities: tuple[str, ...]
    coordinate_system: str

    def __post_init__(self) -> None:
        for records in (
            self.landmarks,
            self.segments,
            self.joints,
            self.chains,
            self.landmark_groups,
            self.landmark_connections,
        ):
            names = [record.name for record in records]
            if len(set(names)) != len(names):
                raise ValueError("Skeleton snapshot contains duplicate names")

    @classmethod
    def capture(cls, skeleton: SkeletonDefinition) -> "SkeletonSnapshot":
        return cls(
            name=skeleton.name,
            landmarks=tuple(
                LandmarkSnapshot.capture(item) for item in skeleton.landmarks.values()
            ),
            segments=tuple(
                SegmentSnapshot(
                    name=item.name,
                    landmarks=tuple(item.landmarks),
                    frame=FrameSnapshot.capture(item.frame_definition),
                    aliases=item.aliases,
                    anatomical_segment=item.anatomical_segment,
                )
                for item in skeleton.segments.values()
            ),
            joints=tuple(
                JointSnapshot(
                    name=item.name,
                    parent=item.parent.name,
                    child=item.child.name,
                    connect_at=item.connect_at.name,
                    joint_type=item.joint_type,
                    convention=item.convention,
                )
                for item in skeleton.joints.values()
            ),
            chains=tuple(
                ChainSnapshot(
                    name=item.name,
                    segments=tuple(segment.name for segment in item.segments),
                    joints=tuple(joint.name for joint in item.joints),
                )
                for item in skeleton.chains.values()
            ),
            landmark_groups=tuple(skeleton.landmark_groups.values()),
            landmark_connections=tuple(skeleton.landmark_connections.values()),
            derived_quantities=tuple(sorted(skeleton.derived_quantities)),
            coordinate_system=skeleton.coordinate_system,
        )

    def restore(self) -> SkeletonDefinition:
        landmarks = {item.name: item.restore() for item in self.landmarks}
        segments = {
            item.name: RigidBodySegment(
                name=item.name,
                landmarks={name: landmarks[name] for name in item.landmarks},
                frame_definition=item.frame.restore(),
                aliases=item.aliases,
                anatomical_segment=item.anatomical_segment,
            )
            for item in self.segments
        }
        joints = {
            item.name: JointDefinition(
                name=item.name,
                parent=segments[item.parent],
                child=segments[item.child],
                connect_at=landmarks[item.connect_at],
                joint_type=item.joint_type,
                convention=item.convention,
            )
            for item in self.joints
        }
        chains = {
            item.name: KinematicChain(
                name=item.name,
                segments=tuple(segments[name] for name in item.segments),
                joints=tuple(joints[name] for name in item.joints),
            )
            for item in self.chains
        }
        return SkeletonDefinition(
            name=self.name,
            landmarks=landmarks,
            segments=segments,
            joints=joints,
            chains=chains,
            landmark_groups={item.name: item for item in self.landmark_groups},
            landmark_connections={
                item.name: item for item in self.landmark_connections
            },
            derived_quantities=frozenset(self.derived_quantities),
            coordinate_system=self.coordinate_system,
        )


@dataclass(frozen=True, slots=True)
class RestPoseSnapshot:
    name: str
    root_segment_name: str
    parents: dict[str, str | None]
    connect_ats: dict[str, str]
    orientations: dict[str, RotationQuaternion]

    @classmethod
    def capture(cls, rest: RestPose) -> "RestPoseSnapshot":
        return cls(
            name=rest.name,
            root_segment_name=rest.root_segment_name,
            parents=dict(rest.parents),
            connect_ats=dict(rest.connect_ats),
            orientations=dict(rest.relative_orientations),
        )

    def restore(self, skeleton: SkeletonDefinition) -> RestPose:
        if set(self.parents) != set(skeleton.segments) or set(self.orientations) != set(
            skeleton.segments
        ):
            raise ValueError("Rest pose must cover the skeleton segments")
        if {name for name, parent in self.parents.items() if parent is None} != {
            self.root_segment_name
        }:
            raise ValueError("Rest pose must have exactly its declared root")
        for joint in skeleton.joints.values():
            if (
                self.parents[joint.child.name] != joint.parent.name
                or self.connect_ats[joint.child.name] != joint.connect_at.name
            ):
                raise ValueError("Rest pose topology disagrees with the skeleton")
        rotations, origins, landmarks = build_rest_pose(
            skeleton=skeleton,
            parents=self.parents,
            connect_ats=self.connect_ats,
            orientations=self.orientations,
        )
        return RestPose(
            name=self.name,
            root_segment_name=self.root_segment_name,
            parents=self.parents,
            connect_ats=self.connect_ats,
            relative_orientations=self.orientations,
            segment_orientations=rotations,
            segment_origins=origins,
            landmark_positions=landmarks,
        )
