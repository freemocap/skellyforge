"""The skeleton's rest pose: each segment's parent and T-pose orientation.

The rest pose is the neutral T-pose the skeleton is authored in, mirroring the VRM
default humanoid. Each segment names its parent and a parent-relative rotation; walking
the tree composes them into per-segment world transforms. Lengths stay derived from the
landmarks' rest positions, so the rest pose adds only the hierarchy and the roll that the
per-segment reference frames leave unspecified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Displacement, Point
from skellyforge.core.skeleton_parts.skeleton_definition import SkeletonDefinition
from skellyforge.type_overloads import LandmarkNameString, RigidBodySegmentName

SEGMENT_ENTRY_KEYS = frozenset({"parent", "connect_at", "orientation"})


@dataclass(frozen=True, slots=True, eq=False)
class RestPose:
    """The T-pose, resolved to world transforms per segment and landmark."""

    name: str
    segment_orientations: Mapping[RigidBodySegmentName, RotationQuaternion]
    segment_origins: Mapping[RigidBodySegmentName, Point]
    landmark_positions: Mapping[LandmarkNameString, Point]

    @classmethod
    def from_yaml(cls, *, path: Path, skeleton: SkeletonDefinition) -> "RestPose":
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise ValueError(f"{path} must parse to a mapping, got {type(document).__name__}")

        entries = document.get("segments")
        if not isinstance(entries, Mapping):
            raise ValueError(f"{path} needs a 'segments' mapping")

        parents: dict[RigidBodySegmentName, RigidBodySegmentName | None] = {}
        orientations: dict[RigidBodySegmentName, RotationQuaternion] = {}
        connect_ats: dict[RigidBodySegmentName, LandmarkNameString | None] = {}

        for segment in skeleton.segments.values():
            entry = entries.get(segment.name) or {}
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"rest pose entry for {segment.name!r} must be a mapping, got {type(entry).__name__}"
                )
            unknown_keys = set(entry) - SEGMENT_ENTRY_KEYS
            if unknown_keys:
                raise ValueError(
                    f"rest pose entry for {segment.name!r}: unknown keys {sorted(unknown_keys)}"
                )

            parent = entry.get("parent")
            if parent is not None and parent not in skeleton.segments:
                raise ValueError(
                    f"rest pose names parent {parent!r} for {segment.name!r}, which is not a segment"
                )

            orientation_list = entry.get("orientation")
            if orientation_list is None:
                orientation = RotationQuaternion.identity()
            else:
                if (
                    not isinstance(orientation_list, Sequence)
                    or isinstance(orientation_list, (str, bytes))
                    or len(orientation_list) != 4
                ):
                    raise ValueError(
                        f"rest pose orientation for {segment.name!r} must be [w, x, y, z]"
                    )
                w, x, y, z = (float(value) for value in orientation_list)
                orientation = RotationQuaternion.from_components(w=w, x=x, y=y, z=z)

            connect_at = entry.get("connect_at")
            if connect_at is not None:
                if connect_at not in skeleton.landmarks:
                    raise ValueError(
                        f"rest pose names connect_at {connect_at!r} for {segment.name!r}, not a landmark"
                    )
                if parent is not None and skeleton.landmarks[connect_at].segment != parent:
                    raise ValueError(
                        f"rest pose connect_at {connect_at!r} for {segment.name!r} must be owned by its parent {parent!r}"
                    )

            parents[segment.name] = parent
            orientations[segment.name] = orientation
            connect_ats[segment.name] = connect_at

        world_orientations: dict[RigidBodySegmentName, RotationQuaternion] = {}
        world_origins: dict[RigidBodySegmentName, Point] = {}
        visiting: set[RigidBodySegmentName] = set()
        visited: set[RigidBodySegmentName] = set()

        def resolve(name: RigidBodySegmentName) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"rest pose has a parent cycle at {name!r}")
            visiting.add(name)
            parent = parents[name]
            if parent is None:
                world_orientations[name] = orientations[name]
                world_origins[name] = Point.from_xyz(x=0.0, y=0.0, z=0.0)
            else:
                resolve(parent)
                parent_orientation = world_orientations[parent]
                world_orientations[name] = parent_orientation * orientations[name]
                connect_at = (
                    connect_ats[name]
                    or skeleton.segments[name].frame_definition.origin_point_name
                )
                offset = Displacement.from_prevalidated_array(
                    array=parent_orientation.rotate_vector(
                        skeleton.landmarks[connect_at].local_position.array
                    )
                )
                world_origins[name] = world_origins[parent] + offset
            visiting.remove(name)
            visited.add(name)

        for segment_name in skeleton.segments:
            resolve(segment_name)

        landmark_positions: dict[LandmarkNameString, Point] = {}
        for landmark in skeleton.landmarks.values():
            offset = Displacement.from_prevalidated_array(
                array=world_orientations[landmark.segment].rotate_vector(
                    landmark.local_position.array
                )
            )
            landmark_positions[landmark.name] = world_origins[landmark.segment] + offset

        return cls(
            name=str(document.get("name", path.stem)),
            segment_orientations=world_orientations,
            segment_origins=world_origins,
            landmark_positions=landmark_positions,
        )
