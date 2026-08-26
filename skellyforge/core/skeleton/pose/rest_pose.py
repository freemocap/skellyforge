"""The rest pose (T-pose): per-segment parent-relative orientations.

The rest pose is the neutral T-pose the skeleton is authored in, mirroring the VRM
default humanoid. The segment tree itself lives on the skeleton's joints (the
linkage layer); this file adds only what the joints cannot say - each segment's
parent-relative rest rotation - and composes the two into world transforms.
Lengths stay derived from the landmarks' rest positions.

The loader repairs nothing. Every segment of the skeleton must have an entry and
the entries may carry nothing but an `orientation` (a PARENT-RELATIVE
[w, x, y, z] quaternion, defaulting to identity) - anything else is a rest pose
whose geometry would be silently wrong, so it raises instead of loading.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from skellyforge.core.math.geometry.spatial_vectors import Displacement, Point
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.type_overloads import LandmarkNameString, RigidBodySegmentName

SEGMENT_ENTRY_KEYS = frozenset({"orientation"})
NUMBER_OF_QUATERNION_COMPONENTS = 4


def build_rest_pose(
    *,
    skeleton: SkeletonDefinition,
    parents: Mapping[RigidBodySegmentName, RigidBodySegmentName | None],
    connect_ats: Mapping[RigidBodySegmentName, LandmarkNameString],
    orientations: Mapping[RigidBodySegmentName, RotationQuaternion],
) -> tuple[
    dict[RigidBodySegmentName, RotationQuaternion],
    dict[RigidBodySegmentName, Point],
    dict[LandmarkNameString, Point],
]:
    """Forward kinematics: compose parent-relative orientations into world transforms.

    Walks the tree once, composing each segment's parent-relative orientation onto its
    parent's world orientation, and placing each segment's origin on its parent's
    connect_at landmark. Returns the world orientations, world origins, and world landmark
    positions - the forward direction of hydration, and what a synthetic pose is generated
    from.

    Args:
        skeleton: the skeleton whose segments to place.
        parents: each segment's parent, or `None` for the single root.
        connect_ats: each segment's already-resolved connection landmark, which must be
            owned by that segment's parent. `RestPose.from_yaml` sources these from the
            skeleton's joints; callers building the maps by hand are asserting the same
            invariants.
        orientations: each segment's parent-relative rotation.

    Raises:
        ValueError: the parent map contains a cycle.
        KeyError: a segment, parent or landmark named here is not in the skeleton.
    """
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
            offset = Displacement.from_prevalidated_array(
                array=parent_orientation.rotate_vector(
                    vector=skeleton.landmarks[connect_ats[name]].local_position.array
                )
            )
            world_origins[name] = world_origins[parent] + offset
        visiting.remove(name)
        visited.add(name)

    for segment_name in skeleton.segments:
        resolve(segment_name)

    landmark_positions: dict[LandmarkNameString, Point] = {}
    for landmark in skeleton.landmarks.values():
        owning_segment = skeleton.owning_segment_name_of(landmark=landmark)
        offset = Displacement.from_prevalidated_array(
            array=world_orientations[owning_segment].rotate_vector(
                vector=landmark.local_position.array
            )
        )
        landmark_positions[landmark.name] = world_origins[owning_segment] + offset

    return world_orientations, world_origins, landmark_positions


@dataclass(frozen=True, slots=True, eq=False)
class RestPose:
    """The T-pose, resolved to world transforms per segment and landmark.

    Attributes:
        name: what this rest pose is called.
        root_segment_name: the one segment with no parent; everything hangs off it.
        parents: each segment's parent, `None` for the root (sourced from the
            skeleton's joints).
        connect_ats: each segment's resolved connection landmark, owned by its parent.
            The root's entry is its own origin landmark, which it sits on at the world
            origin.
        relative_orientations: each segment's parent-relative rotation, as authored.
        segment_orientations: each segment's world orientation.
        segment_origins: each segment's world origin.
        landmark_positions: every landmark's world position.
    """

    name: str
    root_segment_name: RigidBodySegmentName
    parents: Mapping[RigidBodySegmentName, RigidBodySegmentName | None]
    connect_ats: Mapping[RigidBodySegmentName, LandmarkNameString]
    relative_orientations: Mapping[RigidBodySegmentName, RotationQuaternion]
    segment_orientations: Mapping[RigidBodySegmentName, RotationQuaternion]
    segment_origins: Mapping[RigidBodySegmentName, Point]
    landmark_positions: Mapping[LandmarkNameString, Point]

    @classmethod
    def from_yaml(cls, *, path: Path, skeleton: SkeletonDefinition) -> RestPose:
        """Load a rest pose and resolve it against a skeleton, refusing anything malformed."""
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise ValueError(f"{path} must parse to a mapping, got {type(document).__name__}")

        entries = document.get("segments")
        if not isinstance(entries, Mapping):
            raise ValueError(f"{path} needs a 'segments' mapping")

        parents, connect_ats = _topology_from_skeleton(path=path, skeleton=skeleton)
        relative_orientations = _read_segment_entries(
            path=path, entries=entries, skeleton=skeleton
        )
        root_segment_name = next(
            name for name, parent in parents.items() if parent is None
        )

        world_orientations, world_origins, landmark_positions = build_rest_pose(
            skeleton=skeleton,
            parents=parents,
            connect_ats=connect_ats,
            orientations=relative_orientations,
        )

        return cls(
            name=str(document.get("name", path.stem)),
            root_segment_name=root_segment_name,
            parents=parents,
            connect_ats=connect_ats,
            relative_orientations=relative_orientations,
            segment_orientations=world_orientations,
            segment_origins=world_origins,
            landmark_positions=landmark_positions,
        )

    @classmethod
    def from_default_yaml(cls, *, skeleton: SkeletonDefinition) -> RestPose:
        """Load the shipped standard-human rest (T) pose, resolved against a skeleton."""
        path = (
            Path(__file__).resolve().parents[3]
            / "definitions"
            / "human_skeleton"
            / "rest_pose.yaml"
        )
        return cls.from_yaml(path=path, skeleton=skeleton)


def _topology_from_skeleton(
    *, path: Path, skeleton: SkeletonDefinition
) -> tuple[
    dict[RigidBodySegmentName, RigidBodySegmentName | None],
    dict[RigidBodySegmentName, LandmarkNameString],
]:
    """The parent map and resolved connect points, read off the skeleton's joints.

    The root's connect point defaults to its own origin landmark, which it sits
    on at the world origin.
    """
    if not skeleton.joints:
        raise ValueError(
            f"{path}: skeleton {skeleton.name!r} defines no joints, so there is no "
            f"segment tree for a rest pose to hang orientations on - add the "
            f"`joints:` section to its YAML"
        )

    parents: dict[RigidBodySegmentName, RigidBodySegmentName | None] = {
        segment.name: None for segment in skeleton.segments.values()
    }
    connect_ats: dict[RigidBodySegmentName, LandmarkNameString] = {}
    for joint in skeleton.joints.values():
        parents[joint.child.name] = joint.parent.name
        connect_ats[joint.child.name] = joint.connect_at.name

    root_name = next(name for name, parent in parents.items() if parent is None)
    root = skeleton.segments[root_name]
    connect_ats[root_name] = root.frame_definition.origin_point_name
    return parents, connect_ats


def _read_segment_entries(
    *,
    path: Path,
    entries: Mapping[str, object],
    skeleton: SkeletonDefinition,
) -> dict[RigidBodySegmentName, RotationQuaternion]:
    """Validate every entry against the skeleton and parse its orientation."""
    unknown_segments = sorted(str(name) for name in entries if name not in skeleton.segments)
    if unknown_segments:
        raise ValueError(
            f"{path}: these entries name segments that are not in skeleton "
            f"{skeleton.name!r} - {unknown_segments}. Known segments: "
            f"{sorted(skeleton.segments)}"
        )
    missing_segments = sorted(name for name in skeleton.segments if name not in entries)
    if missing_segments:
        raise ValueError(
            f"{path}: every segment of skeleton {skeleton.name!r} needs a rest pose "
            f"entry - missing {missing_segments}"
        )

    relative_orientations: dict[RigidBodySegmentName, RotationQuaternion] = {}
    for segment in skeleton.segments.values():
        entry = entries[segment.name] or {}
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"{path}: rest pose entry for {segment.name!r} must be a mapping, got "
                f"{type(entry).__name__}"
            )
        unknown_keys = sorted(set(entry) - SEGMENT_ENTRY_KEYS)
        if unknown_keys:
            raise ValueError(
                f"{path}: rest pose entry for {segment.name!r} has unknown keys "
                f"{unknown_keys} - expected {sorted(SEGMENT_ENTRY_KEYS)}. Segment "
                "topology (parent / connect_at) lives in the skeleton's `joints:` "
                "section, not here."
            )
        relative_orientations[segment.name] = _orientation_of(
            path=path, segment_name=segment.name, value=entry.get("orientation")
        )
    return relative_orientations


def _orientation_of(
    *, path: Path, segment_name: RigidBodySegmentName, value: object
) -> RotationQuaternion:
    """One entry's parent-relative orientation, defaulting to identity."""
    if value is None:
        return RotationQuaternion.identity()
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != NUMBER_OF_QUATERNION_COMPONENTS
    ):
        raise ValueError(
            f"{path}: rest pose orientation for {segment_name!r} must be [w, x, y, z] - "
            f"got {value!r}"
        )
    w, x, y, z = (float(component) for component in value)
    return RotationQuaternion.from_components(w=w, x=x, y=y, z=z)
