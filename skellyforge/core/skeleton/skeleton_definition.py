"""A whole skeleton: every landmark and every segment, loaded from YAML.

The skeleton file names its components; each component contributes landmarks and
segments, and this is the first place that can see all of them at once - so this is where
global checks live. A name may be claimed by exactly one landmark and one segment across
the entire skeleton, every landmark's `reference_frame` must name a real segment, every
landmark must end up owned by exactly one segment, and every landmark a segment's frame
definition refers to must exist.

Both name indices - landmark aliases and segment aliases - are built once here and
stored, because they are the answer to "what does this name mean?" for the whole
skeleton, and nothing downstream should be rebuilding them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from skellyforge.core.math.geometry.coordinate_systems.coordinate_system_registry import (
    CoordinateSystemRegistry,
)
from skellyforge.core.skeleton.components.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton.components.landmark_name_resolver import LandmarkNameResolver
from skellyforge.core.skeleton.components.rigid_body_segment import RigidBodySegment
from skellyforge.core.skeleton.loading import (
    build_component,
    load_component,
    resolve_includes,
)
from skellyforge.type_overloads import (
    LandmarkNameString,
    RigidBodySegmentName,
    SkeletonNameString,
)


@dataclass(frozen=True, slots=True, eq=False)
class SkeletonDefinition:
    """Every landmark and segment of one skeleton, keyed by name.

    Attributes:
        name: what this skeleton is called.
        landmarks: every landmark, keyed by its canonical lowercase name.
        segments: every segment, keyed by its canonical lowercase name.
        coordinate_system: the coordinate-system convention (from the registry) the
            authored positions and local frames are expressed in - "blender" for the
            shipped human.
    """

    name: SkeletonNameString
    landmarks: Mapping[LandmarkNameString, AnatomicalLandmark]
    segments: Mapping[RigidBodySegmentName, RigidBodySegment]
    coordinate_system: str = "blender"
    _landmark_name_resolver: LandmarkNameResolver = field(init=False, repr=False)
    _canonical_segment_name_by_known_name: Mapping[
        RigidBodySegmentName, RigidBodySegmentName
    ] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("skeleton name must be non-empty")
        if not self.coordinate_system:
            raise ValueError(
                f"skeleton {self.name!r}: coordinate system name must be non-empty"
            )
        if not self.segments:
            raise ValueError(f"skeleton {self.name!r} has no segments")

        object.__setattr__(
            self,
            "_canonical_segment_name_by_known_name",
            _index_segment_names(skeleton_name=self.name, segments=self.segments),
        )
        # Building the resolver is what enforces that no two landmarks anywhere share a
        # name or an alias, so it is built here rather than on demand.
        object.__setattr__(
            self,
            "_landmark_name_resolver",
            LandmarkNameResolver.from_landmarks(landmarks=self.landmarks.values()),
        )

        orphaned = sorted(
            {
                landmark.name: landmark.segment
                for landmark in self.landmarks.values()
                if landmark.segment not in self._canonical_segment_name_by_known_name
            }.items()
        )
        if orphaned:
            raise ValueError(
                f"skeleton {self.name!r}: these landmarks name an owning segment that "
                f"this skeleton does not have (the `reference_frame` key in the YAML "
                f"becomes a landmark's `segment`) - {orphaned}. Known segment names and "
                f"aliases: {sorted(self._canonical_segment_name_by_known_name)}"
            )

        owners_by_landmark_name: dict[LandmarkNameString, list[RigidBodySegmentName]] = {
            name: [] for name in self.landmarks
        }
        for segment in self.segments.values():
            for landmark_name in segment.landmarks:
                owners_by_landmark_name[landmark_name].append(segment.name)
        unowned = sorted(
            name for name, owners in owners_by_landmark_name.items() if not owners
        )
        if unowned:
            raise ValueError(
                f"skeleton {self.name!r}: these landmarks are owned by no segment - "
                f"{unowned}. A landmark's `reference_frame` must name the segment that "
                "collects it, by that segment's canonical name or one of its aliases, and "
                "in the same component file."
            )
        multiply_owned = sorted(
            (name, owners)
            for name, owners in owners_by_landmark_name.items()
            if len(owners) > 1
        )
        if multiply_owned:
            raise ValueError(
                f"skeleton {self.name!r}: these landmarks are owned by more than one "
                f"segment - {multiply_owned}"
            )

        dangling = sorted(
            {
                (segment.name, point_name)
                for segment in self.segments.values()
                for point_name in segment.frame_definition.point_names
                if point_name not in self.landmarks
            }
        )
        if dangling:
            raise ValueError(
                f"skeleton {self.name!r}: these segments' reference geometry names "
                f"landmarks that do not exist - {dangling}"
            )

    @property
    def landmark_name_resolver(self) -> LandmarkNameResolver:
        """Folds every alias in this skeleton down to its canonical landmark name.

        Built once during construction, because that is also what proves landmark names
        and aliases are globally unique. Reading it is a lookup, not a rebuild.
        """
        return self._landmark_name_resolver

    def resolve_segment_name(self, *, name: RigidBodySegmentName) -> RigidBodySegmentName:
        """The canonical name of the segment answering to `name`, canonical or alias."""
        canonical_name = self._canonical_segment_name_by_known_name.get(name)
        if canonical_name is None:
            raise KeyError(
                f"Unknown segment name {name!r} - known names are "
                f"{sorted(self._canonical_segment_name_by_known_name)}"
            )
        return canonical_name

    def owning_segment_name_of(
        self, *, landmark: AnatomicalLandmark
    ) -> RigidBodySegmentName:
        """The canonical name of the segment whose local frame this landmark lives in.

        A landmark's `reference_frame` may name its segment by an alias, so this is the
        only correct way to go from a landmark to its segment.
        """
        return self.resolve_segment_name(name=landmark.segment)

    @property
    def underspecified_segment_names(self) -> tuple[RigidBodySegmentName, ...]:
        """Segments whose roll is not yet pinned down, in sorted order."""
        return tuple(
            sorted(
                segment.name
                for segment in self.segments.values()
                if not segment.is_fully_specified
            )
        )

    @classmethod
    def from_yaml(cls, *, path: Path) -> SkeletonDefinition:
        """Load a skeleton from its top-level YAML file.

        The file names the skeleton and lists its components, each of which is normally an
        `$include` pointing at a component file. Component paths resolve relative to the
        skeleton file's own directory.

        Args:
            path: the skeleton `.yaml` file.

        Returns:
            The fully built and cross-validated `SkeletonDefinition`.

        Raises:
            FileNotFoundError: the skeleton file or one of its components is missing.
            ValueError: the document is malformed, or two components claim one name.
        """
        if not path.is_file():
            raise FileNotFoundError(f"{path} is not a file")
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise ValueError(f"{path} must parse to a mapping, got {type(document).__name__}")

        components = document.get("components")
        if not isinstance(components, Mapping) or not components:
            raise ValueError(f"{path} needs a non-empty `components` mapping")

        coordinate_system = document.get("coordinate_system", "blender")
        if not isinstance(coordinate_system, str) or not coordinate_system:
            raise ValueError(
                f"{path}: 'coordinate_system' must be a non-empty string naming a "
                f"convention in the coordinate-system registry"
            )
        registry = CoordinateSystemRegistry.from_default_yaml()
        if coordinate_system not in registry.conventions:
            raise ValueError(
                f"{path}: unknown coordinate system {coordinate_system!r} - known "
                f"conventions are {sorted(registry.conventions)}"
            )

        landmarks: dict[LandmarkNameString, AnatomicalLandmark] = {}
        segments: dict[RigidBodySegmentName, RigidBodySegment] = {}
        for component_name, component_node in components.items():
            resolved = resolve_includes(
                node=component_node,
                base_directory=path.parent,
                include_stack=(path.resolve(),),
            )
            if not isinstance(resolved, Mapping):
                raise ValueError(
                    f"component {component_name!r} must resolve to a mapping - got "
                    f"{type(resolved).__name__}"
                )
            component_landmarks, component_segments = build_component(
                component=resolved, name=str(component_name)
            )
            _merge_into(
                merged=landmarks,
                additions=component_landmarks,
                component_name=str(component_name),
                what="landmark",
            )
            _merge_into(
                merged=segments,
                additions=component_segments,
                component_name=str(component_name),
                what="segment",
            )

        return cls(
            name=str(document.get("name", path.stem)),
            landmarks=landmarks,
            segments=segments,
            coordinate_system=coordinate_system,
        )

    @classmethod
    def from_default_yaml(cls) -> SkeletonDefinition:
        """Load the shipped standard-human skeleton (the canonical 61-segment human)."""
        path = (
            Path(__file__).resolve().parents[2]
            / "definitions"
            / "human_skeleton"
            / "human_skeleton.yaml"
        )
        return cls.from_yaml(path=path)

    @classmethod
    def from_component_yaml(cls, *, path: Path, name: SkeletonNameString) -> SkeletonDefinition:
        """Load a single component file as a skeleton in its own right.

        Useful while components are still being brought up one at a time: a component that
        is internally consistent can be loaded and checked before the whole skeleton is.
        """
        landmarks, segments = load_component(path=path, name=name)
        return cls(name=name, landmarks=landmarks, segments=segments)


def _index_segment_names(
    *,
    skeleton_name: SkeletonNameString,
    segments: Mapping[RigidBodySegmentName, RigidBodySegment],
) -> dict[RigidBodySegmentName, RigidBodySegmentName]:
    """Map every segment name and alias to its canonical name, refusing any claimed twice.

    Segment aliases need the same global uniqueness landmark aliases get: a name that two
    segments both answer to makes `reference_frame: <name>` ambiguous, and the ambiguity
    would otherwise be resolved silently by whichever segment happened to be built last.
    """
    canonical_name_by_known_name: dict[RigidBodySegmentName, RigidBodySegmentName] = {}
    for segment in segments.values():
        for known_name in segment.all_names:
            claimed_by = canonical_name_by_known_name.get(known_name)
            if claimed_by is not None:
                raise ValueError(
                    f"skeleton {skeleton_name!r}: segment names and aliases must be "
                    f"globally unique - {known_name!r} is claimed by both {claimed_by!r} "
                    f"and {segment.name!r}"
                )
            canonical_name_by_known_name[known_name] = segment.name
    return canonical_name_by_known_name


def _merge_into(
    *,
    merged: dict[str, object],
    additions: Mapping[str, object],
    component_name: str,
    what: str,
) -> None:
    """Fold one component's entries into the skeleton, refusing a name claimed twice."""
    for entry_name, entry in additions.items():
        if entry_name in merged:
            raise ValueError(
                f"{what} {entry_name!r} is defined by more than one component - the "
                f"second was {component_name!r}"
            )
        merged[entry_name] = entry
