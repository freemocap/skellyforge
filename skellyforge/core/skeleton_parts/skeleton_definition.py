"""A whole skeleton: every landmark and every segment, loaded from YAML.

The skeleton file names its components; each component contributes landmarks and
segments, and this is the first place that can see all of them at once - so this is where
global checks live. A name may be claimed by exactly one landmark and one segment across
the entire skeleton, every landmark's `reference_frame` must name a real segment, and
every landmark a segment's frame definition refers to must exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from skellyforge.core.skeleton_parts.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton_parts.landmark_name_resolver import LandmarkNameResolver
from skellyforge.core.skeleton_parts.rigid_body_segment import RigidBodySegment
from skellyforge.core.skeleton_parts.skeleton_yaml_loader import (
    build_component,
    load_component,
    resolve_includes,
)
from skellyforge.type_overloads import LandmarkNameString, RigidBodySegmentName

SkeletonNameString = str


@dataclass(frozen=True, slots=True, eq=False)
class SkeletonDefinition:
    """Every landmark and segment of one skeleton, keyed by name.

    Attributes:
        name: what this skeleton is called.
        landmarks: every landmark, keyed by its canonical lowercase name.
        segments: every segment, keyed by its canonical lowercase name.
    """

    name: SkeletonNameString
    landmarks: Mapping[LandmarkNameString, AnatomicalLandmark]
    segments: Mapping[RigidBodySegmentName, RigidBodySegment]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("skeleton name must be non-empty")
        if not self.segments:
            raise ValueError(f"skeleton {self.name!r} has no segments")

        # Enforces that no two landmarks anywhere share a name or an alias.
        LandmarkNameResolver.from_landmarks(landmarks=self.landmarks.values())

        segment_names = {
            known_name
            for segment in self.segments.values()
            for known_name in segment.all_names
        }
        orphaned = sorted(
            {
                landmark.name: landmark.segment
                for landmark in self.landmarks.values()
                if landmark.segment not in segment_names
            }.items()
        )
        if orphaned:
            raise ValueError(
                f"skeleton {self.name!r}: these landmarks name a `reference_frame` that "
                f"is not a segment of this skeleton - {orphaned}. Known segments: "
                f"{sorted(segment_names)}"
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
        """Folds every alias in this skeleton down to its canonical landmark name."""
        return LandmarkNameResolver.from_landmarks(landmarks=self.landmarks.values())

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
        )

    @classmethod
    def from_component_yaml(cls, *, path: Path, name: SkeletonNameString) -> SkeletonDefinition:
        """Load a single component file as a skeleton in its own right.

        Useful while components are still being brought up one at a time: a component that
        is internally consistent can be loaded and checked before the whole skeleton is.
        """
        landmarks, segments = load_component(path=path, name=name)
        return cls(name=name, landmarks=landmarks, segments=segments)


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
