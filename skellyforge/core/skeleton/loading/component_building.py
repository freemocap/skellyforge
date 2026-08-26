"""The loading pipeline assembled: an include-resolved component document into objects.

Runs stages 2-4 over one component document and returns what that file contributes -
landmarks, segments, and the groupings that say what its landmarks are and which of them
connect. Cross-component checks live on SkeletonDefinition, which is the first place that
can see every component at once.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import yaml

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.components.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton.components.landmark_grouping import (
    LandmarkConnectionGroup,
    LandmarkGroup,
    build_landmark_connection_group,
    build_landmark_group,
)
from skellyforge.core.skeleton.components.rigid_body_segment import RigidBodySegment
from skellyforge.core.skeleton.loading.include_resolution import resolve_includes
from skellyforge.core.skeleton.loading.name_lowercasing import _as_list, lowercase_names
from skellyforge.core.skeleton.loading.reference_frame_building import (
    build_reference_frame_definition,
)
from skellyforge.core.skeleton.loading.sided_expansion import expand_sided_entries
from skellyforge.type_overloads import LandmarkNameString, RigidBodySegmentName

LANDMARK_KEYS: Final[frozenset[str]] = frozenset(
    {"aliases", "definition", "reference_frame", "local_position", "sided"}
)
SEGMENT_KEYS: Final[frozenset[str]] = frozenset(
    {"aliases", "reference_geometry", "sided", "anatomical_segment"}
)
NUMBER_OF_SPATIAL_DIMENSIONS: Final[int] = 3
COMPONENT_SECTION_KEYS: Final[frozenset[str]] = frozenset(
    {"landmarks", "segments", "joints", "sided", "landmark_groups", "landmark_connections"}
)


@dataclass(frozen=True, slots=True, eq=False)
class LoadedComponent:
    """What one component file contributes to a skeleton.

    A named record rather than a tuple: this grew from two things to four, and positional
    unpacking of four is exactly where a caller starts getting them the wrong way round.

    Attributes:
        landmarks: this file's landmarks, expanded and lowercased.
        segments: this file's segments, each owning its own landmarks.
        landmark_groups: named sets of landmarks, keyed by group name.
        landmark_connections: named sets of landmark edges, keyed by group name.
    """

    landmarks: dict[LandmarkNameString, AnatomicalLandmark]
    segments: dict[RigidBodySegmentName, RigidBodySegment]
    landmark_groups: dict[str, LandmarkGroup] = field(default_factory=dict)
    landmark_connections: dict[str, LandmarkConnectionGroup] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# Putting the stages together
# ═══════════════════════════════════════════════════════════════════════


def load_component(*, path: Path, name: str) -> LoadedComponent:
    """Load one component YAML file into landmark, segment and grouping objects.

    Cross-component checks belong to `SkeletonDefinition`, which is the first place that
    can see every component at once: a landmark here may legitimately name a
    `reference_frame`, or a group here a landmark, that lives in another file.

    Args:
        path: the component `.yaml` file.
        name: the name this component is known by in the skeleton, used in errors.

    Returns:
        This component's contribution, expanded and lowercased.
    """
    if not path.is_file():
        raise FileNotFoundError(f"component {name!r}: {path} is not a file")
    raw = resolve_includes(
        node=yaml.safe_load(path.read_text(encoding="utf-8")),
        base_directory=path.parent,
        include_stack=(path.resolve(),),
    )
    if not isinstance(raw, Mapping):
        raise ValueError(f"component {name!r}: {path} must parse to a mapping")
    return build_component(component=raw, name=name)


def build_component(*, component: Mapping[str, object], name: str) -> LoadedComponent:
    """Run stages 2-4 over one already-include-resolved component document."""
    lowercased = lowercase_names(node=component)
    if not isinstance(lowercased, Mapping):
        raise ValueError(f"component {name!r} must be a mapping, got {type(component).__name__}")
    unexpected_sections = sorted(set(lowercased) - COMPONENT_SECTION_KEYS)
    if unexpected_sections:
        raise ValueError(
            f"component {name!r}: unexpected top-level sections {unexpected_sections} - "
            f"expected {sorted(COMPONENT_SECTION_KEYS)}"
        )
    expanded = expand_sided_entries(component=lowercased)
    landmarks = {
        landmark_name: _build_landmark(name=landmark_name, entry=entry)
        for landmark_name, entry in expanded["landmarks"].items()
    }
    segments = {
        segment_name: _build_segment(
            name=segment_name, entry=entry, landmarks=landmarks
        )
        for segment_name, entry in expanded["segments"].items()
    }
    # Groupings are read from the lowercased document rather than the sided expansion:
    # they name their landmarks explicitly (including `left_`/`right_` ones), so there is
    # no base name for the expansion to mirror and nothing for it to do.
    return LoadedComponent(
        landmarks=landmarks,
        segments=segments,
        landmark_groups={
            group_name: build_landmark_group(name=group_name, entry=entry)
            for group_name, entry in _grouping_section(
                component=lowercased, section="landmark_groups", component_name=name
            ).items()
        },
        landmark_connections={
            group_name: build_landmark_connection_group(name=group_name, entry=entry)
            for group_name, entry in _grouping_section(
                component=lowercased, section="landmark_connections", component_name=name
            ).items()
        },
    )


def _grouping_section(
    *, component: Mapping[str, object], section: str, component_name: str
) -> Mapping[str, object]:
    """One optional grouping section, refusing anything that is not a mapping."""
    node = component.get(section)
    if node is None:
        return {}
    if not isinstance(node, Mapping):
        raise ValueError(
            f"component {component_name!r}: `{section}` must be a mapping of group name -> "
            f"group, got {type(node).__name__}"
        )
    return node


def _build_landmark(*, name: str, entry: Mapping[str, object]) -> AnatomicalLandmark:
    """One expanded landmark entry as an `AnatomicalLandmark`."""
    unexpected_keys = set(entry) - LANDMARK_KEYS
    if unexpected_keys:
        raise ValueError(
            f"landmark {name!r}: unexpected keys {sorted(unexpected_keys)} - expected "
            f"{sorted(LANDMARK_KEYS)}"
        )
    local_position = _as_list(
        name=name, field_name="local_position", value=entry.get("local_position")
    )
    if len(local_position) != NUMBER_OF_SPATIAL_DIMENSIONS:
        raise ValueError(
            f"landmark {name!r}: `local_position` must have "
            f"{NUMBER_OF_SPATIAL_DIMENSIONS} values - got {local_position}"
        )
    x, y, z = (float(value) for value in local_position)
    return AnatomicalLandmark(
        name=name,
        anatomical_definition=str(entry.get("definition", "")),
        local_position=Point.from_xyz(x=x, y=y, z=z),
        segment=str(entry.get("reference_frame", "")),
        aliases=tuple(str(alias) for alias in entry.get("aliases", ())),
    )


def _build_segment(
    *,
    name: str,
    entry: Mapping[str, object],
    landmarks: Mapping[str, AnatomicalLandmark],
) -> RigidBodySegment:
    """One expanded segment entry as a `RigidBodySegment`, owning its own landmarks."""
    unexpected_keys = set(entry) - SEGMENT_KEYS
    if unexpected_keys:
        raise ValueError(
            f"segment {name!r}: unexpected keys {sorted(unexpected_keys)} - expected "
            f"{sorted(SEGMENT_KEYS)}"
        )
    reference_geometry = entry.get("reference_geometry")
    if reference_geometry is None:
        raise ValueError(
            f"segment {name!r} has no `reference_geometry`, so there is no origin and no "
            "axis to build it from. Give it at least an `origin` and one `type: exact` "
            "axis."
        )
    frame_definition = build_reference_frame_definition(
        segment_name=name, reference_geometry=reference_geometry
    )
    aliases = tuple(str(alias) for alias in entry.get("aliases", ()))
    # A landmark may name its segment by any of the segment's names. Matching only the
    # canonical one would leave an alias-referencing landmark owned by nothing, which
    # `SkeletonDefinition` would then have to catch as an error rather than a typo.
    owned_by_names = {name, *aliases}
    owned_landmarks = {
        landmark_name: landmark
        for landmark_name, landmark in landmarks.items()
        if landmark.segment in owned_by_names
    }
    # The origin may name a landmark this segment does NOT own - that is the shared
    # point where this segment meets its parent (e.g. the elbow belongs to the upper arm
    # but is the lower arm's origin). The primary and secondary must still be owned.
    owned_requirements = [frame_definition.primary_point_name]
    if frame_definition.secondary_point_name is not None:
        owned_requirements.append(frame_definition.secondary_point_name)
    missing_from_owned = [
        point_name for point_name in owned_requirements if point_name not in owned_landmarks
    ]
    if missing_from_owned:
        raise ValueError(
            f"segment {name!r}: `reference_geometry` names {missing_from_owned}, which "
            f"is not among the landmarks whose `reference_frame` is {name!r} "
            f"({sorted(owned_landmarks)})"
        )
    return RigidBodySegment(
        name=name,
        landmarks=owned_landmarks,
        frame_definition=frame_definition,
        aliases=aliases,
        anatomical_segment=entry.get("anatomical_segment"),
    )
