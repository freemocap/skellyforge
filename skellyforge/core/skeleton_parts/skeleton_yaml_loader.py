"""Loading skeleton component YAML into the landmark and segment types.

The YAML is authored for humans: names are SCREAMING_SNAKE so they scan as constants,
axes are keyed by the axis they name rather than by their role in Gram-Schmidt, and a
structure that exists on both sides of the body is written once with `sided: true`. None
of that survives loading - what comes out is `AnatomicalLandmark` and `RigidBodySegment`
objects with lowercase names.

Four stages, each a pure function that raises rather than repairing:

1. `resolve_includes` - a mapping whose only key is `$include` is replaced by the parsed
   contents of the file it names, resolved relative to the INCLUDING file's directory. A
   document with includes is exactly equivalent to the same document with the included
   files pasted in.
2. `lowercase_names` - one walk that lowercases every key and every string in the tree.
   Names, aliases, references and reference frames are all just strings, so they all come
   out canonical; the prose under `definition` is the one thing left as written.
3. `expand_sided_entries` - `sided: true` becomes `left_<name>` and `right_<name>`, with
   the right side's `local_position` mirrored across the sagittal plane (x -> -x). A sided
   entry is authored for the LEFT side, so its x must be >= 0 or the sides would silently
   swap.
4. `build_reference_frame_definition` - `reference_geometry` becomes a
   `ReferenceFrameDefinition`. The `type: exact` axis is the primary one, `type:
   approximate` the secondary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import yaml

from skellyforge.core.math.geometry.orthonormal_basis.reference_frame_definition import (
    ReferenceFrameDefinition,
)
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton_parts.anatomical_landmark import AnatomicalLandmark
from skellyforge.core.skeleton_parts.rigid_body_segment import RigidBodySegment
from skellyforge.type_overloads import LandmarkNameString, RigidBodySegmentName

INCLUDE_KEY: Final[str] = "$include"
SIDE_PREFIXES: Final[tuple[str, str]] = ("left", "right")
EXACT_AXIS_TYPE: Final[str] = "exact"
APPROXIMATE_AXIS_TYPE: Final[str] = "approximate"
AXIS_KEY_TO_SPATIAL_AXIS: Final[dict[str, SpatialAxis]] = {
    "x_axis": SpatialAxis.X,
    "y_axis": SpatialAxis.Y,
    "z_axis": SpatialAxis.Z,
}
NEGATED_AXIS: Final[dict[SpatialAxis, SpatialAxis]] = {
    SpatialAxis.X: SpatialAxis.NEGATIVE_X,
    SpatialAxis.Y: SpatialAxis.NEGATIVE_Y,
    SpatialAxis.Z: SpatialAxis.NEGATIVE_Z,
}
PROSE_KEYS: Final[frozenset[str]] = frozenset({"definition"})
LANDMARK_KEYS: Final[frozenset[str]] = frozenset(
    {"aliases", "definition", "reference_frame", "local_position", "sided"}
)
SEGMENT_KEYS: Final[frozenset[str]] = frozenset({"aliases", "reference_geometry", "sided"})
AXIS_ENTRY_KEYS: Final[frozenset[str]] = frozenset({"landmark", "type", "negate"})
NUMBER_OF_SPATIAL_DIMENSIONS: Final[int] = 3


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: includes
# ═══════════════════════════════════════════════════════════════════════


def resolve_includes(*, node: object, base_directory: Path, include_stack: tuple[Path, ...]) -> object:
    """Replace every `{$include: path}` mapping with the parsed contents of that file.

    Args:
        node: the parsed YAML node to walk.
        base_directory: directory that relative include paths are resolved against - the
            directory of the file this node came from, so an include means the same thing
            no matter who included it.
        include_stack: files currently being included, innermost last, for cycle detection.

    Returns:
        The node with every include replaced by the document it names.

    Raises:
        ValueError: an include mapping carries other keys, or the includes form a cycle.
        FileNotFoundError: an included path does not exist.
    """
    if isinstance(node, Mapping):
        if INCLUDE_KEY in node:
            if len(node) != 1:
                raise ValueError(
                    f"an `{INCLUDE_KEY}` mapping means 'paste this file here' and so may "
                    f"carry no other keys - got {sorted(node)}"
                )
            return _load_included_file(
                relative_path=str(node[INCLUDE_KEY]),
                base_directory=base_directory,
                include_stack=include_stack,
            )
        return {
            key: resolve_includes(
                node=value, base_directory=base_directory, include_stack=include_stack
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [
            resolve_includes(
                node=item, base_directory=base_directory, include_stack=include_stack
            )
            for item in node
        ]
    return node


def _load_included_file(
    *, relative_path: str, base_directory: Path, include_stack: tuple[Path, ...]
) -> object:
    """Parse one included file and recursively resolve the includes inside it."""
    included_path = (base_directory / relative_path).resolve()
    if included_path in include_stack:
        cycle = " -> ".join(path.name for path in (*include_stack, included_path))
        raise ValueError(f"`{INCLUDE_KEY}` cycle: {cycle}")
    if not included_path.is_file():
        raise FileNotFoundError(
            f"`{INCLUDE_KEY}: {relative_path}` (from {base_directory}) does not name a "
            f"file - looked at {included_path}"
        )
    return resolve_includes(
        node=yaml.safe_load(included_path.read_text(encoding="utf-8")),
        base_directory=included_path.parent,
        include_stack=(*include_stack, included_path),
    )


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: lowercasing
# ═══════════════════════════════════════════════════════════════════════


def lowercase_names(*, node: object) -> object:
    """Lowercase every key and every string in the document, except prose.

    The SCREAMING_SNAKE in the YAML is there to make the document scannable; the canonical
    runtime name is lowercase. Names, aliases, references, reference frames and the keys
    themselves are all just strings in the tree, so one walk lowercases the lot. The only
    thing that is not a name is the prose under `definition`, which is left as written.
    """
    if isinstance(node, str):
        return node.lower()
    if isinstance(node, Mapping):
        lowercased: dict[str, object] = {}
        for key, value in node.items():
            lowercased_key = str(key).lower()
            lowercased[lowercased_key] = (
                value if lowercased_key in PROSE_KEYS else lowercase_names(node=value)
            )
        return lowercased
    if isinstance(node, list):
        return [lowercase_names(node=item) for item in node]
    return node


def _as_list(*, name: str, field_name: str, value: object) -> list[object]:
    """A YAML sequence field as a list, refusing a bare scalar written by mistake."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            f"{name!r}: `{field_name}` must be a list - got {type(value).__name__} "
            f"({value!r})"
        )
    return list(value)


# ═══════════════════════════════════════════════════════════════════════
# Stage 3: sidedness
# ═══════════════════════════════════════════════════════════════════════


def expand_sided_entries(*, component: Mapping[str, object]) -> dict[str, dict[str, dict[str, object]]]:
    """Turn every sided entry into a `left_` and a `right_` one.

    A component may be sided as a whole (`sided: true` at the top of the file), or per
    entry. A sided entry is authored for the LEFT side; the right side is its mirror
    across the sagittal plane, so `local_position` has its x negated.

    References follow the same side: inside a sided entry, a bare reference to another
    sided name in this component resolves to the same side. An unsided entry has no side
    to inherit, so its references must already be explicit - which is why the pelvis names
    `left_hip_socket` outright.

    A sided landmark's x may be negative: mirror-symmetric structures (pelvis) sit on one
    side of the sagittal plane, but fan-shaped ones (hand, foot) legitimately span both,
    with the thumb and pinky on opposite sides of the hand's own midline.
    """
    component_is_sided = bool(component.get("sided", False))
    landmarks = _section(component=component, section="landmarks")
    segments = _section(component=component, section="segments")
    sided_names = {
        name
        for entries in (landmarks, segments)
        for name, entry in entries.items()
        if bool(entry.get("sided", component_is_sided))
    }

    expanded_landmarks: dict[str, dict[str, object]] = {}
    for name, entry in landmarks.items():
        if name not in sided_names:
            expanded_landmarks[name] = _sided_entry(
                entry=entry, side=None, sided_names=sided_names
            )
            continue
        for side in SIDE_PREFIXES:
            expanded_landmarks[f"{side}_{name}"] = _sided_entry(
                entry=entry, side=side, sided_names=sided_names
            )

    expanded_segments: dict[str, dict[str, object]] = {}
    for name, entry in segments.items():
        if name not in sided_names:
            expanded_segments[name] = _sided_entry(
                entry=entry, side=None, sided_names=sided_names
            )
            continue
        for side in SIDE_PREFIXES:
            expanded_segments[f"{side}_{name}"] = _sided_entry(
                entry=entry, side=side, sided_names=sided_names
            )

    return {"landmarks": expanded_landmarks, "segments": expanded_segments}


def _section(
    *, component: Mapping[str, object], section: str
) -> dict[str, Mapping[str, object]]:
    """One section of a component, with an absent or empty section reading as no entries."""
    entries = component.get(section) or {}
    if not isinstance(entries, Mapping):
        raise ValueError(
            f"`{section}` must be a mapping of name -> entry - got {type(entries).__name__}"
        )
    return {
        name: {} if entry is None else entry
        for name, entry in entries.items()
    }



def _sided_entry(
    *, entry: Mapping[str, object], side: str | None, sided_names: frozenset[str] | set[str]
) -> dict[str, object]:
    """One entry resolved for one side, or as-is when it has no side."""
    resolved = {key: value for key, value in entry.items() if key != "sided"}
    if side is None:
        return resolved
    if side == SIDE_PREFIXES[1] and "local_position" in resolved:
        x, y, z = (float(value) for value in resolved["local_position"])
        resolved["local_position"] = [-x, y, z]
    if "aliases" in resolved:
        resolved["aliases"] = [f"{side}_{alias}" for alias in resolved["aliases"]]
    if "reference_frame" in resolved:
        resolved["reference_frame"] = _sided_reference(
            reference=str(resolved["reference_frame"]), side=side, sided_names=sided_names
        )
    reference_geometry = resolved.get("reference_geometry")
    if reference_geometry is not None:
        resolved["reference_geometry"] = _sided_reference_geometry(
            reference_geometry=reference_geometry, side=side, sided_names=sided_names
        )
    return resolved


def _sided_reference(
    *, reference: str, side: str, sided_names: frozenset[str] | set[str]
) -> str:
    """A bare reference to a sided name, resolved to this side; anything else untouched."""
    return f"{side}_{reference}" if reference in sided_names else reference


def _sided_reference_geometry(
    *, reference_geometry: Mapping[str, object], side: str, sided_names: frozenset[str] | set[str]
) -> dict[str, object]:
    """A `reference_geometry` with its landmark references resolved to one side."""
    resolved: dict[str, object] = {}
    for key, value in reference_geometry.items():
        if key == "origin":
            resolved["origin"] = _sided_reference(
                reference=str(value), side=side, sided_names=sided_names
            )
            continue
        axis_entry = dict(value)
        axis_entry["landmark"] = _sided_reference(
            reference=str(axis_entry["landmark"]), side=side, sided_names=sided_names
        )
        resolved[key] = axis_entry
    return resolved


# ═══════════════════════════════════════════════════════════════════════
# Stage 4: reference geometry -> ReferenceFrameDefinition
# ═══════════════════════════════════════════════════════════════════════


def build_reference_frame_definition(
    *, segment_name: str, reference_geometry: Mapping[str, object]
) -> ReferenceFrameDefinition:
    """Turn one `reference_geometry` block into a `ReferenceFrameDefinition`.

    The axis marked `type: exact` becomes the primary axis - the one the solver points
    exactly at its landmark - and `type: approximate` becomes the secondary axis, which
    Gram-Schmidt orthogonalizes against the primary. `negate: true` selects the negative
    half of that axis. A block with no approximate axis yields an underspecified
    definition, which is a segment whose roll is not yet pinned down.

    Raises:
        ValueError: no origin, no exact axis, an unknown axis key or type, or more than
            one axis of either type.
    """
    origin_point_name = reference_geometry.get("origin")
    if not origin_point_name:
        raise ValueError(f"segment {segment_name!r}: `reference_geometry` needs an `origin`")

    axes_by_type: dict[str, list[tuple[SpatialAxis, str]]] = {
        EXACT_AXIS_TYPE: [],
        APPROXIMATE_AXIS_TYPE: [],
    }
    for axis_key, axis_entry in reference_geometry.items():
        if axis_key == "origin":
            continue
        spatial_axis = AXIS_KEY_TO_SPATIAL_AXIS.get(axis_key)
        if spatial_axis is None:
            raise ValueError(
                f"segment {segment_name!r}: unknown `reference_geometry` key "
                f"{axis_key!r} - expected `origin` or one of "
                f"{sorted(AXIS_KEY_TO_SPATIAL_AXIS)}"
            )
        unexpected_keys = set(axis_entry) - AXIS_ENTRY_KEYS
        if unexpected_keys:
            raise ValueError(
                f"segment {segment_name!r}: `{axis_key}` has unexpected keys "
                f"{sorted(unexpected_keys)} - expected {sorted(AXIS_ENTRY_KEYS)}"
            )
        axis_type = axis_entry.get("type")
        if axis_type not in axes_by_type:
            raise ValueError(
                f"segment {segment_name!r}: `{axis_key}.type` must be "
                f"{EXACT_AXIS_TYPE!r} or {APPROXIMATE_AXIS_TYPE!r} - got {axis_type!r}"
            )
        if bool(axis_entry.get("negate", False)):
            spatial_axis = NEGATED_AXIS[spatial_axis]
        axes_by_type[axis_type].append((spatial_axis, str(axis_entry["landmark"])))

    for axis_type, found in axes_by_type.items():
        if len(found) > 1:
            raise ValueError(
                f"segment {segment_name!r}: exactly one axis may be {axis_type!r} - got "
                f"{[axis.name for axis, _ in found]}"
            )
    if not axes_by_type[EXACT_AXIS_TYPE]:
        raise ValueError(
            f"segment {segment_name!r}: `reference_geometry` needs one axis marked "
            f"`type: {EXACT_AXIS_TYPE}` to point exactly at its landmark"
        )

    primary_axis, primary_point_name = axes_by_type[EXACT_AXIS_TYPE][0]
    approximate = axes_by_type[APPROXIMATE_AXIS_TYPE]
    if not approximate:
        return ReferenceFrameDefinition(
            origin_point_name=str(origin_point_name),
            primary_axis=primary_axis,
            primary_point_name=primary_point_name,
        )
    secondary_axis, secondary_point_name = approximate[0]
    return ReferenceFrameDefinition(
        origin_point_name=str(origin_point_name),
        primary_axis=primary_axis,
        primary_point_name=primary_point_name,
        secondary_axis=secondary_axis,
        secondary_point_name=secondary_point_name,
    )


# ═══════════════════════════════════════════════════════════════════════
# Putting the stages together
# ═══════════════════════════════════════════════════════════════════════


def load_component(
    *, path: Path, name: str
) -> tuple[dict[LandmarkNameString, AnatomicalLandmark], dict[RigidBodySegmentName, RigidBodySegment]]:
    """Load one component YAML file into landmark and segment objects.

    Returns the two dicts rather than a component object, because a component is not a
    thing that outlives loading - it is a file, and what a file contributes is landmarks
    and segments. Cross-component checks belong to `SkeletonDefinition`, which is the
    first place that can see every component at once: a landmark here may legitimately
    name a `reference_frame` that lives in another file.

    Args:
        path: the component `.yaml` file.
        name: the name this component is known by in the skeleton, used in errors.

    Returns:
        This component's landmarks and segments, expanded and lowercased.
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


def build_component(
    *, component: Mapping[str, object], name: str
) -> tuple[dict[LandmarkNameString, AnatomicalLandmark], dict[RigidBodySegmentName, RigidBodySegment]]:
    """Run stages 2-4 over one already-include-resolved component document."""
    lowercased = lowercase_names(node=component)
    if not isinstance(lowercased, Mapping):
        raise ValueError(f"component {name!r} must be a mapping, got {type(component).__name__}")
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
    return landmarks, segments


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
    owned_landmarks = {
        landmark_name: landmark
        for landmark_name, landmark in landmarks.items()
        if landmark.segment == name
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
        aliases=tuple(str(alias) for alias in entry.get("aliases", ())),
    )
