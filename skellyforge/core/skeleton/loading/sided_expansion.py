"""Stage 3 of loading: expand sided entries into a left_ and a right_ one.

A component may be sided as a whole or per entry; a sided entry is authored for the LEFT
side and the right side is its mirror across the sagittal plane (x negated, and any x-axis
declaration in its reference_geometry negated to match).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Final

SIDE_PREFIXES: Final[tuple[str, str]] = ("left", "right")
MIRRORED_AXIS_KEY: Final[str] = "x_axis"


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
    """A `reference_geometry` resolved to one side, with the x axis flipped on the right.

    Mirroring the coordinates is only half of mirroring a component. The x coordinate of
    every landmark is negated, so an axis DECLARED along x would come out pointing the
    opposite way in the world on the right side from the way it points on the left - which
    would make local `+y` anterior on one side of the body and posterior on the other, and
    a joint angle mean different things left and right.

    Negating x-axis declarations on the right side fixes that: both sides end up with
    local `+x` toward the subject's right, `+y` forward and `+z` up, matching the Blender
    convention the whole package is authored in, and both stay right-handed. The cost is
    that local `+x` is medial on the left and lateral on the right - unavoidable, since a
    right-handed triad cannot mirror all three axes at once - so it is anterior and distal
    that correspond across the body, not lateral.

    The named landmark still lies EXACTLY on its declared signed axis; the declaration
    just becomes the negative half of that axis.
    """
    mirror_the_x_axis = side == SIDE_PREFIXES[1]
    resolved: dict[str, object] = {}
    for key, value in reference_geometry.items():
        if key == "origin":
            resolved["origin"] = _sided_reference(
                reference=str(value), side=side, sided_names=sided_names
            )
            continue
        if not isinstance(value, Mapping):
            raise ValueError(
                f"`reference_geometry.{key}` must be a mapping with a `landmark` and a "
                f"`type` - got {type(value).__name__} ({value!r})"
            )
        axis_entry = dict(value)
        axis_entry["landmark"] = _sided_reference(
            reference=str(axis_entry["landmark"]), side=side, sided_names=sided_names
        )
        if mirror_the_x_axis and key == MIRRORED_AXIS_KEY:
            axis_entry["negate"] = not bool(axis_entry.get("negate", False))
        resolved[key] = axis_entry
    return resolved
