"""Stage 4 of loading: turn a reference_geometry block into a ReferenceFrameDefinition.

The axis marked type: exact becomes the primary; type: approximate becomes the secondary.
negate: true selects the negative half of an axis. No approximate axis leaves the segment
underspecified.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from skellyforge.core.math.geometry.orthonormal_basis.reference_frame_definition import (
    ReferenceFrameDefinition,
)
from skellyforge.core.math.geometry.orthonormal_basis.spatial_axis import SpatialAxis

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
AXIS_ENTRY_KEYS: Final[frozenset[str]] = frozenset({"landmark", "type", "negate"})


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
