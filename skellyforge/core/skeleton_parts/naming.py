"""Name and alias rules shared by landmarks and segments.

Aliases exist because incoming motion capture data does not agree on names: the same
point may arrive as `left_shoulder`, `LSHO`, or `shoulder.L`. A definition declares the
alternate names it answers to, and `landmark_name_resolver` folds them all down to one
canonical name at the boundary - once, at setup, never in the per-frame path.

Uniqueness is checked at whatever scope can actually see it. These functions cover the
scope of a single definition; a segment checks across its own landmarks; and true global
uniqueness is enforced by `LandmarkNameResolver`, which is the first thing that knows
what "global" contains.
"""

from __future__ import annotations

from collections.abc import Sequence

from skellyforge.type_overloads import RigidBodySegmentName


def raise_unless_snake_case_segment_name(*, name: RigidBodySegmentName) -> None:
    """Segment names are snake_case.

    Sidedness is carried by a `left_`/`right_` PREFIX, which the YAML loader adds, so a
    sided name is already snake_case and needs no special case here.
    """
    if not name or name != name.lower() or " " in name:
        raise ValueError(f"segment name must be snake_case, got {name!r}")


def raise_unless_aliases_are_valid(*, name: str, aliases: Sequence[str]) -> None:
    """Aliases must be non-empty, distinct from each other, and distinct from `name`."""
    empty_alias_positions = [index for index, alias in enumerate(aliases) if not alias]
    if empty_alias_positions:
        raise ValueError(
            f"{name!r}: aliases must be non-empty strings - empty at positions "
            f"{empty_alias_positions}"
        )
    if len(set(aliases)) != len(aliases):
        duplicates = sorted({alias for alias in aliases if list(aliases).count(alias) > 1})
        raise ValueError(f"{name!r}: aliases must be distinct - repeated: {duplicates}")
    if name in aliases:
        raise ValueError(f"{name!r}: an alias must differ from the canonical name itself")
