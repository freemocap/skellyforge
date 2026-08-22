"""Two segments joined at the landmark they share. The linkage layer, not yet built out."""

from __future__ import annotations

from dataclasses import dataclass

from skellyforge.core.skeleton_parts.rigid_body_segment import RigidBodySegment
from skellyforge.type_overloads import LinkageNameString


@dataclass(frozen=True, slots=True, eq=False)
class SegmentLinkage:
    """Two segments joined at a shared landmark; the parent precedes the child.

    A placeholder for the ontology's linkage layer. Nothing constructs one yet - the
    hierarchy currently lives in the rest pose's `parent` / `connect_at` fields, and
    reconciling the two is what building this layer means.

    Attributes:
        name: what this joint is called.
        segments: the joined segments, parent first. A tuple, because a linkage is a value
            rather than a collection to append to.
    """

    name: LinkageNameString
    segments: tuple[RigidBodySegment, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("linkage name must be non-empty")
        if len(self.segments) < 2:
            raise ValueError(
                f"linkage {self.name!r} joins two segments - got "
                f"{[segment.name for segment in self.segments]}"
            )
