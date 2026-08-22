"""A sequence of linkages moving together. The chain layer, not yet built out."""

from __future__ import annotations

from dataclasses import dataclass

from skellyforge.core.skeleton_parts.segment_linkage import SegmentLinkage
from skellyforge.type_overloads import ChainNameString


@dataclass(frozen=True, slots=True, eq=False)
class KinematicChain:
    """Linkages that move as one chain - an arm, a leg, a finger.

    A placeholder for the ontology's chain layer, above `SegmentLinkage` and below the
    skeleton. Nothing constructs one yet.

    Attributes:
        name: what this chain is called.
        linkages: the chain's linkages, proximal first. A tuple rather than a list: a
            frozen dataclass holding a mutable field is frozen in name only.
    """

    name: ChainNameString
    linkages: tuple[SegmentLinkage, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("chain name must be non-empty")
        if not self.linkages:
            raise ValueError(f"chain {self.name!r} needs at least one linkage")
