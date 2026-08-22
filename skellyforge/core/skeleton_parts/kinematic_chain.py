
from dataclasses import dataclass

from skellyforge.core.skeleton_parts.segment_linkage import SegmentLinkage

ChainNameString = str

@dataclass(frozen=True, slots=True)
class KinematicChain:
    name: ChainNameString
    linkages: list[SegmentLinkage]
