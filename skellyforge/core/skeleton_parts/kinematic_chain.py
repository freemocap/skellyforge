
from dataclasses import dataclass

from skellyforge.skeleton_parts.segment_linkage import SegmentLinkage

ChainNameString = str

@dataclass(frozen=True, slots=True)
class KinematicChain:
    name: ChainNameString
    linkages: list[SegmentLinkage]
