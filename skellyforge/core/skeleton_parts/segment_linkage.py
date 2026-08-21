from dataclasses import dataclass

LinkageNameString = str


@dataclass(frozen=True, slots=True)
class SegmentLinkage:
    name: LinkageNameString
