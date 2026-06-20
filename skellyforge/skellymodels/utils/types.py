from typing import TypeAlias, Dict, List
from typing_extensions import TypedDict

MarkerName: TypeAlias = str
SegmentName: TypeAlias = str
BoneKey: TypeAlias = str  # "parent->child" format

class VirtualMarkerDefinition(TypedDict):
    marker_names: List[MarkerName]
    marker_weights: list[float]

class SegmentConnection(TypedDict):
    proximal: MarkerName
    distal: MarkerName

class SegmentCenterOfMassDefinition(TypedDict):
    segment_com_length : float
    segment_com_percentage: float

# bone_length / total_standing_height, keyed by "parent->child"
BoneLengthRatios: TypeAlias = Dict[BoneKey, float]

