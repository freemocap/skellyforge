from typing import TypeAlias, List
from typing_extensions import TypedDict
MarkerName: TypeAlias = str
SegmentName: TypeAlias = str

class VirtualMarkerDefinition(TypedDict):
    marker_names: List[MarkerName]
    marker_weights: list[float]

class SegmentConnection(TypedDict):
    proximal: MarkerName
    distal: MarkerName

class SegmentCenterOfMassDefinition(TypedDict):
    segment_com_length : float
    segment_com_percentage: float

