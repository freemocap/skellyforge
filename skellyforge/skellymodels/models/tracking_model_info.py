from pathlib import Path
from typing import Dict, List
import yaml
from pydantic import BaseModel
from skellymodels.utils.types import MarkerName, SegmentName, VirtualMarkerDefinition, SegmentConnection, SegmentCenterOfMassDefinition

class AspectInfo(BaseModel):
    tracked_points_names: List[MarkerName]
    num_tracked_points: int
    virtual_marker_definitions: Dict[str, VirtualMarkerDefinition]|None = None
    segment_connections: Dict[SegmentName, SegmentConnection]|None = None
    center_of_mass_definitions: Dict[SegmentName, SegmentCenterOfMassDefinition]|None = None
    joint_hierarchy: Dict[MarkerName, List[MarkerName]]|None = None

class ModelInfo(BaseModel):
    name: str
    tracker_name: str
    aspects: Dict[str, AspectInfo]
    order: List[str]

    @classmethod
    def from_config_path(cls, config_path: Path|str):
        """
        Create a ModelInfo instance from a configuration file path.
        """
        config_path = Path(config_path)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        aspects: dict[str, AspectInfo] = cls.get_aspects_from_config(config=config)

        return cls(
            name = config['name'],
            tracker_name = config['tracker_name'],
            aspects = aspects,
            order = config['order'],
        )

    @classmethod
    def from_model_dict(cls, model_dict: dict):
        """
        Create a ModelInfo instance from a dictionary with the required data (which is saved into the Parquet file)
        """
        if not all(key in model_dict for key in ['name', 'tracker_name', 'aspects', 'order']):
            raise ValueError("Dictionary must contain 'name', 'tracker_name', 'aspects', and 'order' keys.")
        return cls(
            name = model_dict['name'],
            tracker_name = model_dict['tracker_name'],
            aspects = model_dict['aspects'],
            order = model_dict['order'],
        )

    @staticmethod
    def get_aspects_from_config(config:dict):
        aspects = {}
        for aspect_name, aspect_info in config['aspects'].items():
            # Handle different ways of naming tracked points (names provided in a list vs. generating identifiers)
            if aspect_info['tracked_points']['type'] == 'list':
                tracked_points_names = aspect_info['tracked_points']['names']
            elif aspect_info['tracked_points']['type'] == 'generated':
                naming_convention = aspect_info['tracked_points']['names']['convention']
                count = aspect_info['tracked_points']['names']['count']
                tracked_points_names = [naming_convention.format(i) for i in range(count)]
            
            aspects[aspect_name] = AspectInfo(
                tracked_points_names = tracked_points_names,
                num_tracked_points= len(tracked_points_names),
                virtual_marker_definitions = aspect_info.get('virtual_marker_definitions'),
                segment_connections = aspect_info.get('segment_connections'),
                center_of_mass_definitions = aspect_info.get('center_of_mass_definitions'),
                joint_hierarchy = aspect_info.get('joint_hierarchy')
            )
        return aspects
    
    @property
    def tracked_point_slices(self) -> Dict[str, slice]:
        return self._build_slices(include_virtuals=False)
    
    @property
    def tracked_point_names(self) -> List[MarkerName]:
        return [tp for aspect in self.aspects.values() for tp in aspect.tracked_points_names]
    
    @property
    def num_tracked_points(self) -> int:
        return sum([aspect.num_tracked_points for aspect in self.aspects.values()])

    def _build_slices(self, *, include_virtuals:bool) -> Dict[str, slice]:
        """
        Build slices for each aspect based on the configuration and whether to include virtual markers.
        """
        slices = {}
        current_marker = 0
        for aspect in self.order:
            try:
                num_tracked_points = self.aspects[aspect].num_tracked_points
                num_virtual_markers = len(self.aspects[aspect].virtual_marker_definitions) if self.aspects[aspect].virtual_marker_definitions else 0
                num_landmarks = num_tracked_points + (num_virtual_markers if include_virtuals else 0)
                slices[aspect] = slice(current_marker, current_marker + num_landmarks)
                current_marker += num_landmarks
            except KeyError:
                raise KeyError(f"Aspect '{aspect}' is included in the aspect order of the YAML, but no configuration was found for it. Available aspects: {list(aspects.keys())}")
        return slices


def MediapipeModelInfo():
    return ModelInfo.from_config_path(config_path = Path(__file__).parents[1]/'tracker_info'/'mediapipe_model_info.yaml')

def RTMPoseModelInfo():
    return ModelInfo.from_config_path(config_path = Path(__file__).parents[1]/'tracker_info'/'rtmpose_model_info.yaml')

f = 2
