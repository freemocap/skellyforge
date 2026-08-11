from pydantic import BaseModel, model_validator, ConfigDict
from typing import Dict, List
from skellyforge.skellymodels.types import (BoneKey,
                                      BoneLengthRatios,
                                      MarkerName,
                                      SegmentName,
                                      SegmentConnection,
                                      SegmentCenterOfMassDefinition
)
from skellyforge.skellymodels.models.tracking_model_info import ModelInfo

class AnatomicalStructure(BaseModel):
    """
    A validated data structure representing the anatomical layout of a tracked subject.

    This model defines tracked markers, segment definitions,
    center of mass mappings, and joint hierarchies. It is typically constructed from a
    ModelInfo instance and used downstream for anatomical calculations (e.g. center of
    mass computation, rigid segment enforcement).

    Parameters
    ----------
    tracked_point_names : list of str
        Ordered list of marker names being output from the pose estimation tracker
    segment_connections : dict[SegmentName, SegmentConnection], optional
        Mapping of segments to their proximal/distal marker definitions.
    center_of_mass_definitions : dict[SegmentName, SegmentCenterOfMassDefinition], optional
        Definitions for computing segment-level center of mass.
    joint_hierarchy : dict[MarkerName, list[MarkerName]], optional
        Mapping of parent marker names to their immediate children, used for building
        skeletal graphs.
    bone_length_ratios : dict[BoneKey, float], optional
        Bone-length-to-height ratios keyed by "parent->child". Used to seed
        initial segment length estimates before online adaptation.
    """
    tracked_point_names: List[MarkerName]
    segment_connections: Dict[SegmentName, SegmentConnection]|None = None
    center_of_mass_definitions: Dict[SegmentName, SegmentCenterOfMassDefinition]|None = None
    bone_length_ratios: BoneLengthRatios|None = None
    joint_hierarchy: Dict[MarkerName, List[MarkerName]]|None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @model_validator(mode="after")
    def _cross_field_checks(self):
        """
        Cross-field validator to ensure all references between components are valid.

        Raises
        ------
        ValueError
            If segment connections, CoM definitions, or joint hierarchies
            reference undefined or invalid markers.
        """
        valid_marker_names = set(self.tracked_point_names)

        if self.segment_connections:
            for segment_name, segment_connection in self.segment_connections.items():
                # Check if proximal and distal markers exist in marker_names
                if segment_connection["proximal"] not in valid_marker_names:
                    raise ValueError(
                        f"The proximal marker {segment_connection['proximal']} for {segment_name} is not in the list of markers."
                    )
                if segment_connection["distal"] not in valid_marker_names:
                    raise ValueError(
                        f"The distal marker {segment_connection['distal']} for {segment_name} is not in the list of markers."
                    )
            
        if self.center_of_mass_definitions:
            if not self.segment_connections:
                raise ValueError("Center of mass definitions require defined segment_connections")
            
            for segment_name, com_definition in self.center_of_mass_definitions.items():
                if segment_name not in self.segment_connections:
                    raise ValueError(f"Center of mass contains segment: {segment_name}, which is not in segment connections.")

        if self.joint_hierarchy:
            for parent, children in self.joint_hierarchy.items():
                
                if parent not in valid_marker_names:
                    raise ValueError(f"Joint hierarchy contains parent joint: {parent}, which not in list of markers.")

                bad_children = [child for child in children if child not in valid_marker_names]
                if bad_children:
                    raise ValueError(f"{parent} in joint hierarchy contains children not in the list of markers. Unknown markers: {bad_children}")
        return self

    @classmethod
    def from_model_info(cls, model_info:ModelInfo, aspect_name:str):
        """
        Factory method to create an AnatomicalStructure from a ModelInfo instance.

        Parameters
        ----------
        model_info : ModelInfo
            Parsed model configuration.
        aspect_name : str
            Name of the aspect to build (e.g. 'body', 'face').

        Returns
        -------
        AnatomicalStructure
            Fully validated structure for the given aspect.
        """
        aspect_structure = model_info.aspects[aspect_name]
        return cls(
            tracked_point_names = aspect_structure.tracked_points_names,
            segment_connections = aspect_structure.segment_connections,
            center_of_mass_definitions = aspect_structure.center_of_mass_definitions,
            bone_length_ratios = aspect_structure.bone_length_ratios,
            joint_hierarchy = aspect_structure.joint_hierarchy
        )

    @property
    def landmark_names(self) -> list[MarkerName]:
        """
        Returns the canonical landmark names. Every tracked point is a
        first-class landmark (the tracker→canonical mapping produces them),
        so this is simply the ordered tracked-point list.

        Returns
        -------
        list of str
            Full list of marker names in order.
        """
        return self.tracked_point_names.copy()

    def __str__(self):
        segments = (
            f"{len(self.segment_connections)} segments"
            if self.segment_connections else "No segment connections"
        )
        com_definitions = (
            f"{len(self.center_of_mass_definitions)} center of mass definitions"
            if self.center_of_mass_definitions else "No center of mass definitions"
        )
        joint_hierarchy = (
            f"{len(self.joint_hierarchy)} joint hierarchies"
            if self.joint_hierarchy else "No joint hierarchy"
        )
        return (f"  {len(self.tracked_point_names)} tracked points\n"
                f"  {segments}\n"
                f"  {com_definitions}\n"
                f"  {joint_hierarchy}")
