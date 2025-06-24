from pydantic import BaseModel, model_validator, ConfigDict
from typing import Dict, List
from skellymodels.utils.types import MarkerName, SegmentName, VirtualMarkerDefinition, SegmentConnection, SegmentCenterOfMassDefinition
from skellymodels.models.tracking_model_info import ModelInfo

class AnatomicalStructure(BaseModel):
    """
    A validated data structure representing the anatomical layout of a tracked subject.

    This model defines tracked markers, optional virtual markers, segment definitions,
    center of mass mappings, and joint hierarchies. It is typically constructed from a
    ModelInfo instance and used downstream for anatomical calculations (e.g. center of
    mass computation, rigid segment enforcement).

    Parameters
    ----------
    tracked_point_names : list of str
        Ordered list of marker names being output from the pose estimation tracker 
    virtual_markers_definitions : dict[str, VirtualMarkerDefinition], optional
        Definitions for computed markers based on weighted combinations of other markers.
    segment_connections : dict[SegmentName, SegmentConnection], optional
        Mapping of segments to their proximal/distal marker definitions.
    center_of_mass_definitions : dict[SegmentName, SegmentCenterOfMassDefinition], optional
        Definitions for computing segment-level center of mass.
    joint_hierarchy : dict[MarkerName, list[MarkerName]], optional
        Mapping of parent marker names to their immediate children, used for building
        skeletal graphs.
    """
    tracked_point_names: List[MarkerName]
    virtual_markers_definitions: Dict[str, VirtualMarkerDefinition]|None = None
    segment_connections: Dict[SegmentName, SegmentConnection]|None = None
    center_of_mass_definitions: Dict[SegmentName, SegmentCenterOfMassDefinition]|None = None
    joint_hierarchy: Dict[MarkerName, List[MarkerName]]|None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @model_validator(mode="after")
    def _cross_field_checks(self):
        """
        Cross-field validator to ensure all references between components are valid.

        Raises
        ------
        ValueError
            If virtual markers, segment connections, CoM definitions, or joint hierarchies
            reference undefined or invalid markers.
        """
        valid_marker_names = set(self.tracked_point_names)

        if self.virtual_markers_definitions:
            for virtual_marker_name, virtual_marker_values in self.virtual_markers_definitions.items():
                marker_names = virtual_marker_values.get("marker_names", [])
                marker_weights = virtual_marker_values.get("marker_weights", [])

                if len(marker_names) != len(marker_weights):
                    raise ValueError(
                        f"The number of marker names must match the number of marker weights for virtual marker{virtual_marker_name}. "
                        f"Currently there are {len(marker_names)} names and {len(marker_weights)} weights."
                    )

                # Check if all marker names are in our valid set
                invalid_markers = [name for name in marker_names if name not in valid_marker_names]
                if invalid_markers:
                    raise ValueError(
                        f"The marker(s) {invalid_markers} used to calculate virtual marker {virtual_marker_name} are not in tracked_points list"
                    )
                
                # Validate weights sum
                weight_sum = sum(marker_weights)
                if not 0.99 <= weight_sum <= 1.01:  # Allowing a tiny bit of floating-point leniency
                    raise ValueError(
                        f"Marker weights must sum to approximately 1 for virtual marker {virtual_marker_name}. Current sum is {weight_sum}."
                    )
                
                # Add this virtual marker to our valid set for below validations
                valid_marker_names.add(virtual_marker_name)
        
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
            virtual_markers_definitions = aspect_structure.virtual_marker_definitions,
            segment_connections = aspect_structure.segment_connections,
            center_of_mass_definitions = aspect_structure.center_of_mass_definitions,
            joint_hierarchy = aspect_structure.joint_hierarchy
        )

    @property
    def landmark_names(self) -> list[MarkerName]:
        """
        Returns a combined list of tracked and virtual marker names.

        Returns
        -------
        list of str
            Full list of marker names in order.
        """
        landmark_names = self.tracked_point_names.copy()
        if self.virtual_markers_definitions:
            landmark_names.extend(self.virtual_markers_definitions.keys())
        return landmark_names

    @property
    def virtual_marker_names(self) -> list[MarkerName]:
        """
        Returns only the virtual marker names.

        Returns
        -------
        list of str
            Names of virtual markers defined in the structure.
        """
        if not self.virtual_markers_definitions:
            return []
        return list(self.virtual_markers_definitions.keys())
    
    def __str__(self):
        virtual_markers = (
            f"{len(self.virtual_markers_definitions)} virtual markers"
            if self.virtual_markers_definitions else "No virtual markers"
        )
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
                f"  {virtual_markers}\n"
                f"  {segments}\n"
                f"  {com_definitions}\n"
                f"  {joint_hierarchy}")