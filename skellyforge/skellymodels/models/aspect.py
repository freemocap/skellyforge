from pydantic import BaseModel, Field, model_validator, ConfigDict
from skellymodels.models.anatomical_structure import AnatomicalStructure
from skellymodels.models.tracking_model_info import ModelInfo
from skellymodels.models.anatomical_structure import AnatomicalStructure
from skellymodels.models.error import Error
from skellymodels.models.trajectory import Trajectory
from skellymodels.utils.types import  SegmentName
from typing import Dict, Any, Optional
import numpy as np
from enum import Enum

class TrajectoryNames(Enum):
    """Enum for common trajectory names used in aspects."""
    XYZ = '3d_xyz'
    RIGID_XYZ = 'rigid_3d_xyz'
    TOTAL_BODY_COM = 'total_body_center_of_mass'
    SEGMENT_COM = 'segment_center_of_mass'


class Aspect(BaseModel):
    """
    A modular unit representing a tracked anatomical region (e.g., body, face, hand).

    Each `Aspect` contains:
    - an `AnatomicalStructure` definition (which provides the scaffold),
    - a dictionary of `Trajectory` objects for time-series motion data,
    - optional `Error` data (e.g., reprojection error),
    - metadata for labeling or downstream tools.

    Parameters
    ----------
    name : str
        Name of the aspect (e.g. "body", "face", "left_hand").
    anatomical_structure : AnatomicalStructure
        Model defining marker layout, virtual markers, segments, and joint structure.
    trajectories : dict[str, Trajectory], optional
        Dictionary mapping trajectory names (e.g. "3d_xyz") to trajectory data.
    reprojection_error : Error, optional
        2D array of reprojection errors (frames x markers) from the original tracker.
    metadata : dict[str, Any], optional
        Arbitrary metadata attached to this aspect (e.g. tracker name, units).

    Notes
    -----
    Typical usage involves initializing from a `ModelInfo` with `.from_model_info()`,
    followed by calls to `.add_tracked_points()` and optionally `.add_reprojection_error()`.
    """
    name: str
    anatomical_structure: AnatomicalStructure
    trajectories: dict[str,Trajectory] = Field(default_factory=dict)
    reprojection_error: Optional[Error] = None
    metadata: dict[str,Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True, #to allow for numpy in the model
        validate_assignment=True) #validates after the model is changed
    
    @classmethod
    def from_model_info(cls, name: str, model_info: ModelInfo, metadata: Optional[Dict[str, Any]] = None) -> "Aspect":
        """
        Creates an Aspect from a `ModelInfo` configuration.

        Parameters
        ----------
        name : str
            Name of the aspect (e.g. "body", "face").
        model_info : ModelInfo
            Full configuration containing marker and segment definitions.
        metadata : dict, optional
            Extra data to attach (e.g. tracker_type).

        Returns
        -------
        Aspect
            An initialized Aspect with structure loaded.
        """
        anatomical_structure = AnatomicalStructure.from_model_info(
            model_info=model_info,
            aspect_name=name
        )
        return cls(name=name, anatomical_structure=anatomical_structure, metadata=metadata)

    def add_trajectory(self, 
                       dict_of_trajectories: Dict[str, Trajectory]):
        """
        Adds one or more named trajectory objects to this aspect.

        Parameters
        ----------
        dict_of_trajectories : dict
            Dictionary mapping names to Trajectory instances.

        Raises
        ------
        TypeError
            If any value is not a `Trajectory`.
        """
        for name, trajectory in dict_of_trajectories.items():
            if not isinstance(trajectory, Trajectory):
                raise TypeError(f"Expected Trajectory instance for {name}, got {type(trajectory)}")
            self.trajectories.update({name: trajectory})
            #add check for whether the trajectory name is in the expected list (and make an expected enum list)

    def add_tracked_points(self, tracked_points:np.ndarray):
        """
        Ingests a raw `(num_frames, num_markers, 3)` array of marker data and adds it as a
        trajectory using the anatomical structure.

        Parameters
        ----------
        tracked_points : np.ndarray
            Tracked XYZ marker array, excluding virtual markers.
        """

        if self.anatomical_structure is None:
            raise ValueError("Anatomical structure and tracked point names are required to ingest tracker data.")

        trajectory = Trajectory.from_tracked_points_data(
            name = TrajectoryNames.XYZ.value,
            tracked_points_array=tracked_points,
            anatomical_structure=self.anatomical_structure
        )

        self.add_trajectory({TrajectoryNames.XYZ.value: trajectory})

    def add_reprojection_error(self, reprojection_error_data: np.ndarray):
        """
        Attaches a 2D array of reprojection error values.

        Parameters
        ----------
        reprojection_error_data : np.ndarray of shape (num_frames, num_markers)
            Error values per marker per frame.

        Raises
        ------
        ValueError
            If the array shape does not match trajectory expectations.
        """
        if self.trajectories.get(TrajectoryNames.XYZ.value) is not None:
            if reprojection_error_data.shape[0] != self.trajectories[TrajectoryNames.XYZ.value].num_frames:
                raise ValueError(
                    "First dimension of reprojection error must match the number of frames in the trajectory.")
            if reprojection_error_data.shape[1] != len(self.anatomical_structure.tracked_point_names):
                raise ValueError(
                    "Second dimension of reprojection error must match the number of landmark names in the trajectory.")

        self.reprojection_error = Error(name='reprojection_error',
                                        data=reprojection_error_data,
                                        marker_names=self.anatomical_structure.tracked_point_names)

    def add_metadata(self, metadata: Dict[str, Any]):
        """
        Adds additional metadata fields to this aspect.

        Parameters
        ----------
        metadata : dict
            Dictionary of key–value pairs to merge into existing metadata.
        """
        self.metadata.update(metadata)

    @property
    def xyz(self) -> Optional[Trajectory]:
        """
        Returns the 3D XYZ trajectory (raw marker positions), if present.

        Returns
        -------
        Trajectory or None
        """
        return self.trajectories.get(TrajectoryNames.XYZ.value)
    
    @property
    def rigid_xyz(self) -> Optional[Trajectory]:
        """
        Returns the rigid-body 3D trajectory, if present.

        Returns
        -------
        Trajectory or None
        """        
        return self.trajectories.get(TrajectoryNames.RIGID_XYZ.value)

    @property
    def total_body_com(self) -> Optional[Trajectory]:
        """
        Returns the trajectory for total-body center of mass, if present.

        Returns
        -------
        Trajectory or None
        """
        return self.trajectories.get(TrajectoryNames.TOTAL_BODY_COM.value)
    
    @property
    def segment_com(self) -> Optional[Dict[SegmentName, Trajectory]]:
        """
        Returns the dictionary of segment center of mass trajectories, if present.

        Returns
        -------
        dict or None
        """        
        return self.trajectories.get(TrajectoryNames.SEGMENT_COM.value)

    def __str__(self):
        anatomical_info = (
            str(self.anatomical_structure) if self.anatomical_structure else "No anatomical structure"
        )
        trajectory_info = (
            f"{len(self.trajectories)} trajectories: {list(self.trajectories.keys())}"
            if self.trajectories else "No trajectories"
        )
        error_info = (
            f"Has reprojection error"
            if self.reprojection_error else "No reprojection error"
        )
        metadata_info = (
            f": {self.metadata}"
            if self.metadata else "No metadata"
        )
        return (f"Aspect: {self.name}\n"
                f"  Anatomical Structure:\n{anatomical_info}\n"
                f"  Trajectories: {trajectory_info}\n"
                f"  Error: {error_info}\n"
                f"  Metadata: {metadata_info}\n\n")

    def __repr__(self):
        return self.__str__()

