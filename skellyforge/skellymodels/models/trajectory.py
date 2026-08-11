from pydantic import BaseModel, ConfigDict, model_validator
import numpy as np
import pandas as pd
from typing import Dict, List
from skellyforge.skellymodels.types import MarkerName, SegmentName, SegmentConnection
from skellyforge.skellymodels.models.anatomical_structure import AnatomicalStructure
import warnings
class Trajectory(BaseModel):
    """
    A Trajectory stores 3D data for a given anatomical component
    (e.g., body, face, hand) as a NumPy array of shape (num_frames, num_markers, 3).

    It includes utilities for dictionary and DataFrame views of the data.

    Parameters
    ----------
    name : str
        Identifier for the trajectory (e.g., "3d_xyz", "rigid_3d_xyz").
    array : np.ndarray, shape (num_frames, num_markers, 3)
        Main data array containing 3D positions for all markers across frames.
    landmark_names : list of str
        Ordered list of marker names corresponding to the columns in the array.

    Notes
    -----
    Validation ensures that the number of landmark names matches the number
    of markers in the second dimension of the array, and that the last dimension
    is exactly 3 (for xyz).
    """
    name: str
    array: np.ndarray
    landmark_names: List[MarkerName]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_tracked_points_data(cls, 
                                 name:str,
                                tracked_points_array:np.ndarray, 
                                anatomical_structure:AnatomicalStructure) -> "Trajectory":
        """
        Factory method to create a Trajectory from tracked points and
        a corresponding AnatomicalStructure.

        Every tracked point is a first-class canonical landmark. Computed
        landmarks (e.g. neck_center) are produced upstream by the
        tracker→canonical mapping, so the array maps 1:1 onto the anatomical
        structure's landmarks.

        Parameters
        ----------
        name : str
            Name of the trajectory (e.g. '3d_xyz').
        tracked_points_array : ndarray of shape (num_frames, num_markers, 3)
            Array of canonical landmark positions.
        anatomical_structure : AnatomicalStructure
            Provides the landmark names.

        Returns
        -------
        Trajectory
        """
        return cls(name = name,
                   array = tracked_points_array,
                   landmark_names = anatomical_structure.tracked_point_names.copy())
    
    @model_validator(mode="after")
    def _check_shape(self):
        """
        Post-initialization shape validation.

        Raises
        ------
        ValueError
            If shape mismatches between landmark_names and the data array,
            or if the last dimension is not 3.
        """
        if self.array.shape[1] != len(self.landmark_names):
            raise ValueError(
                f"{self.name}: data has {self.array.shape[1]} columns but "
                f"{len(self.landmark_names)} marker names supplied"
            )
        if self.array.shape[2] != 3:
            raise ValueError(f"{self.name}: last dim must be 3 (xyz)")
        return self

    @property
    def as_array(self) -> np.ndarray:
        """
        Returns NumPy array 

        Returns
        ------
        np.ndarray
            Shape (num_frames, num_markers, 3)
        """
        return self.array
    
    @property
    def as_dict(self) -> dict[MarkerName, np.ndarray]:
        """
        Returns the marker data as a dictionary mapping names to arrays.

        Returns
        -------
        dict
            Keys are marker names, values are arrays of shape (num_frames, 3)
        """
        return {n: self.array[:, i, :]
            for i, n in enumerate(self.landmark_names)
            }

    @property
    def as_dataframe(self) -> pd.DataFrame:
        """
        Returns the data in tidy long-form format.

        Returns
        -------
        DataFrame
            Columns: ["frame", "keypoint", "x", "y", "z"]
        """

        df = pd.DataFrame(
            self.array.reshape(self.num_frames*self.num_markers,3),
            columns=['x','y','z'],
        )
        
        df['frame'] = np.repeat(np.arange(self.num_frames),self.num_markers)
        df['keypoint'] = np.tile(self.landmark_names, self.num_frames)
        return df[['frame', 'keypoint', 'x', 'y', 'z']]
         
    @property
    def num_frames(self) -> int:
        """Total number of frames in the trajectory."""
        return self.array.shape[0]
    
    @property
    def num_markers(self) -> int:
        """Total number of markers (columns) in the trajectory."""
        return self.array.shape[1]
    
    def segment_data(self, segment_connections:Dict[SegmentName, SegmentConnection]) -> Dict[SegmentName, Dict[str, np.ndarray]]:
        """
        Returns a dictionary of proximal/distal points for each segment.

        Parameters
        ----------
        segment_connections : dict
            Each entry should define 'proximal' and 'distal' marker names.

        Returns
        -------
        dict
            Keys are segment names; values are dictionaries with keys
            'proximal' and 'distal', each mapping to an array of shape (F, 3).
        """
        if not segment_connections:
            return {}
        d = self.as_dict
        segment_positions = {}
        for name, connection in segment_connections.items():
            proximal = d.get(connection["proximal"])
            distal = d.get(connection["distal"])

            segment_positions.update({name: {'proximal': proximal, 'distal': distal}})

        return segment_positions
    
    @property
    def data(self): #this is for me to figure out where I use this in my validation pipeline. Will remove this after that.
        warnings.warn(".data is deprecated - use .as_dict for the same output",
                      DeprecationWarning,
                      stacklevel=2)  # TODO: elsewhere this is used for the numpy array, but here its a dictionary 
        return self.as_dict
    
    def __str__(self) -> str:
        return f"Trajectory with {self.num_frames} frames and {len(self.landmark_names)} markers"

    def __repr__(self) -> str:
        return self.__str__()
