from pydantic import BaseModel, ConfigDict, model_validator
import numpy as np
import pandas as pd
from typing import List
from skellymodels.utils.types import MarkerName

class Error(BaseModel):
    """
    A container for time-series error data associated with tracked 3D markers.

    This class holds error values (e.g., reprojection error) for each marker across time,
    and provides utilities to access that data as an array, dictionary, or tidy dataframe.

    Attributes:
        name (str): Identifier for this error dataset.
        array (np.ndarray): A 2D array of shape (num_frames, num_markers) containing error values.
        marker_names (List[MarkerName]): The ordered list of marker names corresponding to columns in `array`.
    """
    name: str
    array: np.ndarray
    marker_names: List[MarkerName]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_data(self):
        if self.array.shape[1] != len(self.marker_names):
            raise ValueError(
                f"Error data must have the same number of markers as input name list. Data has {self.array.shape[1]} markers and list has {len(self.marker_names)} markers.")
        return self
    
    @property
    def as_array(self) -> np.ndarray:
        return self.array
    
    @property
    def as_dict(self) -> dict:
        return {marker_name: self.array[:, i] for i, marker_name in enumerate(self.marker_names)}
    
    @property
    def as_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            self.array.reshape(self.num_frames*self.num_markers,1),
            columns = ['error']
        )

        df['frame'] = np.repeat(np.arange(self.num_frames),self.num_markers)
        df['keypoint'] = np.tile(self.marker_names, self.num_frames)

        return df[['frame', 'keypoint', 'error']]
         
    @property
    def num_frames(self) -> int:
        """Total number of frames in the trajectory."""
        return self.array.shape[0]

    @property
    def num_markers(self) -> int:
        """Total number of markers (columns) in the trajectory."""
        return self.array.shape[1]
    
    def get_frame(self, frame_number: int):
        return {marker_name: trajectory[frame_number] for marker_name, trajectory in self.as_dict.items()}

    def __str__(self) -> str:
        return f"Error with {self.num_frames} frames and {len(self.marker_names)} markers"

    def __repr__(self) -> str:
        return self.__str__()
