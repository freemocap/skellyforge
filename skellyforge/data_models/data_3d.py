from pydantic import BaseModel
import numpy as np


class Observation3d(BaseModel):
    # TODO: since this is calculated data, it shouldn't be called 'observation' - need a different name
    frame_number: int
    triangulated_data: np.ndarray
    reprojection_error: np.ndarray
    reprojection_error_by_camera: np.ndarray


class Trajectory3d(BaseModel):
    start_frame: int
    end_frame: int
    triangulated_data: np.ndarray
    reprojection_error: np.ndarray
    reprojection_error_by_camera: np.ndarray

    @classmethod
    def from_observations(cls, observations):
        observations = sorted(observations, key=lambda x: x.frame_number)
        start_frame = observations[0].frame_number
        end_frame = observations[-1].frame_number
        if end_frame - start_frame != len(observations) - 1:
            raise ValueError("Observations are not contiguous")
        triangulated_data = np.stack(
            [observation.triangulated_data for observation in observations]
        )
        reprojection_error = np.stack(
            [observation.reprojection_error for observation in observations]
        )
        reprojection_error_by_camera = np.stack(
            [observation.reprojection_error_by_camera for observation in observations]
        )
        return cls(
            start_frame=start_frame,
            end_frame=end_frame,
            triangulated_data=triangulated_data,
            reprojection_error=reprojection_error,
            reprojection_error_by_camera=reprojection_error_by_camera,
        )
