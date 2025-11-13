from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

class Point3d(BaseModel):
    x: float
    y: float
    z: float

class Observation3d(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # TODO: since this is calculated data, it shouldn't be called 'observation' - need a different name
    frame_number: int
    names: list[str]
    triangulated_data: np.ndarray
    reprojection_error: np.ndarray | None = None
    reprojection_error_by_camera: np.ndarray | None = None

    def to_point_dictionary(self) -> dict[str, Point3d]:
        points = {}
        for i, name in enumerate(self.names):
            points[name] = Point3d(x=float(self.triangulated_data[i,0]),
                                   y=float(self.triangulated_data[i,1]),
                                   z=float(self.triangulated_data[i,2]),)
        return points


class Trajectory3d(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
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

    def save_to_arrays(self, output_folder: str | Path, prefix: str = ""):
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        if len(prefix) > 0 and not prefix.endswith("_"):
            prefix += "_"

        np.save(output_folder / f"{prefix}3d_data_spatial_xyz.npy", self.triangulated_data)
        np.save(output_folder / f"{prefix}reprojection_error.npy", self.reprojection_error)
        np.save(output_folder / f"{prefix}reprojection_error_by_camera.npy", self.reprojection_error_by_camera)

    def to_tidy_daframe(self):
        # TODO
        pass

    def to_wide_dataframe(self):
        # TODO
        pass
