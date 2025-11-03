from skellyforge.post_processing.interpolation.interpolation_config import InterpolationConfig
from skellyforge.data_models.data_3d import Trajectory3d
from skellyforge.post_processing.interpolation.interpolation_registry import INTERPOLATION_REGISTRY
import numpy as np
from tqdm import tqdm

def fill_in_nans(data:np.ndarray) -> np.ndarray:
    out = data.copy()
    return np.where(np.isfinite(out), out, np.nanmean(out))

def interpolate_trajectory(
        trajectory: Trajectory3d,
        config:InterpolationConfig
):
    data = trajectory.triangulated_data

    num_frames, num_markers, num_dims = data.shape
    interpolated_data =  np.empty_like(data, dtype=float)

    interp_function = INTERPOLATION_REGISTRY[config.method]
    for marker in tqdm(range(num_markers), desc = "Interpolating data"):
        interp_marker = interp_function(
            data[:, marker, :], #passing all dimensions works for pandas, not necessarily for other methods
            config
        )
        interpolated_data[:, marker, :] = fill_in_nans(interp_marker)

    return Trajectory3d(
        start_frame=trajectory.start_frame,
        end_frame=trajectory.end_frame,
        triangulated_data=interpolated_data,
        reprojection_error=trajectory.reprojection_error,
        reprojection_error_by_camera=trajectory.reprojection_error_by_camera,
    )