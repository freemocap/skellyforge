import numpy as np
from skellyforge.data_models.trajectory_3d import Trajectory3d
from skellyforge.post_processing.filters.filter_config import FilterConfig
from skellyforge.post_processing.filters.filter_registry import FILTER_REGISTRY
from tqdm import tqdm

def filter_trajectory(
    trajectory:Trajectory3d,
    config: FilterConfig 
):
    data = trajectory.triangulated_data

    num_frames, num_markers, num_dims = data.shape
    filtered_data =  np.empty_like(data, dtype=float)

    filter_function = FILTER_REGISTRY[config.method]
    for marker in tqdm(range(num_markers), desc = "Filtering data"):
        for dim in range(num_dims):
            filtered_data[:, marker, dim] = filter_function(
                data[:, marker, dim],
                config
            )
    
    return Trajectory3d(
        start_frame=trajectory.start_frame,
        end_frame=trajectory.end_frame,
        triangulated_data=filtered_data,
        reprojection_error=trajectory.reprojection_error,
        reprojection_error_by_camera=trajectory.reprojection_error_by_camera,
    )