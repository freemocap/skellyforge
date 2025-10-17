from pathlib import Path
import numpy as np
from skellyforge.data_models.data_3d import Trajectory3d

from skellyforge.post_processing.interpolation.apply_interpolation import interpolate_trajectory
from skellyforge.post_processing.interpolation.interpolation_config import InterpolationConfig,InterpolationMethod

from skellyforge.post_processing.filters.apply_filter import filter_trajectory
from skellyforge.post_processing.filters.filter_config import FilterConfig, FilterMethod



def test_pipeline(data:np.ndarray,
                  interp_config: InterpolationConfig,
                  filter_config: FilterConfig):

    raw_trajectory = Trajectory3d(
        start_frame=0,
        end_frame=data.shape[0] - 1,
        triangulated_data=data,
        reprojection_error=np.random.rand(data.shape[0], 1),
        reprojection_error_by_camera=np.random.rand(data.shape[0], 4),
    )

    interpolated_trajectory = interpolate_trajectory(raw_trajectory, interp_config)
    
    filtered_trajectory = filter_trajectory(interpolated_trajectory, filter_config)
    
    f = 2

if __name__ == "__main__":

    interp_config = InterpolationConfig(
        method = InterpolationMethod.linear
    )

    filter_config = FilterConfig(
        method = FilterMethod.butter_low_pass,
        cutoff = 6.0,
        sampling_rate= 30.0,
        order = 4
    )
    
    path_to_data = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\validation\mediapipe_dlc\mediapipe_dlc_body_3d_xyz.npy")
    data = np.load(path_to_data)

    test_pipeline(data, interp_config, filter_config)
