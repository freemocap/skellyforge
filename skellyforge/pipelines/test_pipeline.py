from pathlib import Path
import numpy as np
from skellyforge.data_models.data_3d import Trajectory3d


path_to_data = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\validation\mediapipe_dlc\mediapipe_dlc_body_3d_xyz.npy")
data = np.load(path_to_data)

trajectory = Trajectory3d(
    start_frame=0,
    end_frame=data.shape[0] - 1,
    triangulated_data=data,
    reprojection_error=np.random.rand(data.shape[0], 1),
    reprojection_error_by_camera=np.random.rand(data.shape[0], 4),
)


f = 2