from pathlib import Path
import numpy as np
from skellyforge.data_models.data_3d import Trajectory3d

from skellyforge.post_processing.interpolation.apply_interpolation import interpolate_trajectory
from skellyforge.post_processing.interpolation.interpolation_config import InterpolationConfig,InterpolationMethod

from skellyforge.post_processing.filters.apply_filter import filter_trajectory
from skellyforge.post_processing.filters.filter_config import FilterConfig, FilterMethod
from skellyforge.skellymodels.managers.human import Human, Animal
from skellyforge.skellymodels.models.tracking_model_info import ModelInfo

from pydantic import BaseModel

class TrackingConfig(BaseModel): #this seems like something that could be integrated or made earlier in the pipeline, so just leaving it here for now
    tracker: str
    name: str = "human"
    model_info: ModelInfo

def test_pipeline(data:np.ndarray,
                  interp_config: InterpolationConfig,
                  filter_config: FilterConfig,
                  tracking_config: TrackingConfig):

    raw_trajectory = Trajectory3d(
        start_frame=0,
        end_frame=data.shape[0] - 1,
        triangulated_data=data,
        reprojection_error=np.random.rand(data.shape[0], 1),
        reprojection_error_by_camera=np.random.rand(data.shape[0], 4),
    )

    interpolated_trajectory = interpolate_trajectory(raw_trajectory, interp_config)
    
    filtered_trajectory = filter_trajectory(interpolated_trajectory, filter_config)

    match tracking_config.tracker:
        case "mediapipe":
            skellymodel:Human = Human.from_tracked_points_numpy_array(
                name = tracking_config.name,
                model_info = tracking_config.model_info,
                tracked_points_numpy_array = filtered_trajectory.triangulated_data
            )

            skellymodel.put_skeleton_on_ground()
            skellymodel.fix_hands_to_wrist()
        
        case "dlc":
            skellymodel:Animal = Animal.from_tracked_points_numpy_array( #perhaps need to rethink the 'Animal'/'Human' distinction - there might be a more elegant solution?
                name = tracking_config.name,
                model_info = tracking_config.model_info,
                tracked_points_numpy_array = filtered_trajectory.triangulated_data
            )

    ## Uncomment if you want to save out data - would need to give it a path to where you want it to save to    
    # skellymodel.save_out_numpy_data()
    # skellymodel.save_out_csv_data()
    # skellymodel.save_out_all_data_csv()
    # skellymodel.save_out_all_data_parquet()
    # skellymodel.save_out_all_xyz_numpy_data()

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
    
    ## FOR MEDIAPIPE DATA
    # from skellyforge.skellymodels.models.tracking_model_info import MediapipeModelInfo #don't love importing this here but we're just in draft mode
    # tracking_config = TrackingConfig(
    #     tracker="mediapipe",
    #     name="human",
    #     model_info=MediapipeModelInfo()
    # )
    # path_to_raw_data = Path(r"D:\2025_09_03_OKK\freemocap\2025-09-03_14-56-30_GMT-4_okk_treadmill_1\output_data\raw_data\mediapipe_3dData_numFrames_numTrackedPoints_spatialXYZ.npy")

    ## FOR DLC DATA
    from skellyforge.skellymodels.models.tracking_model_info import ModelInfo 
    path_to_raw_data = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\output_data\raw_data\dlc_3dData_numFrames_numTrackedPoints_spatialXYZ.npy")
    path_to_model_yaml = Path(r"C:\Users\aaron\Documents\GitHub\freemocap_playground\dlc_reconstruction\model_infos\prosthetic_leg.yaml")    
    model_info = ModelInfo.from_config_path(path_to_model_yaml)
   
    tracking_config = TrackingConfig(
    tracker="dlc",
    name= model_info.name,
    model_info=model_info
)

    data = np.load(path_to_raw_data)

    test_pipeline(data, 
                  interp_config, 
                  filter_config,
                  tracking_config=tracking_config)