from pathlib import Path

import logging
logger = logging.getLogger(__name__)
try:
    from skellytracker.trackers.dlc_tracker.__dlc_tracker import (
    DeepLabCutTracker,
    DeepLabCutTrackerConfig,
    )
except ImportError as e:
    logger.warning("Error importing DeepLabCutTracker. Make sure skellytracker.DeepLabCutTracker is properly installed.")

from skellyforge.triangulation.triangulate import (
    TriangulationConfig,
    triangulate_dict,
)
from skellyforge.triangulation.load_camera_group import load_camera_group_from_toml
from skellyforge.utilities.get_files_from_folder import get_csvs_from_folder, get_videos_from_folder


def run_dlc_pipeline(
    video_paths: list[str | Path],
    annotated_video_folder: str | Path,
    output_data_folder: str | Path,
    dlc_config_path: str,
    camera_calibration_data_toml_path: str | Path,
    triangulation_config: TriangulationConfig,
):
    tracker_output = {}
    raw_data_folder = Path(output_data_folder) / "raw_data"
    raw_data_folder.mkdir(parents=True, exist_ok=True)
    Path(annotated_video_folder).mkdir(parents=True, exist_ok=True)
    for video_path in video_paths:
        video_path = Path(video_path)
        tracker = DeepLabCutTracker.create(
            DeepLabCutTrackerConfig.from_config_yaml(dlc_config_path)
        )
        print(f"Processing video: {video_path}")
        tracker.recorder.load_deeplabcut_csv(csv_path=video_path)
        # tracker.process_video(input_video_filepath=video_path)
        # # TODO: apply confidence thresholding, either as part of tracker or as preprocessing
        # print(f"Annotating video: {video_path}")
        # tracker.annotate_video(
        #     input_video_filepath=video_path,
        #     output_video_filepath=Path(annotated_video_folder)
        #     / f"{video_path.stem}_annotated.{video_path.suffix}",
        # )
        tracker_output[video_path.stem] = tracker.recorder.to_array
        # tracker.recorder.save_array(
        #     output_path=raw_data_folder
        #     / f"{video_path.stem}_{tracker.config.tracker_name}_{tracker.config.iteration}.npy"
        # )
    camera_group = load_camera_group_from_toml(camera_calibration_data_toml_path)

    raw_trajectory_3d = triangulate_dict(
        data_dict=tracker_output, camera_group=camera_group, config=triangulation_config
    )

    raw_trajectory_3d.save_to_arrays(output_folder=output_data_folder)

if __name__ == "__main__":
    # video_folder = "/Users/philipqueen/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s/mocap_data/synchronized_videos/"
    # video_paths = get_videos_from_folder(video_folder=video_folder)
    video_folder = "/Users/philipqueen/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s/mocap_data/dlc_output/head_body_eyecam_v1_model_outputs_iteration_17/"
    video_paths = get_csvs_from_folder(video_folder=video_folder)
    calibration_toml = "/Users/philipqueen/session_2025-07-01_ferret_757_EyeCameras_P33EO5/calibration/session_2025-07-01_calibration_camera_calibration.toml"

    dlc_config_path = "/Users/philipqueen/head_body_eyecam_v1/config.yaml"

    recording_folder = "/Users/philipqueen/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/forge_test"
    output_data_folder = Path(recording_folder) / "output_data"
    annotated_video_folder = Path(recording_folder) / "annotated_videos"

    triangulation_config = TriangulationConfig(use_ransac=False)

    run_dlc_pipeline(
        video_paths=video_paths,
        annotated_video_folder=annotated_video_folder,
        output_data_folder=output_data_folder,
        dlc_config_path=dlc_config_path,
        camera_calibration_data_toml_path=calibration_toml,
        triangulation_config=triangulation_config,
    )