import logging
import numpy as np
from pydantic import BaseModel

from skellyforge.calibration.freemocap_anipose import CameraGroup
from skellyforge.data_models.data_3d import Observation3d, Trajectory3d
from skellyforge.data_models.frame_group import FrameGroup, Trajectory2dGroup

logger = logging.getLogger(__name__)


class TriangulationConfig(BaseModel):
    use_ransac: bool


def triangulate_array(
    data_2d: np.ndarray,
    camera_group: CameraGroup,
    config: TriangulationConfig,
):
    number_of_cameras = data_2d.shape[0]
    number_of_frames = data_2d.shape[1]
    number_of_tracked_points = data_2d.shape[2]
    number_of_spatial_dimensions = data_2d.shape[3]

    if number_of_spatial_dimensions != 2:
        logger.error(
            f"Expected 2D data but got {number_of_spatial_dimensions} dimensions"
        )
        raise ValueError("Input data must have 2 spatial dimensions")

    data2d_flat = data_2d.reshape(number_of_cameras, -1, 2)

    # Triangulate 2D points to 3D
    if config.use_ransac:
        logger.info("Using RANSAC triangulation method")
        triangulated_data_flat = camera_group.triangulate_ransac(
            data2d_flat, progress=True, kill_event=None
        )
    else:
        logger.info("Using standard triangulation method")
        triangulated_data_flat = camera_group.triangulate(
            data2d_flat, progress=False, kill_event=None
        )

    if triangulated_data_flat is None:
        raise ValueError("Triangulation stopped due to kill event")

    # Reshape to frames × points × xyz
    triangulated_data = triangulated_data_flat.reshape(
        number_of_frames, number_of_tracked_points, 3
    )

    # Calculate reprojection errors
    reprojection_error_full = camera_group.reprojection_error(
        triangulated_data_flat, data2d_flat
    )
    reprojection_error_flat = camera_group.calculate_mean_reprojection_error(
        reprojection_error_full
    )

    # Reshape reprojection errors
    reprojection_error_by_camera = np.linalg.norm(
        reprojection_error_full, axis=2
    ).reshape(number_of_cameras, number_of_frames, number_of_tracked_points)
    reprojection_error = reprojection_error_flat.reshape(
        number_of_frames, number_of_tracked_points
    )

    return triangulated_data, reprojection_error, reprojection_error_by_camera


def triangulate_frame_group(
    frame_number: int,
    frame_group: FrameGroup,
    camera_group: CameraGroup,
    config: TriangulationConfig,
) -> Observation3d:
    if list(frame_group.keys()) != [camera.name for camera in camera_group.cameras]:
        if set(frame_group.keys()).issubset(
            set(camera.name for camera in camera_group.cameras)
        ):
            logger.warning(
                "Frame group is missing cameras from camera group, triangulating with only cameras in frame group"
            )
            camera_group = camera_group.subset_cameras_names(list(frame_group.keys()))
        raise ValueError(
            "Camera names in frame group do not match camera names in camera group. Make sure calibration matches input data."
        )

    data_2d = np.array(
        [frame_group[camera.name].to_array for camera in camera_group.cameras]
    )

    if len(frame_group) != data_2d.shape[0]:
        logger.error(
            f"Expected {len(frame_group)} cameras but got {data_2d.shape[0]} cameras"
        )
        raise ValueError(
            "Input data must have the same number of cameras as the camera group"
        )
    
    triangulated_data, reprojection_error, reprojection_error_by_camera = triangulate_array(
        data_2d, camera_group, config
    )


    return Observation3d(
        frame_number=frame_number,
        triangulated_data=triangulated_data,
        reprojection_error=reprojection_error,
        reprojection_error_by_camera=reprojection_error_by_camera,
    )


def triangulate_frame_groups(
    frame_groups: dict[int, FrameGroup],
    camera_group: CameraGroup,
    config: TriangulationConfig,
) -> Trajectory3d:
    observations_3d = []
    frame_groups = dict(sorted(frame_groups.items()))
    for frame_number, frame_group in frame_groups.items():
        observations_3d.append(triangulate_frame_group(
            frame_number, frame_group, camera_group, config
        ))
    return Trajectory3d.from_observations(list(frame_groups.values()))


def triangulate_trajectories(
    trajectory_group: Trajectory2dGroup,
    camera_group: CameraGroup,
    config: TriangulationConfig,
):
    # TODO: move this validation into Trajectory2dGroup creation
    if len(set(trajectory.start_frame for trajectory in trajectory_group.values())) != 1:
        raise ValueError(
            "Input data must have the same start frame for all trajectories"
    )
    if len(set(trajectory.end_frame for trajectory in trajectory_group.values())) != 1:
        raise ValueError(
            "Input data must have the same end frame for all trajectories"
    )
    if list(trajectory_group.keys()) != [camera.name for camera in camera_group.cameras]:
        if set(trajectory_group.keys()).issubset(
            set(camera.name for camera in camera_group.cameras)
        ):
            logger.warning(
                "Frame group is missing cameras from camera group, triangulating with only cameras in frame group"
            )
            camera_group = camera_group.subset_cameras_names(list(frame_group.keys()))
        raise ValueError(
            "Camera names in frame group do not match camera names in camera group. Make sure calibration matches input data."
        )

    data_2d = np.array(
        [trajectory_group[camera.name].points_2d for camera in camera_group.cameras]
    )

    if len(trajectory_group) != data_2d.shape[0]:
        logger.error(
            f"Expected {len(trajectory_group)} cameras but got {data_2d.shape[0]} cameras"
        )
        raise ValueError(
            "Input data must have the same number of cameras as the camera group"
        )
    
    triangulated_data, reprojection_error, reprojection_error_by_camera = triangulate_array(
        data_2d, camera_group, config
    )


    return Trajectory3d(
        triangulated_data=triangulated_data,
        reprojection_error=reprojection_error,
        reprojection_error_by_camera=reprojection_error_by_camera,
    )
    