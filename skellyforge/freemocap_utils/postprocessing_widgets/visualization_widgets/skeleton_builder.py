from typing import List, Tuple
import numpy as np
from rich.progress import track


def build_skeleton(
    skeleton_3d_frame_marker_xyz: np.ndarray,
    pose_estimation_connections: List[Tuple[int, int]],
) -> List[List[List[np.ndarray]]]:
    num_frames = skeleton_3d_frame_marker_xyz.shape[0]

    skeleton_connection_coordinates = []

    for frame in track(range(num_frames)):
        frame_connection_coordinates = []
        for connection in pose_estimation_connections:
            joint_1_coordinates = skeleton_3d_frame_marker_xyz[frame, connection[0], :]
            joint_2_coordinates = skeleton_3d_frame_marker_xyz[frame, connection[1], :]

            connection_coordinates = [joint_1_coordinates, joint_2_coordinates]

            frame_connection_coordinates.append(connection_coordinates)
        skeleton_connection_coordinates.append(frame_connection_coordinates)

    return skeleton_connection_coordinates


def get_index_by_name(name: str, landmark_names: list) -> int:
    return landmark_names.index(name)


# this is just here to have a default/backup for sessions that didn't have model info included
mediapipe_model_info = {
    "body_landmark_names": [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ],
    "landmark_names": [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ],
    "body_connections": [
        (15, 21),
        (16, 20),
        (18, 20),
        (3, 7),
        (14, 16),
        (23, 25),
        (28, 30),
        (11, 23),
        (27, 31),
        (6, 8),
        (15, 17),
        (24, 26),
        (16, 22),
        (4, 5),
        (5, 6),
        (29, 31),
        (12, 24),
        (23, 24),
        (0, 1),
        (9, 10),
        (1, 2),
        (0, 4),
        (11, 13),
        (30, 32),
        (28, 32),
        (15, 19),
        (16, 18),
        (25, 27),
        (26, 28),
        (12, 14),
        (17, 19),
        (2, 3),
        (11, 12),
        (27, 29),
        (13, 15),
    ],
}
