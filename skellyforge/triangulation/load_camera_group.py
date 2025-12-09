from pathlib import Path
from typing import Union

from skellyforge.calibration.freemocap_anipose import AniposeCameraGroup



def load_camera_group_from_toml(
        camera_calibration_data_toml_path: Union[str, Path],
) -> AniposeCameraGroup:
    try:
        anipose_calibration_object = AniposeCameraGroup.load(str(camera_calibration_data_toml_path))

        return anipose_calibration_object
    except Exception as e:
        print(f"Failed to load anipose calibration info from {str(camera_calibration_data_toml_path)}")
        raise e
