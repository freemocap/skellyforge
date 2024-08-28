import json
from pathlib import Path
from typing import Tuple, Union
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow,
    QApplication,
    QTabWidget,
    QWidget,
    QVBoxLayout,
)

from skellyforge.freemocap_utils.constants import (
    OUTPUT_DATA_FOLDER_NAME,
    POSTPROCESSING_SETTINGS_FILE_NAME,
    RAW_DATA_FILE_NAME,
    RAW_DATA_FOLDER_NAME,
    RECORDING_SETTINGS_FILE_NAME,
)
from skellyforge.freemocap_utils.postprocessing_widgets.visualization_widgets.skeleton_builder import (
    mediapipe_model_info,
)
from skellyforge.freemocap_utils.postprocessing_widgets.menus.main_menu import MainMenu
from skellyforge.freemocap_utils.postprocessing_widgets.menus.interpolation_menu import (
    InterpolationMenu,
)
from skellyforge.freemocap_utils.postprocessing_widgets.menus.filtering_menu import (
    FilteringMenu,
)
import toml


class FileManager:
    def __init__(self, path_to_recording: Union[str, Path]):
        self.path_to_recording = Path(path_to_recording)
        # self.data_array_path = self.path_to_recording/'DataArrays'
        self.output_data_array_path = self.path_to_recording / OUTPUT_DATA_FOLDER_NAME
        self.raw_data_array_path = self.output_data_array_path / RAW_DATA_FOLDER_NAME

    def load_skeleton_data(self) -> np.ndarray:
        # freemocap_raw_data = np.load(self.data_array_path/'mediaPipeSkel_3d.npy')
        freemocap_raw_data = np.load(self.raw_data_array_path / RAW_DATA_FILE_NAME)
        freemocap_raw_data = freemocap_raw_data[:, :, :]
        return freemocap_raw_data

    def save_skeleton_data(
        self, skeleton_data: np.ndarray, skeleton_file_name: str, settings_dict: dict
    ):
        np.save(self.output_data_array_path / skeleton_file_name, skeleton_data)

        output_toml_name = (
            self.output_data_array_path / POSTPROCESSING_SETTINGS_FILE_NAME
        )
        toml_string = toml.dumps(settings_dict)

        with open(output_toml_name, "w") as toml_file:
            toml_file.write(toml_string)

    def load_recording_settings(self) -> dict:
        try:
            with open(
                self.output_data_array_path / RECORDING_SETTINGS_FILE_NAME, "r"
            ) as json_file:
                recording_settings = json.load(json_file)
        except FileNotFoundError: # if there isn't a settings file, passing an empty dict down will lead to mediapipe as a default
            recording_settings = {}

        return recording_settings


class PostProcessingGUI(QWidget):
    def __init__(self, path_to_data_folder: Path):
        super().__init__()

        layout = QVBoxLayout()

        self.file_manager = FileManager(path_to_recording=path_to_data_folder)

        self.resize(1256, 1029)

        self.setWindowTitle("Freemocap Data Post-processing")

        self.tab_widget = QTabWidget()

        freemocap_raw_data = self.file_manager.load_skeleton_data()
        recording_settings = self.file_manager.load_recording_settings()
        landmark_names, connections = self.get_landmarks_and_connections(
            recording_settings=recording_settings
        )

        self.main_menu_tab = MainMenu(
            freemocap_raw_data=freemocap_raw_data,
            connections=connections,
            landmark_names=landmark_names,
        )
        self.tab_widget.addTab(self.main_menu_tab, "Main Menu")

        self.interp_tab = InterpolationMenu(
            freemocap_raw_data=freemocap_raw_data, landmark_names=landmark_names
        )
        self.tab_widget.addTab(self.interp_tab, "Interpolation")
        # layout.addWidget(self.main_menu)

        self.filter_tab = FilteringMenu(
            freemocap_raw_data=freemocap_raw_data, landmark_names=landmark_names
        )
        self.tab_widget.addTab(self.filter_tab, "Filtering")

        layout.addWidget(self.tab_widget)

        self.main_menu_tab.save_skeleton_data_signal.connect(
            self.file_manager.save_skeleton_data
        )

        self.setLayout(layout)

    def get_landmarks_and_connections(
        self, recording_settings: dict
    ) -> Tuple[list, list]:
        if "tracking_model_info" in recording_settings:
            model_info = recording_settings["tracking_model_info"]
        else:
            model_info = mediapipe_model_info  # older sessions won't have tracking model info, so default to mediapipe

        # we want to exclude hands and face, so we default to body if it's specified
        if "body_landmark_names" in model_info:
            landmark_names = model_info["body_landmark_names"]
        else:
            landmark_names = model_info["landmark_names"]

        if "body_connections" in model_info:
            connections = model_info["body_connections"]
        else:
            connections = model_info["connections"]

        return landmark_names, connections


class MainWindow(QMainWindow):
    def __init__(self, path_to_data_folder: Path):
        super().__init__()

        layout = QVBoxLayout()

        widget = QWidget()
        postprocessing_window = PostProcessingGUI(path_to_data_folder)

        layout.addWidget(postprocessing_window)

        widget.setLayout(layout)
        self.setCentralWidget(widget)


def main(path_to_data_folder=None):
    if not path_to_data_folder:
        path_to_data_folder = Path(
            input("Enter path to data folder (no quotations around the path): ")
        )

    app = QApplication([])
    win = MainWindow(path_to_data_folder)
    win.show()
    app.exec()


if __name__ == "__main__":

    main(
        Path(
            r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3"
        )
    )
