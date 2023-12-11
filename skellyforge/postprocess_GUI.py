from pathlib import Path
import numpy as np

from PySide6.QtWidgets import QMainWindow, QApplication, QTabWidget, QWidget, QVBoxLayout

from skellyforge.freemocap_utils.postprocessing_widgets.menus.main_menu import MainMenu
from skellyforge.freemocap_utils.postprocessing_widgets.menus.interpolation_menu import InterpolationMenu
from skellyforge.freemocap_utils.postprocessing_widgets.menus.filtering_menu import FilteringMenu
import toml

class FileManager:
    def __init__(self, path_to_recording: str):
        self.path_to_recording = path_to_recording
        #self.data_array_path = self.path_to_recording/'DataArrays'
        self.output_data_array_path = self.path_to_recording/'output_data'
        self.raw_data_array_path = self.output_data_array_path/'raw_data'

    def load_skeleton_data(self):
        # freemocap_raw_data = np.load(self.data_array_path/'mediaPipeSkel_3d.npy')
        freemocap_raw_data = np.load(self.raw_data_array_path/'mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy')
        freemocap_raw_data = freemocap_raw_data[:,:,:]
        return freemocap_raw_data

    def save_skeleton_data(self, skeleton_data:np.ndarray, skeleton_file_name:str, settings_dict:dict):
        np.save(self.output_data_array_path/skeleton_file_name,skeleton_data)

        output_toml_name = self.output_data_array_path/'postprocessing_settings.toml'
        toml_string = toml.dumps(settings_dict)

        with open(output_toml_name, 'w') as toml_file:
            toml_file.write(toml_string)


class PostProcessingGUI(QWidget):
    def __init__(self,path_to_data_folder:Path):
        super().__init__()

        layout = QVBoxLayout()

        self.file_manager = FileManager(path_to_recording=path_to_data_folder)

        self.resize(1256, 1029)

        self.setWindowTitle("Freemocap Data Post-processing")

        self.tab_widget = QTabWidget()

        freemocap_raw_data = self.file_manager.load_skeleton_data()

        self.main_menu_tab = MainMenu(freemocap_raw_data=freemocap_raw_data)
        self.tab_widget.addTab(self.main_menu_tab, 'Main Menu')

        self.interp_tab = InterpolationMenu(freemocap_raw_data = freemocap_raw_data)
        self.tab_widget.addTab(self.interp_tab, 'Interpolation')
        # layout.addWidget(self.main_menu)

        self.filter_tab = FilteringMenu(freemocap_raw_data=freemocap_raw_data)
        self.tab_widget.addTab(self.filter_tab, 'Filtering')

        layout.addWidget(self.tab_widget)

        self.main_menu_tab.save_skeleton_data_signal.connect(self.file_manager.save_skeleton_data)

        self.setLayout(layout)


        f = 2

class MainWindow(QMainWindow):
    def __init__(self,path_to_data_folder:Path):
        super().__init__()

        layout = QVBoxLayout()

        widget = QWidget()
        postprocessing_window = PostProcessingGUI(path_to_data_folder)

        layout.addWidget(postprocessing_window)

        widget.setLayout(layout)
        self.setCentralWidget(widget)
        

def main(path_to_data_folder=None):
    if not path_to_data_folder:
        path_to_data_folder = Path(input("Enter path to data folder (no quotations around the path): "))
        

    app = QApplication([])
    win = MainWindow(path_to_data_folder)
    win.show()
    app.exec()



if __name__ == "__main__":

    main(Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3"))        

