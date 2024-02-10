from PySide6.QtWidgets import QWidget, QHBoxLayout
import numpy as np
from skellyforge.freemocap_utils.postprocessing_widgets.visualization_widgets.skeleton_view_widget import (
    SkeletonViewWidget,
)


class SkeletonViewersContainer(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()

        self.raw_skeleton_viewer = SkeletonViewWidget("Raw data")
        layout.addWidget(self.raw_skeleton_viewer)

        self.processed_skeleton_viewer = SkeletonViewWidget("Post-processed data")
        layout.addWidget(self.processed_skeleton_viewer)

        self.setLayout(layout)

    def plot_raw_skeleton(self, raw_skeleton_data: np.ndarray, connections: list):
        self.raw_skeleton_viewer.load_skeleton(
            skeleton_3d_data=raw_skeleton_data, connections=connections
        )

    def plot_processed_skeleton(
        self, processed_skeleton_data: np.ndarray, connections: list
    ):
        self.processed_skeleton_viewer.load_skeleton(
            skeleton_3d_data=processed_skeleton_data, connections=connections
        )

    def update_raw_viewer_plot(self, frame_number):
        self.raw_skeleton_viewer.replot(frame_number)

    def update_processed_viewer_plot(self, frame_number):
        if self.processed_skeleton_viewer.skeleton_loaded:
            self.processed_skeleton_viewer.replot(frame_number)
