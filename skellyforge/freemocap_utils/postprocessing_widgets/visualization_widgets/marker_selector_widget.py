from PySide6.QtWidgets import QWidget, QVBoxLayout, QComboBox
from PySide6.QtCore import Signal


class MarkerSelectorWidget(QWidget):
    marker_to_plot_updated_signal = Signal()
    def __init__(self, landmark_names: list):
        super().__init__()

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        combo_box_items = landmark_names
        # combo_box_items.insert(0,'')
        self.marker_combo_box = QComboBox()
        self.marker_combo_box.addItems(combo_box_items)
        self.marker_combo_box.addItems(['center of mass'])
        self._layout.addWidget(self.marker_combo_box)

        self.current_marker = self.marker_combo_box.currentText()
        self.marker_combo_box.currentTextChanged.connect(self.return_marker)

    def return_marker(self):
        self.current_marker = self.marker_combo_box.currentText()
        self.marker_to_plot_updated_signal.emit()

        return self.current_marker


