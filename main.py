import sys
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# ===== IMPORT YOUR PIPELINE =====
try:
    from core.filtering import filter_window
    from core.features import extract_features
except:
    # fallback (for testing)
    def filter_window(a, b):
        return a, b, np.random.randn(100)

    def extract_features(final, phase, rssi):
        return {
            "rms": float(np.sqrt(np.mean(final**2))),
            "variance": float(np.var(final)),
            "doppler_frequency": np.random.rand(),
            "peak_freq": np.random.rand()
        }

# ===== MOCK MODELS (REPLACE WITH YOUR REAL MODELS) =====
class DummyModel:
    def predict(self, x):
        return {"activity": np.random.choice(["Walking", "Standing", "Empty"]),
                "confidence": np.random.randint(60, 95)}

threshold_model = DummyModel()
ml_model = DummyModel()
dl_model = DummyModel()


# ===== DATASET VIEWER =====
class DatasetViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Viewer")
        self.resize(600, 400)

        layout = QtWidgets.QVBoxLayout(self)

        self.table = QtWidgets.QTableWidget()
        layout.addWidget(self.table)

        self.load_data()

    def load_data(self):
        try:
            df = pd.read_csv("dataset.csv")
        except:
            df = pd.DataFrame(np.random.randn(50, 5),
                              columns=["RMS", "Var", "Doppler", "Freq", "Label"])

        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns)

        for i in range(len(df)):
            for j in range(len(df.columns)):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(str(df.iloc[i, j])))


# ===== MAIN GUI =====
class ISAC_GUI(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("🚀 ISAC Wi-Fi Sensing Dashboard")
        self.setGeometry(100, 100, 1500, 900)

        # ===== MAIN LAYOUT =====
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # ===== TOP STATUS =====
        self.status_label = QtWidgets.QLabel("Status: Disconnected")
        main_layout.addWidget(self.status_label)

        # ===== MIDDLE SPLIT =====
        middle_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(middle_layout)

        # ===== LEFT PANEL =====
        left_panel = QtWidgets.QVBoxLayout()

        self.start_btn = QtWidgets.QPushButton("▶ Start")
        self.stop_btn = QtWidgets.QPushButton("⏹ Stop")
        self.calib_btn = QtWidgets.QPushButton("🎯 Calibrate")
        self.dataset_btn = QtWidgets.QPushButton("📁 View Dataset")

        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.addItems(["ALL", "Threshold", "ML", "DL"])

        left_panel.addWidget(self.start_btn)
        left_panel.addWidget(self.stop_btn)
        left_panel.addWidget(self.calib_btn)
        left_panel.addWidget(self.dataset_btn)
        left_panel.addWidget(self.model_selector)

        middle_layout.addLayout(left_panel, 1)

        # ===== CENTER GRAPH PANEL =====
        graph_layout = QtWidgets.QVBoxLayout()

        self.csi_plot = pg.PlotWidget(title="CSI Signal")
        self.csi_curve = self.csi_plot.plot()

        self.feature_plot = pg.PlotWidget(title="Features")
        self.rms_curve = self.feature_plot.plot(pen='r')
        self.var_curve = self.feature_plot.plot(pen='g')

        graph_layout.addWidget(self.csi_plot)
        graph_layout.addWidget(self.feature_plot)

        middle_layout.addLayout(graph_layout, 3)

        # ===== RIGHT OUTPUT PANEL =====
        right_panel = QtWidgets.QVBoxLayout()

        self.threshold_label = QtWidgets.QLabel("Threshold: ---")
        self.ml_label = QtWidgets.QLabel("ML: ---")
        self.dl_label = QtWidgets.QLabel("DL: ---")
        self.final_label = QtWidgets.QLabel("FINAL: ---")

        right_panel.addWidget(self.threshold_label)
        right_panel.addWidget(self.ml_label)
        right_panel.addWidget(self.dl_label)
        right_panel.addWidget(self.final_label)

        middle_layout.addLayout(right_panel, 1)

        # ===== BOTTOM TABLE =====
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Frame", "RMS", "Variance", "Doppler", "Freq"]
        )
        main_layout.addWidget(self.table)

        # ===== DATA STORAGE =====
        self.frame_count = 0
        self.rms_history = []
        self.var_history = []

        # ===== TIMER =====
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        # ===== BUTTON CONNECTIONS =====
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.dataset_btn.clicked.connect(self.open_dataset)

    # ===== FUNCTIONS =====
    def start(self):
        self.status_label.setText("Status: Running")
        self.timer.start(50)

    def stop(self):
        self.status_label.setText("Status: Stopped")
        self.timer.stop()

    def open_dataset(self):
        self.viewer = DatasetViewer()
        self.viewer.show()

    def update_frame(self):
        self.frame_count += 1

        # ===== SIMULATED INPUT =====
        amp = np.random.randn(100)
        phase = np.random.randn(100)
        rssi = np.random.randn(100)

        # ===== PIPELINE =====
        amp_f, phase_f, final = filter_window(amp, phase)
        features = extract_features(final, phase, rssi)

        # ===== MODELS =====
        th = threshold_model.predict(final)
        ml = ml_model.predict(final)
        dl = dl_model.predict(final)

        # ===== UPDATE GRAPHS =====
        self.csi_curve.setData(final)

        self.rms_history.append(features["rms"])
        self.var_history.append(features["variance"])

        self.rms_curve.setData(self.rms_history[-100:])
        self.var_curve.setData(self.var_history[-100:])

        # ===== UPDATE OUTPUT =====
        self.threshold_label.setText(f"Threshold: {th}")
        self.ml_label.setText(f"ML: {ml}")
        self.dl_label.setText(f"DL: {dl}")
        self.final_label.setText(f"FINAL: {ml['activity']}")

        # ===== UPDATE TABLE =====
        row = self.table.rowCount()
        self.table.insertRow(row)

        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(self.frame_count)))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(features["rms"])))
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(features["variance"])))
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(features["doppler_frequency"])))
        self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(features["peak_freq"])))


# ===== RUN =====
def run():
    app = QtWidgets.QApplication(sys.argv)
    window = ISAC_GUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()