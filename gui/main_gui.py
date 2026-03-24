import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import serial
from collections import deque
import re
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.filtering import preprocess_frame, filter_window
from core.features import extract_features
from Machine_learning import predict_all, select_12_features
from deep_learning import model as dl_model

class ISAC_GUI(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.ser1 = serial.Serial("COM5", 115200, timeout=1)
        self.ser2 = serial.Serial("COM6", 115200, timeout=1)
        self.WINDOW_SIZE = 30
        self.FS = 20

        self.amp_w1 = deque(maxlen=self.WINDOW_SIZE)
        self.phase_w1 = deque(maxlen=self.WINDOW_SIZE)
        self.rssi_w1 = deque(maxlen=self.WINDOW_SIZE)

        self.amp_w2 = deque(maxlen=self.WINDOW_SIZE)
        self.phase_w2 = deque(maxlen=self.WINDOW_SIZE)
        self.rssi_w2 = deque(maxlen=self.WINDOW_SIZE)

        self.setWindowTitle("🚀 ISAC Dashboard")
        self.setGeometry(100, 100, 1200, 800)

        # =========================
        # 🔥 SCROLLABLE MAIN AREA
        # =========================
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)

        container = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QVBoxLayout(container)

        scroll.setWidget(container)
        self.setCentralWidget(scroll)

        # =========================
        # 🔥 DARK THEME
        # =========================
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }

            QLabel { color: #ffffff; font-size: 16px; }

            QPushButton {
                background-color: #1f1f1f;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 16px;
            }

            QPushButton:hover { background-color: #333333; }

            QFrame {
                background-color: #1e1e1e;
                border-radius: 15px;
                padding: 15px;
            }

            QComboBox {
                background-color: #1f1f1f;
                color: #ffffff;
                padding: 5px;
                font-size: 14px;
            }

            QTextEdit {
                background-color: #1e1e1e;
                color: #00ffcc;
                font-size: 12px;
                border-radius: 10px;
            }

            QScrollBar:vertical {
                background: #1e1e1e;
                width: 10px;
            }

            QScrollBar::handle:vertical {
                background: #555;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical:hover {
                background: #777777;
            }

            QTableWidget {
                background-color: #1e1e1e;
                color: white;
                gridline-color: #333;
                font-size: 12px;
            }

            QHeaderView::section {
                background-color: #2a2a2a;
                color: white;
                padding: 5px;
            }
        """)

        # =========================
        # 🔷 NAVBAR
        # =========================
        nav_layout = QtWidgets.QHBoxLayout()
        nav_layout.addStretch()

        self.start_btn = QtWidgets.QPushButton("▶ Start")
        self.stop_btn = QtWidgets.QPushButton("⏹ Stop")

        nav_layout.addWidget(self.start_btn)
        nav_layout.addSpacing(20)
        nav_layout.addWidget(self.stop_btn)

        nav_layout.addStretch()
        self.main_layout.addLayout(nav_layout)
        self.main_layout.setStretch(self.main_layout.count()-1, 0)

        # =========================
        # 🔷 OUTPUT CARDS
        # =========================
        output_layout = QtWidgets.QHBoxLayout()

        self.threshold_card = self.create_card("THRESHOLD")
        self.ml_card = self.create_card("ML MODEL")
        self.dl_card = self.create_card("DEEP LEARNING")

        output_layout.addWidget(self.threshold_card)
        output_layout.addWidget(self.ml_card)
        output_layout.addWidget(self.dl_card)

        self.main_layout.addLayout(output_layout)
        self.main_layout.setStretch(self.main_layout.count()-1, 1)

        # =========================
        # 🔷 FEATURE SELECTOR
        # =========================
        feature_layout = QtWidgets.QHBoxLayout()

        self.feature_dropdown = QtWidgets.QComboBox()
        self.feature_dropdown.addItems([
            "rms", "variance", "doppler_frequency", "peak_freq"
        ])

        feature_layout.addWidget(QtWidgets.QLabel("Feature:"))
        feature_layout.addWidget(self.feature_dropdown)
        feature_layout.addStretch()

        self.main_layout.addLayout(feature_layout)
        self.main_layout.setStretch(self.main_layout.count()-1, 0)

        # =========================
        # 🔷 GRAPH
        # =========================
        self.plot = pg.PlotWidget(title="Feature Graph")
        self.plot.setBackground('#121212')
        self.curve = self.plot.plot(pen=pg.mkPen(color='cyan', width=2))
        self.plot.setMinimumHeight(400)
        self.plot.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.main_layout.addWidget(self.plot)
        self.main_layout.setStretch(self.main_layout.count()-1, 3)

        # =========================
        # 🔷 RAW INPUT
        # =========================
        raw_layout = QtWidgets.QHBoxLayout()

        self.raw1_text = QtWidgets.QTextEdit()
        self.raw2_text = QtWidgets.QTextEdit()

        for text in [self.raw1_text, self.raw2_text]:
            text.setReadOnly(True)
            text.setMinimumHeight(250)
            text.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        raw_layout.addWidget(self.wrap("Raw Input 1", self.raw1_text))
        raw_layout.addWidget(self.wrap("Raw Input 2", self.raw2_text))

        self.main_layout.addLayout(raw_layout)
        self.main_layout.setStretch(self.main_layout.count()-1, 2)

        # =========================
        # 🔷 DATASET
        # =========================
        dataset_layout = QtWidgets.QVBoxLayout()

        top = QtWidgets.QHBoxLayout()
        self.dataset_dropdown = QtWidgets.QComboBox()
        self.dataset_dropdown.addItems(["ML Dataset", "DL Dataset"])

        top.addWidget(QtWidgets.QLabel("Dataset:"))
        top.addWidget(self.dataset_dropdown)
        top.addStretch()

        self.dataset_table = QtWidgets.QTableWidget()
        self.dataset_table.setMinimumHeight(300)

        dataset_layout.addLayout(top)
        dataset_layout.addWidget(self.dataset_table)

        self.main_layout.addLayout(dataset_layout)
        self.main_layout.setStretch(self.main_layout.count()-1, 2)

        # =========================
        # 🔷 DATA
        # =========================
        self.history = {k: [] for k in ["rms","variance","doppler_frequency","peak_freq"]}

        self.load_datasets()
        self.update_dataset_view()
        self.dataset_dropdown.currentIndexChanged.connect(self.update_dataset_view)

        # =========================
        # 🔷 TIMER
        # =========================
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_output)

        self.start_btn.clicked.connect(lambda: self.timer.start(200))
        self.stop_btn.clicked.connect(self.timer.stop)

    # =========================
    def wrap(self, title, widget):
        frame = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout(frame)

        label = QtWidgets.QLabel(title)
        label.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(label)
        layout.addWidget(widget)

        return frame

    # =========================
    def create_card(self, title):
        frame = QtWidgets.QFrame()
        frame.setMinimumHeight(150)
        frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        layout = QtWidgets.QVBoxLayout(frame)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        title_label = QtWidgets.QLabel(title)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("font-size:20px; font-weight:bold;")

        output = QtWidgets.QLabel("---")
        output.setAlignment(QtCore.Qt.AlignCenter)
        output.setStyleSheet("font-size:18px;")

        layout.addWidget(title_label)
        layout.addWidget(output)

        frame.output_label = output
        return frame

    # =========================
    def update_output(self):
        

        # ===== READ SERIAL =====
        line1 = self.ser1.readline().decode(errors='ignore').strip()
        line2 = self.ser2.readline().decode(errors='ignore').strip()

        if not line1 or not line2:
            return

        # ===== SHOW RAW DATA =====
        self.raw1_text.append(line1)   # COM5
        self.raw2_text.append(line2)   # COM6

        # ===== PARSE =====
        amp1, phase1, rssi1 = self.parse_line(line1)
        amp2, phase2, rssi2 = self.parse_line(line2)

        if amp1 is None or amp2 is None:
            return

        # ===== PREPROCESS =====
        amp1, phase1 = preprocess_frame(amp1, phase1)
        amp2, phase2 = preprocess_frame(amp2, phase2)

        # ===== STORE WINDOWS =====
        self.amp_w1.append(amp1)
        self.phase_w1.append(phase1)
        self.rssi_w1.append(rssi1)

        self.amp_w2.append(amp2)
        self.phase_w2.append(phase2)
        self.rssi_w2.append(rssi2)

        # ===== WAIT UNTIL WINDOW FULL =====
        if len(self.amp_w1) < self.WINDOW_SIZE:
            return

        # ===== FILTER =====
        amp_f1, phase_f1, sig1 = filter_window(self.amp_w1, self.phase_w1)
        amp_f2, phase_f2, sig2 = filter_window(self.amp_w2, self.phase_w2)

        # ===== PHASE SIGNAL =====
        phase_sig1 = np.mean(phase_f1, axis=0)
        phase_sig2 = np.mean(phase_f2, axis=0)

        # ===== FEATURE EXTRACTION =====
        f1 = extract_features(sig1, phase_sig1, self.rssi_w1, self.FS)
        f2 = extract_features(sig2, phase_sig2, self.rssi_w2, self.FS)

        # ===== GRAPH UPDATE =====
        selected = self.feature_dropdown.currentText()
        value = f1.get(selected, 0)

        self.history[selected].append(value)
        if len(self.history[selected]) > 100:
            self.history[selected].pop(0)

        self.curve.setData(self.history[selected])

        # ===== THRESHOLD MODEL =====
        threshold_pred = "HUMAN" if f1["variance"] > 1.0 else "EMPTY"

        self.threshold_card.output_label.setText(
            f"{threshold_pred}\nVar: {f1['variance']:.2f}"
        )

        # ===== ML MODEL =====
        f1_sel = select_12_features(f1)
        f2_sel = select_12_features(f2)

        features = f1_sel + f2_sel

        xgb_pred, xgb_prob, ens_pred, ens_prob = predict_all(features)

        self.ml_card.output_label.setText(
            f"XGB: {xgb_pred} ({xgb_prob:.2f})\nENS: {ens_pred} ({ens_prob:.2f})"
        )

        # ===== DL MODEL =====
        dl_input = np.array(features).reshape(1, -1)

        dl_pred = dl_model.predict(dl_input, verbose=0)
        dl_label = np.argmax(dl_pred)
        dl_conf = np.max(dl_pred)

        self.dl_card.output_label.setText(
            f"State: {dl_label}\nConf: {dl_conf:.2f}"
        )

        # ===== AUTO SCROLL =====
        for t in [self.raw1_text, self.raw2_text]:
                t.verticalScrollBar().setValue(
                    t.verticalScrollBar().maximum()
                )             
        # =========================
        
        
    def load_datasets(self):
        self.ml_data = np.random.randint(0,100,(20,6))
        self.dl_data = np.random.randint(0,100,(20,6))

    def update_dataset_view(self):
        data = self.ml_data if self.dataset_dropdown.currentText()=="ML Dataset" else self.dl_data

        self.dataset_table.setRowCount(data.shape[0])
        self.dataset_table.setColumnCount(data.shape[1])

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.dataset_table.setItem(i,j,QtWidgets.QTableWidgetItem(str(data[i,j])))
                
    def parse_line(self, line):
        try:
            rssi_match = re.search(r"RSSI\s*:\s*(-?\d+)", line)
            data_match = re.search(r"\[(.*?)\]", line)

            if not rssi_match or not data_match:
                return None, None, None

            rssi = int(rssi_match.group(1))
            data = list(map(int, data_match.group(1).split(",")))

            amp = data[::2]
            phase = data[1::2]

            return amp, phase, rssi
        except:
            return None, None, None
# ===== RUN =====
def run():
    app = QtWidgets.QApplication(sys.argv)
    window = ISAC_GUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()