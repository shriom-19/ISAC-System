import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import serial
from collections import deque
import re
from presence_detection import PresenceDetector
from filtering import preprocess_frame, filter_window
from features import extract_features
from Machine_learning.Machine_learning import predict_all, select_12_features

from deeplearning.deep_learning import compute_doppler, normalize
from keras.models import load_model
# =========================
# 📦 LOAD DL MODEL (FULL PATH)
# =========================




class ISAC_GUI(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        # =========================
        # 🔌 SERIAL CONNECTION
        # =========================
        self.ser1 = serial.Serial("COM5", 115200, timeout=1)
        self.ser2 = serial.Serial("COM6", 115200, timeout=1)

        # =========================
        # 🧠 THRESHOLD MODEL (REAL)
        # =========================
        self.detector = PresenceDetector()

        # =========================
        # 📊 CALIBRATION SYSTEM
        # =========================
        self.calibration_data = []
        self.calibrated = False
        self.CALIBRATION_FRAMES = 50

        # =========================
        # ⚙️ BASIC SETTINGS
        # =========================
        self.WINDOW_SIZE = 25
        self.FS = 20

        # =========================
        # 📦 BUFFERS (FUSED PIPELINE)
        # =========================
        self.amp_w1 = deque(maxlen=self.WINDOW_SIZE)
        self.phase_w1 = deque(maxlen=self.WINDOW_SIZE)
        self.rssi_w1 = deque(maxlen=self.WINDOW_SIZE)

        # =========================
        # 🔥 DL PIPELINE (NEW)
        # =========================
        self.DL_WINDOW = 100
        self.SUBCARRIER_STEP = 16

        self.dl_buffer = deque(maxlen=self.DL_WINDOW)
        self.dl_pred_buffer = deque(maxlen=5)
    

        print("📦 Loading DL model...")

        model_path = "D:/Mini Project Code/Core/Python_code/deeplearning/ruview_csi_model_grouped.h5"

        self.dl_model = load_model(model_path)

        print("✅ DL Model Loaded")

        # =========================
        # 🖥️ WINDOW SETTINGS
        # =========================
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
        # 🎨 DARK THEME
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

        # =========================
        # 🔷 OUTPUT CARDS
        # =========================
        output_layout = QtWidgets.QHBoxLayout()

        self.threshold_card = self.create_card("THRESHOLD")
        self.ml_card = self.create_card("ML MODEL")
        self.dl_card = self.create_card("DEEP LEARNING")

        self.threshold_card.output_label.setText("Waiting...")

        output_layout.addWidget(self.threshold_card)
        output_layout.addWidget(self.ml_card)
        output_layout.addWidget(self.dl_card)

        self.main_layout.addLayout(output_layout)

        # =========================
        # 🔷 FEATURE SELECTOR
        # =========================
        feature_layout = QtWidgets.QHBoxLayout()

        self.feature_dropdown = QtWidgets.QComboBox()
        self.feature_list = [
            "rms",
            "variance",
            "mean",
            "std",
            "max",
            "min",
            "energy",
            "doppler_frequency",
            "peak_freq",
            "entropy",
            "skewness",
            "kurtosis"
        ]

        self.feature_dropdown.addItems(self.feature_list)

        feature_layout.addWidget(QtWidgets.QLabel("Feature:"))
        feature_layout.addWidget(self.feature_dropdown)
        feature_layout.addStretch()

        self.main_layout.addLayout(feature_layout)

        # =========================
        # 📊 GRAPH
        # =========================
        self.plot = pg.PlotWidget(title="Feature Graph")
        self.plot.setBackground('#121212')
        self.curve = self.plot.plot(pen=pg.mkPen(color='cyan', width=2))

        self.main_layout.addWidget(self.plot)

        # =========================
        # 📡 RAW DATA BOXES
        # =========================
        raw_layout = QtWidgets.QHBoxLayout()

        self.raw1_text = QtWidgets.QTextEdit()
        self.raw2_text = QtWidgets.QTextEdit()

        for t in [self.raw1_text, self.raw2_text]:
            t.setReadOnly(True)
            t.setMinimumHeight(250)

        raw_layout.addWidget(self.wrap("Raw Input 1", self.raw1_text))
        raw_layout.addWidget(self.wrap("Raw Input 2", self.raw2_text))

        self.main_layout.addLayout(raw_layout)

        # =========================
        # 📊 HISTORY DATA
        # =========================
        self.history = {k: [] for k in self.feature_list}

        # =========================
        # ⏱️ TIMER
        # =========================
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_output)

        self.start_btn.clicked.connect(lambda: self.timer.start(100))
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
    
    def parse_line(self, line):
        try:
            if "DATA" not in line:
                return None, None, None

            rssi = int(re.search(r'RSSI[:=]?\s*(-?\d+)', line).group(1))
            raw = list(map(int, re.search(r'DATA:\[(.*?)\]', line).group(1).split(',')))

            i = np.array(raw[::2], dtype=float)
            q = np.array(raw[1::2], dtype=float)

            i[i == 0] = 0.001
            q[q == 0] = 0.001

            TARGET_LEN = 128

            i = np.pad(i[:TARGET_LEN], (0, max(0, TARGET_LEN - len(i))))
            q = np.pad(q[:TARGET_LEN], (0, max(0, TARGET_LEN - len(q))))

            amp = np.sqrt(i**2 + q**2)
            phase = np.arctan2(q, i)

            return amp, phase, rssi

        except:
            return None, None, None

    # =========================
    def update_output(self):

        # ===== READ SERIAL =====
        line1 = self.ser1.readline().decode(errors='ignore').strip()
        line2 = self.ser2.readline().decode(errors='ignore').strip()

        if not line1 or not line2:
            return

        # ===== SHOW RAW DATA =====
        self.raw1_text.append(self.format_raw(line1, "📡 SENSOR 1 (COM5)"))
        self.raw2_text.append(self.format_raw(line2, "📡 SENSOR 2 (COM6)"))

        # ===== LIMIT RAW TEXT =====
        for t in [self.raw1_text, self.raw2_text]:
            if t.document().blockCount() > 100:
                cursor = t.textCursor()
                cursor.movePosition(cursor.Start)
                cursor.select(cursor.LineUnderCursor)
                cursor.removeSelectedText()

        # ===== PARSE =====
        parsed1 = self.parse_line(line1)
        parsed2 = self.parse_line(line2)

        if parsed1 is None or parsed2 is None:
            return

        amp1, phase1, rssi1 = parsed1
        amp2, phase2, rssi2 = parsed2

        # ===== SENSOR FUSION =====
        w1 = abs(rssi1)
        w2 = abs(rssi2)

        if (w1 + w2) == 0:
            return

        amp = (w1 * amp1 + w2 * amp2) / (w1 + w2)
        phase = (w1 * phase1 + w2 * phase2) / (w1 + w2)

        # ===== PREPROCESS =====
        amp, phase = preprocess_frame(amp, phase)

        # ===== 🔥 RUN DL (INDEPENDENT) =====
        self.run_dl_model(amp)

        # ===== STORE WINDOWS =====
        self.amp_w1.append(amp)
        self.phase_w1.append(phase)
        self.rssi_w1.append((rssi1 + rssi2) / 2)

        # ===== WAIT WINDOW =====
        if len(self.amp_w1) < self.WINDOW_SIZE:
            return

        # ===== FILTER =====
        amp_f1, phase_f1, sig1 = filter_window(self.amp_w1, self.phase_w1)

        # ===== NORMALIZE =====
        sig1 = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-6)

        # ===== CALIBRATION =====
        if not self.calibrated:
            self.calibration_data.append(sig1)

            self.threshold_card.output_label.setText(
                f"Calibrating...\n{len(self.calibration_data)}/{self.CALIBRATION_FRAMES}"
            )

            if len(self.calibration_data) >= self.CALIBRATION_FRAMES:
                self.detector.calibrate(self.calibration_data)
                self.calibrated = True

            return

        # ===== THRESHOLD MODEL =====
        result = self.detector.detect(sig1)

        presence = result["presence"]
        confidence = result["confidence"]
        var_r = result.get("var_ratio", 0)

        if presence:
            th_text = f"🟢 HUMAN ({confidence:.1f}%)"
            th_color = "#00ff00"
        else:
            th_text = f"⚫ EMPTY ({confidence:.1f}%)"
            th_color = "#888888"

        self.threshold_card.output_label.setStyleSheet(
            f"color: {th_color}; font-size:18px;"
        )

        self.threshold_card.output_label.setText(
            f"{th_text}\nVAR: {var_r:.2f}"
        )

        # ===== FEATURE EXTRACTION =====
        phase_sig1 = np.mean(phase_f1, axis=0)
        f1 = extract_features(sig1, phase_sig1, self.rssi_w1, self.FS)

        # ===== GRAPH =====
        selected = self.feature_dropdown.currentText()

        # Get ML feature list
        f1_sel = select_12_features(f1)

        # Map selected feature to index
        if selected in self.feature_list:
            idx = self.feature_list.index(selected)
            value = f1_sel[idx]
        else:
            value = 0

        self.history[selected].append(value)
        if len(self.history[selected]) > 100:
            self.history[selected].pop(0)

        self.curve.setData(self.history[selected])

        # ===== ML MODEL =====
    
        features = f1_sel

        xgb_pred, xgb_prob, ens_pred, ens_prob = predict_all(features)

        ml_pred = ens_pred
        ml_conf = ens_prob

        if ml_pred == 1:
            ml_label = "🟢 HUMAN"
            ml_color = "#00ff00"
        else:
            ml_label = "⚫ EMPTY"
            ml_color = "#888888"

        if ml_conf > 0.8:
            conf_text = "HIGH"
        elif ml_conf > 0.6:
            conf_text = "MEDIUM"
        else:
            conf_text = "LOW"

        self.ml_card.output_label.setStyleSheet(
            f"color: {ml_color}; font-size:18px;"
        )

        self.ml_card.output_label.setText(
            f"{ml_label} ({ml_conf:.2f})\nConfidence: {conf_text}"
        )

        # ===== AUTO SCROLL =====
        for t in [self.raw1_text, self.raw2_text]:
            t.verticalScrollBar().setValue(
                t.verticalScrollBar().maximum()
            )
        
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
                
    def format_raw(self, line, label):
        try:
            ts = re.search(r'TS:(.*?) ,', line).group(1)
            rssi = re.search(r'RSSI:(-?\d+)', line).group(1)
            data = re.search(r'DATA:\[(.*?)\]', line).group(1)

            preview = data[:80] + "..." if len(data) > 80 else data

            return (
                f"{label}\n"
                f"⏱ {ts}\n"
                f"📶 RSSI: {rssi}\n"
                f"[{preview}]\n"
                f"{'-'*40}"
            )

        except:
            return line
        
    def run_dl_model(self, amp):

        subsample = amp[::self.SUBCARRIER_STEP]
        self.dl_buffer.append(subsample)

        if len(self.dl_buffer) < self.DL_WINDOW:
            self.dl_card.output_label.setText("Collecting...")
            return

        sample = np.array(self.dl_buffer)

        doppler = compute_doppler(sample, self.FS)
        doppler = normalize(doppler)
        doppler = np.expand_dims(doppler, axis=0)

        if doppler.shape[1:] != dl_model.input_shape[1:]:
            print("⚠️ DL Shape mismatch:", doppler.shape)
            self.dl_card.output_label.setText("Shape Error")
            return

        pred = dl_model.predict(doppler, verbose=0)

        label = np.argmax(pred)
        confidence = np.max(pred)

        self.dl_pred_buffer.append(label)
        final_label = max(set(self.dl_pred_buffer), key=self.dl_pred_buffer.count)

        if final_label == 0:
            state = "🟢 EMPTY"
            color = "#00ff00"
        elif final_label == 1:
            state = "🧍 STANDING"
            color = "#ffaa00"
        else:
            state = "🚶 MOVING"
            color = "#ff4444"

        self.dl_card.output_label.setStyleSheet(
            f"color: {color}; font-size:18px;"
        )

        self.dl_card.output_label.setText(
            f"{state}\nConf: {confidence:.2f}"
        )


# ===== RUN =====
def run():
    app = QtWidgets.QApplication(sys.argv)
    window = ISAC_GUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()