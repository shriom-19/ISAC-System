import sys
import serial
import re
import numpy as np
from collections import deque
import tf2onnx

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from filtering import filter_window
from features import extract_features
from presence_detection import PresenceDetector
from Doppler import DopplerMotionDetector

# =========================
# 🚀 WORKER THREAD
# =========================
class DataWorker(QtCore.QThread):
    data_signal = QtCore.pyqtSignal(dict)
    raw_signal = QtCore.pyqtSignal(str, str)

    def __init__(self):
        super().__init__()

        self.running = False

        # =========================
        # 🔌 SERIAL
        # =========================
        self.ser1 = None
        self.ser2 = None

        # =========================
        # 🪟 WINDOW SIZE
        # =========================
        self.window_size = 30   

        # =========================
        # 📦 SIGNAL BUFFERS
        # =========================
        self.amp1, self.phase1, self.rssi1 = [], [], []
        self.amp2, self.phase2, self.rssi2 = [], [], []

        # =========================
        # 🤖 ML MODELS
        # =========================
        import pickle

        print("📦 Loading ML models...")

        self.xgb_model = pickle.load(open(
            r"D:\Mini Project Code\Core\Python_code\Machine_learning\xgb_model.pkl", "rb"))

        self.ensemble_model = pickle.load(open(
            r"D:\Mini Project Code\Core\Python_code\Machine_learning\ensemble_model.pkl", "rb"))

        self.xgb_features = pickle.load(open(
            r"D:\Mini Project Code\Core\Python_code\Machine_learning\feature_names_xgboost.pkl", "rb"))

        self.ens_features = pickle.load(open(
            r"D:\Mini Project Code\Core\Python_code\Machine_learning\feature_names_ensemble.pkl", "rb"))

        print("✅ ML Models Loaded")

        # =========================
        # 🧠 DL MODEL (DISABLED)
        # =========================
        self.dl_model = None
        print("⚠️ DL Disabled")

        # =========================
        # 🎯 THRESHOLD MODEL (REAL)
        # =========================
        from presence_detection import PresenceDetector
        from Doppler import DopplerMotionDetector

        self.presence_detector = PresenceDetector()
        self.doppler_detector = DopplerMotionDetector(fs=20)

        # 🔥 buffers (same as standalone code)
        self.threshold_buffer = deque(maxlen=25)
        self.doppler_buffer = deque(maxlen=10)

        # 🔥 calibration system
        self.calibration_data = []
        self.calibrated = False
        self.calibration_frames = 20

        # =========================
        # ⚡ PERFORMANCE CONTROL
        # =========================
        self.frame_count = 0
        
    def start_serial(self):
        self.ser1 = serial.Serial("COM5", 115200, timeout=1)
        self.ser2 = serial.Serial("COM7", 115200, timeout=1)
        self.running = True

    def stop_serial(self):
        self.running = False
        if self.ser1:
            self.ser1.close()
        if self.ser2:
            self.ser2.close()

    def parse(self, line):
        try:
            rssi = int(re.search(r"RSSI:(-?\d+)", line).group(1))
            data = list(map(int, re.search(r"DATA:\[(.*?)\]", line).group(1).split(',')))

            # ✅ Ensure correct length
            if len(data) < 256:
                return None, None, None

            data = data[:256]  # trim extra if any

            amp = np.array(data[:128], dtype=float)
            phase = np.array(data[128:256], dtype=float)

            return amp, phase, rssi

        except:
            return None
        
    def reorder(self, features_dict, feature_names):
        return [features_dict.get(f, 0) for f in feature_names]

    def run(self):
        while self.running:
            try:
                # =========================
                # 📥 READ SERIAL
                # =========================
                line1 = self.ser1.readline().decode(errors="ignore").strip()
                line2 = self.ser2.readline().decode(errors="ignore").strip()

                if not line1 or not line2:
                    continue

                self.raw_signal.emit(line1, line2)

                # =========================
                # 📡 PARSE
                # =========================
                p1 = self.parse(line1)
                p2 = self.parse(line2)

                if p1 is None or p2 is None:
                    continue

                amp1, phase1, rssi1 = p1
                amp2, phase2, rssi2 = p2

                if len(amp1) != 128 or len(amp2) != 128:
                    continue

                # =========================
                # 📦 STORE WINDOW
                # =========================
                self.amp1.append(amp1)
                self.phase1.append(phase1)
                self.rssi1.append(rssi1)

                self.amp2.append(amp2)
                self.phase2.append(phase2)
                self.rssi2.append(rssi2)

                if len(self.amp1) > self.window_size:
                    self.amp1.pop(0)
                    self.phase1.pop(0)
                    self.rssi1.pop(0)

                if len(self.amp2) > self.window_size:
                    self.amp2.pop(0)
                    self.phase2.pop(0)
                    self.rssi2.pop(0)

                if len(self.amp1) < self.window_size:
                    continue

                # =========================
                # 🔧 FILTER
                # =========================
                _, _, sig1 = filter_window(self.amp1, self.phase1)
                _, _, sig2 = filter_window(self.amp2, self.phase2)
          
                # =========================
                # 📊 FEATURES (FIXED)
                # =========================

                # =========================
                # 🔧 FILTER (LIKE ML FILE)
                # =========================
                amp_f1, phase_f1, final_signal_1 = filter_window(self.amp1, self.phase1)
                amp_f2, phase_f2, final_signal_2 = filter_window(self.amp2, self.phase2)

                # =========================
                # 📡 PHASE FIX
                # =========================
                phase_signal_1 = np.mean(phase_f1, axis=0)
                phase_signal_2 = np.mean(phase_f2, axis=0)

                # =========================
                # 📊 FEATURES (CORRECT)
                # =========================
                f1 = extract_features(final_signal_1, phase_signal_1, self.rssi1, 20)
                f2 = extract_features(final_signal_2, phase_signal_2, self.rssi2, 20)
                # =========================
                # 🎯 THRESHOLD MODEL (REAL)
                # =========================

                # combine signals (like your standalone code)
                combined = (amp1 + amp2) / 2
                self.threshold_buffer.append(combined)

                if len(self.threshold_buffer) < 25:
                    threshold_result = "🎯 CALIBRATING..."
                else:
                    try:
                        # normalize signal
                        signal = np.array(self.threshold_buffer)
                        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

                        # ================= CALIBRATION =================
                        if not self.calibrated:
                            self.calibration_data.append(signal)

                            if len(self.calibration_data) >= self.calibration_frames:
                                self.presence_detector.calibrate(self.calibration_data)
                                self.calibrated = True

                            threshold_result = f"🎯 CALIBRATING ({len(self.calibration_data)}/{self.calibration_frames})"

                        else:
                            # ================= PRESENCE =================
                            result = self.presence_detector.detect(signal)

                            presence = result["presence"]
                            confidence = result["confidence"]

                            # ================= DOPPLER =================
                            phase_diff = np.diff(np.mean(signal, axis=1))
                            doppler = np.mean(np.abs(phase_diff)) * 20

                            self.doppler_buffer.append(doppler)
                            doppler_final = np.mean(self.doppler_buffer)

                            # ================= FINAL OUTPUT =================
                            if presence and doppler_final > 0.08:
                                threshold_result = f"🟢 HUMAN + MOTION ({confidence:.1f}%)"

                            elif presence:
                                threshold_result = f"🟡 HUMAN PRESENT ({confidence:.1f}%)"

                            elif doppler_final > 0.08:
                                threshold_result = f"🔵 MOTION ONLY ({confidence:.1f}%)"

                            else:
                                threshold_result = f"⚫ EMPTY ROOM ({confidence:.1f}%)"

                    except Exception as e:
                        threshold_result = "🎯 ERROR"

                # =========================
                # 🤖 ML MODEL
                # =========================
                self.dl_buffer = deque(maxlen=30)
                # =========================
                # 🤖 ML MODELS (SEPARATE)
                # =========================
                # =========================
                # 🤖 ML MODELS (FINAL FORMAT)
                # =========================
                try:
                    # =========================
                    # 🎯 FEATURE SELECTION (LIKE ML FILE)
                    # =========================
                    f1_sel = self.select_12_features(f1)
                    f2_sel = self.select_12_features(f2)

                    features = f1_sel + f2_sel

                    X = np.array(features).reshape(1, -1)

                    # 🔵 XGBOOST
                    xgb_pred = self.xgb_model.predict(X)[0]
                    xgb_prob = self.xgb_model.predict_proba(X)[0][1]

                    if xgb_pred == 1:
                        xgb_result = f"🔵 XGBOOST  → HUMAN (Confidence: {xgb_prob:.2f})"
                    else:
                        xgb_result = f"🔵 XGBOOST  → EMPTY (Confidence: {1 - xgb_prob:.2f})"

                    # 🟢 ENSEMBLE
                    ens_pred = self.ensemble_model.predict(X)[0]
                    ens_prob = self.ensemble_model.predict_proba(X)[0][1]

                    if ens_pred == 1:
                        ens_result = f"🟢 ENSEMBLE → HUMAN (Confidence: {ens_prob:.2f})"
                    else:
                        ens_result = f"🟢 ENSEMBLE → EMPTY (Confidence: {1 - ens_prob:.2f})"

                except Exception:
                    xgb_result = "🤖 XGB ERROR"
                    ens_result = "🤖 ENS ERROR"
                # =========================
                # 🧠 DL MODEL (OPTIMIZED)
                # =========================
                combined = (amp1 + amp2) / 2
                self.dl_buffer.append(combined)
                

                self.frame_count += 1

                if len(self.dl_buffer) < 30:
                    dl_result = "🧠 WAIT"

                elif self.frame_count % 3 == 0:   # 🔥 KEY OPTIMIZATION
                    try:
                        sample = np.array(self.dl_buffer)

                        sample = (sample - np.mean(sample)) / (np.std(sample) + 1e-6)

                        sample = np.expand_dims(sample, axis=0)
                        sample = np.expand_dims(sample, axis=-1)

                        pred = self.dl_model.predict(sample, verbose=0)

                        label = np.argmax(pred)
                        conf = np.max(pred)

                        labels = ["EMPTY 🟢", "STANDING 🧍", "MOVING 🚶"]

                        dl_result = "🧠 DL DISABLED"

                    except Exception:
                        dl_result = "🧠 ERROR"

                else:
                    dl_result = "🧠 PROCESSING..."

                # =========================
                # 📤 SEND
                # =========================
                self.data_signal.emit({
                    "features": f1,
                    "threshold": threshold_result,
                    "xgb": xgb_result,
                    "ensemble": ens_result
                })

            except Exception as e:
                print("Thread Error:", e)
                
    def select_12_features(self, f):
        keys = [
            "variance", "rms", "mean_abs_diff", "peak_to_peak",
            "peak_freq", "spectral_energy", "doppler_frequency",
            "phase_std", "phase_coherence", "breathing_period",
            "rssi_var", "rssi_diff"
        ]
        return [f.get(k, 0) for k in keys]
            
# =========================
# 🖥 MAIN GUI
# =========================
class ISAC_GUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ISAC System")
        self.resize(1200, 800)

        # 🌙 DARK MODE
        self.setStyleSheet("""
            QWidget { background-color: #121212; color: #ffffff; }
            QTextEdit { background-color: #1e1e1e; }
            QLabel { font-size: 14px; }
        """)

        pg.setConfigOption('background', '#121212')
        pg.setConfigOption('foreground', 'w')

        # =========================
        # 📊 FEATURES (GRAPH)
        # =========================
        self.features_list = [
            "variance_1", "rms_1", "peak_freq_1",
            "spectral_energy_1", "doppler_frequency_1", "phase_coherence_1",
            "variance_2", "rms_2", "peak_freq_2",
            "spectral_energy_2", "doppler_frequency_2", "phase_coherence_2"
        ]

        self.buffers = {k: deque(maxlen=80) for k in self.features_list}

        # =========================
        # 🖥 MAIN LAYOUT
        # =========================
        layout = QtWidgets.QVBoxLayout()

        # =========================
        # 🔘 BUTTONS
        # =========================
        btn_layout = QtWidgets.QHBoxLayout()

        self.start_btn = QtWidgets.QPushButton("▶ Start")
        self.stop_btn = QtWidgets.QPushButton("⏹ Stop")

        btn_layout.addStretch()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        # =========================
        # 📥 RAW DATA BOXES
        # =========================
        self.text1 = QtWidgets.QTextEdit()
        self.text2 = QtWidgets.QTextEdit()

        raw_layout = QtWidgets.QHBoxLayout()
        raw_layout.addWidget(self.text1)
        raw_layout.addWidget(self.text2)

        layout.addLayout(raw_layout)

        # =========================
        # 🎛 DROPDOWN
        # =========================
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(self.features_list)

        layout.addWidget(self.combo)

        # =========================
        # 📈 GRAPH
        # =========================
        self.plot = pg.PlotWidget(title="Feature Viewer")
        self.curve = self.plot.plot(pen='y')

        layout.addWidget(self.plot)

        # =========================
        # 🧠 MODEL OUTPUT BOXES
        # =========================
        output_layout = QtWidgets.QHBoxLayout()

        self.threshold_box = QtWidgets.QLabel("🎯 Threshold: ---")
        self.ml_box = QtWidgets.QLabel("🌲 XGBoost: ---")
        self.dl_box = QtWidgets.QLabel("🤖 Ensemble: ---")

        for box in [self.threshold_box, self.ml_box, self.dl_box]:
            box.setStyleSheet("""
                background-color: #1e1e1e;
                padding: 12px;
                border-radius: 8px;
            """)
            box.setAlignment(QtCore.Qt.AlignCenter)
            output_layout.addWidget(box)

        layout.addLayout(output_layout)

        # =========================
        # APPLY LAYOUT
        # =========================
        self.setLayout(layout)

        # =========================
        # THREAD
        # =========================
        self.worker = DataWorker()
        self.worker.data_signal.connect(self.update_data)
        self.worker.raw_signal.connect(self.update_raw)

        # =========================
        # BUTTON ACTIONS
        # =========================
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

        # =========================
        # GRAPH TIMER
        # =========================
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_graph)
    
    def start(self):
        self.worker.start_serial()
        self.worker.start()
        self.timer.start(100)

    def stop(self):
        self.worker.stop_serial()
        self.worker.quit()
        self.timer.stop()

    def update_data(self, data):

        # =========================
        # 📊 GRAPH DATA (FEATURES)
        # =========================
        f = data["features"]

        # ESP32 #1 features
        self.buffers["variance_1"].append(f["variance"])
        self.buffers["rms_1"].append(f["rms"])
        self.buffers["peak_freq_1"].append(f["peak_freq"])
        self.buffers["spectral_energy_1"].append(f["spectral_energy"])
        self.buffers["doppler_frequency_1"].append(f["doppler_frequency"])
        self.buffers["phase_coherence_1"].append(f["phase_coherence"])

        # ESP32 #2 (optional — if you want same or separate)
        self.buffers["variance_2"].append(f["variance"])
        self.buffers["rms_2"].append(f["rms"])
        self.buffers["peak_freq_2"].append(f["peak_freq"])
        self.buffers["spectral_energy_2"].append(f["spectral_energy"])
        self.buffers["doppler_frequency_2"].append(f["doppler_frequency"])
        self.buffers["phase_coherence_2"].append(f["phase_coherence"])

        # =========================
        # 🧠 MODEL OUTPUT DISPLAY
        # =========================
        self.threshold_box.setText(data["threshold"])
        self.ml_box.setText(data["xgb"])
        self.dl_box.setText(data["ensemble"])

    def update_raw(self, line1, line2):
        # reduce updates
        if self.text1.document().blockCount() > 50:
            self.text1.clear()
        if self.text2.document().blockCount() > 50:
            self.text2.clear()

        self.text1.append(line1)
        self.text2.append(line2)

    def update_graph(self):
        selected = self.combo.currentText()
        data = list(self.buffers[selected])[::2]
        self.curve.setData(data)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ISAC_GUI()
    window.show()
    sys.exit(app.exec_())