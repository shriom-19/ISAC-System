import os
import sys
import numpy as np
import pickle
from collections import deque
import re
import serial

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add parent folder (Python code/) to path
sys.path.append(os.path.dirname(BASE_DIR))
print("BASE_DIR:", BASE_DIR)
# ===================== IMPORT YOUR MODULES =====================
from filtering import preprocess_frame, filter_window
from features import extract_features
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# ===================== LOAD MODEL =====================
xgb_model = pickle.load(open(os.path.join(BASE_DIR, "xgb_model.pkl"), "rb"))
xgb_feature = pickle.load(open(os.path.join(BASE_DIR, "feature_names_xgboost.pkl"), "rb"))

ensemble_model = pickle.load(open(os.path.join(BASE_DIR, "ensemble_model.pkl"), "rb"))
ensemble_feature = pickle.load(open(os.path.join(BASE_DIR, "feature_names_ensemble.pkl"), "rb"))

# ===================== CONFIG =====================
PORT1 = "COM5"
PORT2 = "COM6"
BAUD = 115200

WINDOW_SIZE = 30
FS = 20

# ===================== LOAD MODEL =====================

# ===================== SERIAL INIT =====================
ser1 = serial.Serial(PORT1, BAUD, timeout=1)
ser2 = serial.Serial(PORT2, BAUD, timeout=1)

# ===================== WINDOWS =====================
amp_window_1 = deque(maxlen=WINDOW_SIZE)
phase_window_1 = deque(maxlen=WINDOW_SIZE)
rssi_window_1 = deque(maxlen=WINDOW_SIZE)

amp_window_2 = deque(maxlen=WINDOW_SIZE)
phase_window_2 = deque(maxlen=WINDOW_SIZE)
rssi_window_2 = deque(maxlen=WINDOW_SIZE)

# ===================== SMOOTHING =====================
pred_buffer = deque(maxlen=10)

def smooth(pred):
    pred_buffer.append(pred)
    return int(np.mean(pred_buffer) > 0.6)

# ===================== PARSER =====================
def parse_line(line):
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

# ===================== FEATURE SELECT =====================
def select_12_features(features_dict):
    keys = [
        "variance", "rms", "mean_abs_diff", "peak_to_peak",
        "peak_freq", "spectral_energy", "doppler_frequency",
        "phase_std", "phase_coherence", "breathing_period",
        "rssi_var", "rssi_diff"
    ]
    return [features_dict.get(k, 0) for k in keys]

# ===================== PREDICTION =====================

def reorder(features_dict, feature_names):
    return [features_dict.get(f, 0) for f in feature_names]

def predict_all(features):

    X = np.array(features).reshape(1, -1)

    xgb_pred = xgb_model.predict(X)[0]
    xgb_prob = xgb_model.predict_proba(X)[0][1]

    ens_pred = ensemble_model.predict(X)[0]
    ens_prob = ensemble_model.predict_proba(X)[0][1]

    return xgb_pred, xgb_prob, ens_pred, ens_prob

# ===================== MAIN =====================
def main():
    print("🚀 Real-Time Human Detection Started...\n")

    try:
        while True:

            # ===== READ SERIAL =====
            line1 = ser1.readline().decode(errors='ignore').strip()
            line2 = ser2.readline().decode(errors='ignore').strip()

            # Debug (optional)
            print("RAW1:", line1)
            print("RAW2:", line2)
           

            # ===== PARSE =====
            amp1, phase1, rssi1 = parse_line(line1)
            amp2, phase2, rssi2 = parse_line(line2)
            print("Parsed1:", amp1 is not None)
            print("Parsed2:", amp2 is not None)

            # Skip invalid data
            if amp1 is None or amp2 is None:
                continue

            # ===== PREPROCESS =====
            amp1, phase1 = preprocess_frame(amp1, phase1)
            amp2, phase2 = preprocess_frame(amp2, phase2)

            # ===== STORE WINDOW =====
            amp_window_1.append(amp1)
            phase_window_1.append(phase1)
            rssi_window_1.append(rssi1)

            amp_window_2.append(amp2)
            phase_window_2.append(phase2)
            rssi_window_2.append(rssi2)

            # ===== WAIT FOR BOTH WINDOWS =====
            if len(amp_window_1) < WINDOW_SIZE or len(amp_window_2) < WINDOW_SIZE:
                continue

            # ===== FILTER =====
            amp_f1, phase_f1, final_signal_1 = filter_window(amp_window_1, phase_window_1)
            amp_f2, phase_f2, final_signal_2 = filter_window(amp_window_2, phase_window_2)

            # ===== PHASE FIX =====
            phase_signal_1 = np.mean(phase_f1, axis=0)
            phase_signal_2 = np.mean(phase_f2, axis=0)

            # ===== FEATURE EXTRACTION =====
            f1 = extract_features(final_signal_1, phase_signal_1, rssi_window_1, FS)
            f2 = extract_features(final_signal_2, phase_signal_2, rssi_window_2, FS)

            # ===== DEBUG FEATURES =====
            # print("F1:", f1)
            # print("F2:", f2)

            # ===== MODEL PREDICTION =====
            # ===== FEATURE SELECTION (MUST MATCH DATASET BUILDER) =====
            f1_sel = select_12_features(f1)
            f2_sel = select_12_features(f2)

            features = f1_sel + f2_sel   # 🔥 FINAL INPUT (24 features)

            # ===== MODEL PREDICTION =====
            xgb_pred, xgb_prob, ens_pred, ens_prob = predict_all(features)

            # ===== SMOOTHING (on ensemble) ====

            # ===== OUTPUT =====
            print("\n==============================")

            # 🔵 XGBOOST
            if xgb_pred == 1:
                print(f"🔵 XGBOOST  → HUMAN (Confidence: {xgb_prob:.2f})")
            else:
                print(f"🔵 XGBOOST  → EMPTY (Confidence: {1 - xgb_prob:.2f})")

            # 🟢 ENSEMBLE
            if ens_pred == 1:
                print(f"🟢 ENSEMBLE → HUMAN (Confidence: {ens_prob:.2f})")
            else:
                print(f"🟢 ENSEMBLE → EMPTY (Confidence: {1 - ens_prob:.2f})")

            print("------------------------------")

            # 🔥 FINAL DECISION (fusion)
            final_prob = 0.6 * ens_prob + 0.4 * xgb_prob
            final_pred = int(final_prob > 0.5)

            if final_pred == 1:
                print("🔥 FINAL → HUMAN DETECTED")
            else:
                print("🟢 FINAL → EMPTY ROOM")

            print("==============================\n")

    except KeyboardInterrupt:
        print("\n🛑 Stopped")

        ser1.close()
        ser2.close()
# ===================== RUN =====================
if __name__ == "__main__":
    main()