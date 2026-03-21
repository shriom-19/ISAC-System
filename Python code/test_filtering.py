import serial
import re
import numpy as np
from collections import deque
import time

# ===================== CONFIG =====================
PORT = "COM5"
BAUD = 115200

WINDOW_SIZE = 20
FS = 20
CALIBRATION_FRAMES = 20


# ===================== SERIAL =====================
def connect_serial():
    while True:
        try:
            ser = serial.Serial(PORT, BAUD, timeout=1)
            print("✅ Serial Connected")
            return ser
        except:
            print("❌ Waiting for ESP32...")
            time.sleep(2)


# ===================== FIX LENGTH =====================
def fix_length(arr, target_len=128):
    arr = np.array(arr, dtype=float)

    if len(arr) > target_len:
        return arr[:target_len]
    elif len(arr) < target_len:
        return np.pad(arr, (0, target_len - len(arr)))

    return arr


# ===================== PARSER =====================
def parse_csi(line):
    try:
        if "DATA" not in line:
            return None

        rssi_match = re.search(r'RSSI[:=]?\s*(-?\d+)', line)
        rssi = int(rssi_match.group(1)) if rssi_match else 0

        data_match = re.search(r'DATA:\[(.*?)\]', line)
        if not data_match:
            return None

        raw_str = data_match.group(1)
        raw_str = re.sub(r'[^0-9,\-]', '', raw_str)

        raw_data = list(map(int, raw_str.split(',')))

        if len(raw_data) % 2 != 0:
            return None

        amp, phase = [], []

        for i in range(0, len(raw_data), 2):
            c = complex(raw_data[i], raw_data[i+1])
            amp.append(abs(c))
            phase.append(np.angle(c))

        return rssi, amp, phase

    except:
        return None


# ===================== FILTER =====================
def filter_window(amp_buffer):
    data = np.array(amp_buffer)

    signal = np.mean(data, axis=1)

    # ✅ ONLY remove DC (NO normalization)
    signal = signal - np.mean(signal)

    return signal


# ===================== FEATURES =====================
def extract_features(signal):
    features = {}

    features["variance"] = float(np.var(signal))
    features["energy"] = float(np.sum(signal**2))

    diff = np.diff(signal)
    features["mean_abs_diff"] = float(np.mean(np.abs(diff))) if len(diff) > 0 else 0.0

    return features


# ===================== DETECTOR =====================
class PresenceDetector:
    def __init__(self):
        self.base_var = None
        self.base_energy = None
        self.base_diff = None

    def calibrate(self, feature_list):
        self.base_var = np.mean([f["variance"] for f in feature_list])
        self.base_energy = np.mean([f["energy"] for f in feature_list])
        self.base_diff = np.mean([f["mean_abs_diff"] for f in feature_list])

        print("\n✅ CALIBRATION DONE")
        print(f"Base VAR: {self.base_var:.4f}")
        print(f"Base ENERGY: {self.base_energy:.2f}")
        print(f"Base DIFF: {self.base_diff:.4f}\n")

    def detect(self, features):
        var_ratio = features["variance"] / (self.base_var + 1e-6)
        energy_ratio = features["energy"] / (self.base_energy + 1e-6)
        diff_ratio = features["mean_abs_diff"] / (self.base_diff + 1e-6)

        # 🔥 Weighted score
        score = 0.4 * var_ratio + 0.3 * energy_ratio + 0.3 * diff_ratio

        presence = score > 1.15

        confidence = max(0, min(100, (score - 1) * 100))

        return presence, confidence, score, var_ratio, energy_ratio, diff_ratio


# ===================== MAIN =====================
ser = connect_serial()

amp_buffer = deque(maxlen=WINDOW_SIZE)
rssi_buffer = deque(maxlen=WINDOW_SIZE)

detector = PresenceDetector()

calibration_features = []
is_calibrated = False

frame = 0

print("🚀 DETECTION SYSTEM STARTED\n")

while True:
    try:
        line = ser.readline().decode(errors="ignore").strip()

        print("\n🔹 RAW:", line)

        if not line or "DATA" not in line:
            continue

        result = parse_csi(line)

        if not result:
            print("❌ Parsing failed")
            continue

        rssi, amp, phase = result

        amp = fix_length(amp)

        amp_buffer.append(amp)
        rssi_buffer.append(rssi)

        frame += 1
        print(f"📦 Frame {frame} | RSSI: {rssi}")

        if len(amp_buffer) < WINDOW_SIZE:
            print("⏳ Waiting...")
            continue

        signal = filter_window(amp_buffer)
        features = extract_features(signal)

        print("📊 Features:", features)

        # ================= CALIBRATION =================
        if not is_calibrated:
            print("📍 CALIBRATING...")
            calibration_features.append(features)

            if len(calibration_features) >= CALIBRATION_FRAMES:
                detector.calibrate(calibration_features)
                is_calibrated = True
                print("🚀 DETECTION STARTED\n")

            continue

        # ================= DETECTION =================
        presence, conf, score, vr, er, dr = detector.detect(features)

        print(f"Score: {score:.2f}")
        print(f"VAR: {vr:.2f} | ENERGY: {er:.2f} | DIFF: {dr:.2f}")

        if presence:
            print(f"🟢 HUMAN DETECTED | {conf:.1f}%")
        else:
            print(f"⚫ EMPTY | {conf:.1f}%")

    except Exception as e:
        print("❌ ERROR:", e)