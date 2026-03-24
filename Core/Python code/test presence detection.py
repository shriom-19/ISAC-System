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
CALIBRATION_FRAMES = 50

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

        amp = []

        for i in range(0, len(raw_data), 2):
            c = complex(raw_data[i], raw_data[i+1])
            amp.append(abs(c))

        return rssi, amp

    except:
        return None

# ===================== FILTER =====================
def filter_window(amp_buffer):
    data = np.array(amp_buffer)  # (window, subcarriers)

    # 🔥 motion-sensitive signal
    signal = np.std(data, axis=1)

    # 🔥 smoothing (VERY IMPORTANT)
    signal = np.convolve(signal, np.ones(3)/3, mode='same')

    # remove DC
    signal = signal - np.mean(signal)

    return signal

# ===================== FEATURES =====================
def extract_features(signal):
    features = {}

    features["variance"] = np.var(signal)
    features["energy"] = np.sum(signal ** 2)

    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal), d=1/FS)

    idx = np.argmax(fft_vals[1:]) + 1
    features["doppler"] = abs(freqs[idx])

    return features

# ===================== DETECTOR =====================
class PresenceDetector:
    def __init__(self):
        self.baseline_rms = None
        self.baseline_var = None
        self.history = deque(maxlen=5)  # 🔥 stability

    def calibrate(self, signals):
        full = np.concatenate(signals)

        self.baseline_rms = np.sqrt(np.mean(full**2))
        self.baseline_var = np.var(full)

        print("\n✅ CALIBRATION DONE")
        print(f"Baseline RMS: {self.baseline_rms:.3f}")
        print(f"Baseline VAR: {self.baseline_var:.3f}\n")

    def detect(self, signal):
        rms = np.sqrt(np.mean(signal**2))
        var = np.var(signal)

        rms_ratio = rms / (self.baseline_rms + 1e-6)
        var_ratio = var / (self.baseline_var + 1e-6)

        # 🔥 motion strength
        diff = np.diff(signal)
        motion_score = np.mean(np.abs(diff))

        # 🔥 FINAL LOGIC (TUNED)
        raw_presence = (
            var_ratio > 1.3 or
            motion_score > 0.15
        )

        # 🔥 STABILITY (anti flicker)
        self.history.append(1 if raw_presence else 0)
        presence = sum(self.history) >= 3

        confidence = min(100, (var_ratio - 1) * 100)

        return presence, confidence, rms_ratio, var_ratio, motion_score

# ===================== MAIN =====================
ser = connect_serial()

amp_buffer = deque(maxlen=WINDOW_SIZE)
rssi_buffer = deque(maxlen=WINDOW_SIZE)

detector = PresenceDetector()

calibration_data = []
is_calibrated = False

frame = 0

print("🚀 HUMAN DETECTION SYSTEM STARTED\n")

while True:
    try:
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        print("\n📥 RAW:", line)

        parsed = parse_csi(line)

        if not parsed:
            print("⚠️ Skipped")
            continue

        rssi, amp = parsed

        # 🔥 FIX LENGTH
        amp = fix_length(amp)

        # 🔥 REMOVE UNSTABLE SUBCARRIERS
        amp = amp[2:]

        amp_buffer.append(amp)
        rssi_buffer.append(rssi)

        frame += 1
        print(f"📦 Frame {frame} | RSSI: {rssi}")
        print(f"📊 Buffer: {len(amp_buffer)}/{WINDOW_SIZE}")

        if len(amp_buffer) < WINDOW_SIZE:
            continue

        # ================= FILTER =================
        signal = filter_window(amp_buffer)

        print("🔬 Signal:", signal[:5])

        # ================= FEATURES =================
        features = extract_features(signal)

        print("📊 FEATURES")
        print(f"Variance: {features['variance']:.3f}")
        print(f"Doppler: {features['doppler']:.3f}")

        # ================= CALIBRATION =================
        if not is_calibrated:
            print("📍 CALIBRATING...")
            calibration_data.append(signal)

            if len(calibration_data) >= CALIBRATION_FRAMES:
                detector.calibrate(calibration_data)
                is_calibrated = True
                print("🚀 DETECTION STARTED\n")

            continue

        # ================= DETECTION =================
        presence, conf, rms_r, var_r, motion = detector.detect(signal)

        print("\n🧠 METRICS")
        print(f"RMS Ratio: {rms_r:.2f}")
        print(f"VAR Ratio: {var_r:.2f}")
        print(f"Motion: {motion:.3f}")

        print("\n🎯 RESULT")
        if presence:
            print(f"🟢 HUMAN DETECTED | {conf:.1f}%")
        else:
            print(f"⚫ EMPTY ROOM | {conf:.1f}%")

    except Exception as e:
        print("❌ ERROR:", e)