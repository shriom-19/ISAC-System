import serial
import re
import numpy as np
from collections import deque

# 🔥 IMPORT YOUR FILES
from filtering import preprocess_frame, filter_window
from features import extract_features
from presence_detection import PresenceDetector

# ================= CONFIG =================
PORT = "COM5"
BAUD = 115200

WINDOW_SIZE = 20
CALIBRATION_WINDOWS = 30
TARGET_LEN = 128   # 🔥 FIXED LENGTH

print("🔌 Opening Serial Port...")
ser = serial.Serial(PORT, BAUD, timeout=1)

# Buffers
amp_window = deque(maxlen=WINDOW_SIZE)
phase_window = deque(maxlen=WINDOW_SIZE)
rssi_window = deque(maxlen=WINDOW_SIZE)

detector = PresenceDetector()
calibration_data = []
calibrated = False

# ================= PARSER =================
def parse_line(line):
    try:
        # ✅ MATCH YOUR FORMAT
        rssi_match = re.search(r"RSSI:(-?\d+)", line)
        csi_match = re.search(r"DATA:\[(.*)\]", line)

        if not rssi_match or not csi_match:
            return None

        rssi = int(rssi_match.group(1))
        csi_raw = list(map(int, csi_match.group(1).split(',')))

        # Must be even
        if len(csi_raw) % 2 != 0:
            return None

        i = np.array(csi_raw[::2], dtype=float)
        q = np.array(csi_raw[1::2], dtype=float)

        # 🔥 REPLACE ZEROS (instead of removing)
        i[i == 0] = 0.001
        q[q == 0] = 0.001

        # 🔥 FORCE FIXED LENGTH (CRITICAL FIX)
        if len(i) >= TARGET_LEN:
            i = i[:TARGET_LEN]
            q = q[:TARGET_LEN]
        else:
            pad_len = TARGET_LEN - len(i)
            i = np.pad(i, (0, pad_len))
            q = np.pad(q, (0, pad_len))

        # Convert
        amp = np.sqrt(i**2 + q**2)
        phase = np.arctan2(q, i)

        return amp, phase, rssi

    except Exception as e:
        print("❌ PARSE ERROR:", e)
        return None


print("🚀 FULL DEBUG HUMAN DETECTION STARTED\n")

# ================= MAIN LOOP =================
while True:
    try:
        line = ser.readline().decode(errors='ignore').strip()

        # 🔥 SHOW RAW DATA
        if line:
            print("\n📥 RAW LINE:", line)

        if not line:
            continue

        parsed = parse_line(line)

        if parsed is None:
            print("⚠️ Skipped (parse failed)")
            continue

        amp, phase, rssi = parsed

        print("\n================ NEW FRAME =================")

        # ================= RAW =================
        print(f"📡 RSSI: {rssi}")
        print(f"AMP size: {len(amp)}")
        print(f"AMP (first 5): {amp[:5]}")

        # ================= PREPROCESS =================
        amp, phase = preprocess_frame(amp, phase)

        print("\n🧹 AFTER PREPROCESS")
        print(f"AMP (first 5): {amp[:5]}")
        print(f"PHASE (first 5): {phase[:5]}")

        # Store
        amp_window.append(amp)
        phase_window.append(phase)
        rssi_window.append(rssi)

        print(f"\n📦 Window size: {len(amp_window)}/{WINDOW_SIZE}")

        if len(amp_window) < WINDOW_SIZE:
            continue

        # ================= FILTER =================
        amp_f, phase_f, final_signal = filter_window(
            list(amp_window),
            list(phase_window)
        )

        print("\n🔍 AFTER FILTER")
        print(f"Final signal length: {len(final_signal)}")
        print(f"FINAL SIGNAL (first 5): {final_signal[:5]}")

        # ================= CALIBRATION =================
        if not calibrated:
            calibration_data.append(final_signal)

            print(f"\n📊 CALIBRATING {len(calibration_data)}/{CALIBRATION_WINDOWS}")

            if len(calibration_data) >= CALIBRATION_WINDOWS:
                detector.calibrate(calibration_data)
                calibrated = True
                print("\n🎯 CALIBRATION DONE — ENTER ROOM")

            continue

        # ================= FEATURES =================
        features = extract_features(
            final_signal,
            np.mean(phase_f, axis=0),
            list(rssi_window)
        )

        print("\n📊 FEATURES")
        for k, v in features.items():
            print(f"{k}: {v:.4f}")

        # ================= DETECTION =================
        result = detector.detect(final_signal, sensitivity="medium")

        print("\n🧠 DETECTION METRICS")
        print(f"RMS Ratio: {result['rms_ratio']:.3f}")
        print(f"VAR Ratio: {result['var_ratio']:.3f}")

        # ================= FINAL RESULT =================
        print("\n🎯 FINAL RESULT")

        if result.get("presence"):
            print(f"🟢 HUMAN DETECTED (Confidence: {result['confidence']:.1f}%)")
        else:
            print(f"⚫ EMPTY ROOM (Confidence: {result['confidence']:.1f}%)")

        print("============================================")

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
        break

    except Exception as e:
        print("❌ ERROR:", e)