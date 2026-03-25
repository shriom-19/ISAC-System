# =========================
# 📥 IMPORTS
# =========================
import numpy as np
import serial
import re
import time
from collections import deque

from presence_detection import PresenceDetector
from Doppler import DopplerMotionDetector
from filtering import preprocess_frame, filter_window

# =========================
# ⚙️ CONFIG
# =========================
PORT1 = "COM5"
PORT2 = "COM7"
BAUD = 115200

WINDOW_SIZE = 25
FS = 20
CALIBRATION_FRAMES = 30   # 🔥 reduced + stable
TARGET_LEN = 128

# =========================
# 🔌 SERIAL CONNECTION
# =========================
def connect_serial(port):
    while True:
        try:
            ser = serial.Serial(port, BAUD, timeout=1)
            print(f"✅ Connected to {port}")
            return ser
        except Exception as e:
            print(f"❌ Waiting for {port}...", e)
            time.sleep(2)

# =========================
# 📡 PARSER (SAFE)
# =========================
def parse_line(line):
    try:
        print("📥 RAW:", line)

        if "DATA" not in line:
            print("⚠️ No DATA found")
            return None

        rssi_match = re.search(r'RSSI[:=]?\s*(-?\d+)', line)
        data_match = re.search(r'DATA:\[(.*?)\]', line)

        if not rssi_match or not data_match:
            print("❌ Regex failed")
            return None

        rssi = int(rssi_match.group(1))
        raw = list(map(int, data_match.group(1).split(',')))

        if len(raw) < 10:
            print("❌ CSI too short")
            return None

        i = np.array(raw[::2], dtype=float)
        q = np.array(raw[1::2], dtype=float)

        i[i == 0] = 0.001
        q[q == 0] = 0.001

        i = np.pad(i[:TARGET_LEN], (0, max(0, TARGET_LEN - len(i))))
        q = np.pad(q[:TARGET_LEN], (0, max(0, TARGET_LEN - len(q))))

        amp = np.sqrt(i**2 + q**2)
        phase = np.arctan2(q, i)
        iq = np.stack((i, q), axis=1)

        print("✅ Parsed OK")
        return amp, phase, rssi, iq

    except Exception as e:
        print("❌ Parse Error:", e)
        return None

# =========================
# 🚀 INIT
# =========================
ser1 = connect_serial(PORT1)
ser2 = connect_serial(PORT2)

amp_buffer = deque(maxlen=WINDOW_SIZE)
phase_buffer = deque(maxlen=WINDOW_SIZE)
iq_buffer = deque(maxlen=WINDOW_SIZE)

detector = PresenceDetector()
doppler_detector = DopplerMotionDetector(fs=FS)

doppler_smooth = deque(maxlen=10)

calibration_data = []
calibrated = False

print("\n⚠️ KEEP ROOM EMPTY FOR CALIBRATION\n")
print("🚀 SYSTEM STARTED\n")

# =========================
# 🔁 MAIN LOOP
# =========================
while True:
    try:
        print("\n================ LOOP =================")

        # ================= SERIAL =================
        line1 = ser1.readline().decode(errors="ignore").strip()
        line2 = ser2.readline().decode(errors="ignore").strip()

        if not line1 or not line2:
            print("⚠️ Empty serial")
            continue

        # ================= PARSE =================
        parsed1 = parse_line(line1)
        parsed2 = parse_line(line2)

        if not parsed1:
            print("❌ Parse1 failed")
        if not parsed2:
            print("❌ Parse2 failed")

        if not parsed1 or not parsed2:
            continue

        amp1, phase1, rssi1, iq1 = parsed1
        amp2, phase2, rssi2, iq2 = parsed2

        # ================= FUSION =================
        print("🔗 Fusion...")

        w1 = 10 ** (rssi1 / 10)
        w2 = 10 ** (rssi2 / 10)

        amp = (w1 * amp1 + w2 * amp2) / (w1 + w2)
        phase = (w1 * phase1 + w2 * phase2) / (w1 + w2)
        iq = (w1 * iq1 + w2 * iq2) / (w1 + w2)

        # ================= PREPROCESS =================
        print("🧹 Preprocessing...")
        amp, phase = preprocess_frame(amp, phase)

        amp_buffer.append(amp)
        phase_buffer.append(phase)
        iq_buffer.append(iq)

        print(f"📦 Buffer: {len(amp_buffer)}/{WINDOW_SIZE}")

        if len(amp_buffer) < WINDOW_SIZE:
            continue

        # ================= FILTER =================
        print("🔧 Filtering...")
        _, _, final_signal = filter_window(
            list(amp_buffer),
            list(phase_buffer)
        )

        final_signal = (final_signal - np.mean(final_signal)) / (np.std(final_signal) + 1e-6)

        print("📊 STD:", np.std(final_signal))

        # ================= CALIBRATION =================
        if not calibrated:
            calibration_data.append(final_signal)
            print(f"📍 Calibrating {len(calibration_data)}/{CALIBRATION_FRAMES}")

            if len(calibration_data) >= CALIBRATION_FRAMES:
                detector.calibrate(calibration_data)
                calibrated = True
                print("\n✅ CALIBRATION DONE\n")

            continue

        # ================= PRESENCE =================
        print("🧠 Presence Detection...")
        result = detector.detect(final_signal)

        presence = result["presence"]
        confidence = result["confidence"]

        print(f"VAR RATIO: {result['var_ratio']:.2f}")

        # ================= DOPPLER =================
        print("⚡ Doppler...")

        iq_np = np.array(iq_buffer)
        phase_full = doppler_detector.extract_phase(iq_np)
        phase_mean = np.mean(phase_full, axis=1)

        diff = np.diff(phase_mean)
        doppler = np.median(np.abs(diff)) * FS

        doppler_smooth.append(doppler)
        doppler_final = np.mean(doppler_smooth)

        print(f"⚡ Doppler: {doppler_final:.3f}")

        # ================= FINAL RESULT =================
        print("\n🎯 RESULT")

        if presence and doppler_final > 0.08:
            print(f"🟢 HUMAN + MOTION ({confidence:.1f}%)")

        elif presence:
            print(f"🟡 HUMAN PRESENT ({confidence:.1f}%)")

        elif doppler_final > 0.08:
            print(f"🔵 MOTION ONLY ({confidence:.1f}%)")

        else:
            print(f"⚫ EMPTY ROOM ({confidence:.1f}%)")

        print("="*50)

    except Exception as e:
        print("❌ ERROR:", e)