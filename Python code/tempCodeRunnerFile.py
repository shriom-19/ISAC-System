from presence_detection import PresenceDetector
from Doppler import DopplerMotionDetector
from filtering import preprocess_frame, filter_window

import serial
import re
import numpy as np
from collections import deque
import time

# ================= CONFIG =================
PORT1 = "COM5"
PORT2 = "COM7"
BAUD = 115200

WINDOW_SIZE = 20
FS = 20
CALIBRATION_FRAMES = 60
TARGET_LEN = 128

# ================= SERIAL =================
def connect_serial(port):
    while True:
        try:
            ser = serial.Serial(port, BAUD, timeout=1)
            print(f"✅ Connected to {port}")
            return ser
        except:
            print(f"❌ Waiting for {port}...")
            time.sleep(2)

# ================= PARSER =================
def parse_line(line):
    try:
        if "DATA" not in line:
            return None

        rssi_match = re.search(r'RSSI[:=]?\s*(-?\d+)', line)
        data_match = re.search(r'DATA:\[(.*?)\]', line)

        if not rssi_match or not data_match:
            return None

        rssi = int(rssi_match.group(1))
        raw = list(map(int, data_match.group(1).split(',')))

        if len(raw) % 2 != 0:
            return None

        i = np.array(raw[::2], dtype=float)
        q = np.array(raw[1::2], dtype=float)

        i[i == 0] = 0.001
        q[q == 0] = 0.001

        if len(i) >= TARGET_LEN:
            i, q = i[:TARGET_LEN], q[:TARGET_LEN]
        else:
            pad = TARGET_LEN - len(i)
            i = np.pad(i, (0, pad))
            q = np.pad(q, (0, pad))

        amp = np.sqrt(i**2 + q**2)
        phase = np.arctan2(q, i)
        iq = np.stack((i, q), axis=1)

        return amp, phase, rssi, iq

    except:
        return None


# ================= INIT =================
ser1 = connect_serial(PORT1)
ser2 = connect_serial(PORT2)

amp_buffer = deque(maxlen=WINDOW_SIZE)
phase_buffer = deque(maxlen=WINDOW_SIZE)
iq_buffer = deque(maxlen=WINDOW_SIZE)

detector = PresenceDetector()
doppler_detector = DopplerMotionDetector(fs=FS)

doppler_smooth = deque(maxlen=8)

calibration_data = []
calibrated = False

print("\n⚠️ KEEP ROOM EMPTY FOR CALIBRATION\n")
print("🚀 FINAL ACCURATE PIPELINE STARTED\n")

# ================= MAIN LOOP =================
while True:
    try:
        line1 = ser1.readline().decode(errors="ignore").strip()
        line2 = ser2.readline().decode(errors="ignore").strip()

        parsed1 = parse_line(line1)
        parsed2 = parse_line(line2)

        if not parsed1 or not parsed2:
            continue

        amp1, phase1, rssi1, iq1 = parsed1
        amp2, phase2, rssi2, iq2 = parsed2

        # ================= FUSION =================
        w1 = abs(rssi1)
        w2 = abs(rssi2)

        amp = (w1 * amp1 + w2 * amp2) / (w1 + w2)
        phase = (w1 * phase1 + w2 * phase2) / (w1 + w2)
        iq = (w1 * iq1 + w2 * iq2) / (w1 + w2)

        # ================= PREPROCESS =================
        amp, phase = preprocess_frame(amp, phase)

        amp_buffer.append(amp)
        phase_buffer.append(phase)
        iq_buffer.append(iq)

        if len(amp_buffer) < WINDOW_SIZE:
            continue

        # ================= FILTER =================
        _, _, final_signal = filter_window(
            list(amp_buffer),
            list(phase_buffer)
        )

        # ================= NORMALIZATION =================
        final_signal = (final_signal - np.mean(final_signal)) / (np.std(final_signal) + 1e-6)

        # ================= CALIBRATION =================
        if not calibrated:
            calibration_data.append(final_signal)

            print(f"📍 Calibrating {len(calibration_data)}/{CALIBRATION_FRAMES}")

            if len(calibration_data) >= CALIBRATION_FRAMES:
                detector.calibrate(calibration_data)
                calibrated = True
                print("\n✅ CALIBRATION COMPLETE\n")

            continue

        # ================= PRESENCE =================
        result = detector.detect(final_signal)

        presence = result["presence"]
        confidence = result["confidence"]

        var_r = result["var_ratio"]
        diff_r = result["diff_ratio"]

        print("\n🧠 PRESENCE")
        print(f"VAR: {var_r:.2f} | DIFF: {diff_r:.2f}")

        # ================= DOPPLER (ROBUST) =================
        iq_np = np.array(iq_buffer)

        phase_full = doppler_detector.extract_phase(iq_np)
        phase_full = phase_full[:, 10:100]

        phase_mean = np.mean(phase_full, axis=1)

        x = np.arange(len(phase_mean))
        coeffs = np.polyfit(x, phase_mean, 1)
        phase_detrended = phase_mean - (coeffs[0]*x + coeffs[1])

        diff = np.diff(phase_detrended)

        # 🔥 MAD FILTER (NO CLIPPING)
        median = np.median(diff)
        mad = np.median(np.abs(diff - median)) + 1e-6

        clean_diff = diff[np.abs(diff - median) < 3 * mad]

        if len(clean_diff) < 5:
            doppler = 0.0
        else:
            doppler = np.median(np.abs(clean_diff)) * FS

        doppler_smooth.append(doppler)
        doppler_final = np.mean(doppler_smooth)

        print(f"⚡ Doppler: {doppler_final:.3f}")

        # ================= FINAL DECISION =================
        print("\n🎯 RESULT")

        var_th = 1.25
        doppler_th = max(0.15, np.mean(doppler_smooth))

        if presence and doppler_final > doppler_th and var_r > var_th:
            print(f"🟢 HUMAN + MOTION ({confidence:.1f}%)")

        elif presence and var_r > 1.15:
            print(f"🟡 HUMAN PRESENT ({confidence:.1f}%)")

        elif doppler_final > doppler_th * 1.2:
            print(f"🔵 MOTION ONLY ({confidence:.1f}%)")

        else:
            print(f"⚫ EMPTY ROOM ({confidence:.1f}%)")

        print("="*50)

    except Exception as e:
        print("❌ ERROR:", e)