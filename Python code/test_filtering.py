import serial
import re
import math
import numpy as np
from collections import deque

# 🔥 Import filtering
from filtering import preprocess_frame, filter_window


# ===================== CONFIG =====================
PORT = "COM5"
BAUD = 115200
WINDOW_SIZE = 50

ser = serial.Serial(PORT, BAUD, timeout=1)

csi_buffer = deque(maxlen=WINDOW_SIZE)


# ===================== PARSER =====================
def parse_csi(line):
    try:
        if "DATA:[" not in line:
            return None

        ts_match = re.search(r'TS:(\d+:\d+:\d+)', line)
        ts = ts_match.group(1) if ts_match else None

        rssi_match = re.search(r'RSSI:(-?\d+)', line)
        rssi = int(rssi_match.group(1)) if rssi_match else None

        data_match = re.search(r'DATA:\[(.*)\]', line)
        if not data_match:
            return None

        raw_data = list(map(int, data_match.group(1).split(',')))

        csi = []
        for i in range(0, len(raw_data), 2):
            real = raw_data[i]
            imag = raw_data[i + 1]
            amp = math.sqrt(real**2 + imag**2)
            csi.append(amp)

        return ts, rssi, csi

    except:
        return None


# ===================== 🔥 NEW MOTION DETECTION =====================
def detect_motion(final_signal, threshold=0.002):
    """
    More sensitive motion detection using difference
    """
    diff = np.diff(final_signal)
    motion_score = np.mean(np.abs(diff))

    return motion_score > threshold, motion_score


# ===================== MAIN LOOP =====================
print("🚀 Starting CSI Filtering Test (UPDATED)...\n")

while True:
    try:
        line = ser.readline().decode(errors="ignore").strip()

        result = parse_csi(line)

        if result:
            ts, rssi, csi = result

            # ❌ IMPORTANT: NO NORMALIZATION
            csi = preprocess_frame(csi)   # make sure this does NOT normalize

            csi_buffer.append(csi)

            print(f"\n⏱ TIME: {ts} | RSSI: {rssi}")
            print("RAW CSI:", np.round(csi[:10], 3))

            if len(csi_buffer) == WINDOW_SIZE:

                filtered_signals, final_signal = filter_window(csi_buffer)

                motion, score = detect_motion(final_signal)

                print("FILTERED SIGNAL:", np.round(final_signal[-10:], 3))
                print(f"Motion Score: {score:.5f}")

                # 🔥 DEBUG (IMPORTANT)
                print("Signal Range:", 
                      round(np.min(final_signal), 4), 
                      "→", 
                      round(np.max(final_signal), 4))

                if motion:
                    print("🚶 Motion Detected")
                else:
                    print("🟢 No Motion")

                print("-" * 60)

    except KeyboardInterrupt:
        print("\nStopped")
        break