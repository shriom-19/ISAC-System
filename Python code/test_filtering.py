import serial
import re
import math
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt

from filtering import preprocess_frame, filter_window, detect_motion


# ===================== CONFIG =====================
PORT = "COM5"
BAUD = 115200
WINDOW_SIZE = 20
MAX_POINTS = 300   # history length


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


ser = connect_serial()

amp_buffer = deque(maxlen=WINDOW_SIZE)
phase_buffer = deque(maxlen=WINDOW_SIZE)

# 🔥 HISTORY
raw_history = []
filtered_history = []


# ===================== PARSER =====================
def parse_csi(line):
    try:
        if "[" not in line:
            return None

        ts_match = re.search(r'(\d+:\d+:\d+)', line)
        ts = ts_match.group(1) if ts_match else "NA"

        rssi_match = re.search(r'RSSI[:=]?\s*(-?\d+)', line)
        rssi = int(rssi_match.group(1)) if rssi_match else 0

        data_match = re.search(r'\[(.*?)\]', line)
        if not data_match:
            return None

        raw_data = list(map(int, data_match.group(1).split(',')))

        amp = []
        phase = []

        for i in range(0, len(raw_data) - 1, 2):
            real = raw_data[i]
            imag = raw_data[i + 1]

            complex_val = complex(real, imag)

            amp.append(abs(complex_val))
            phase.append(np.angle(complex_val))

        return ts, rssi, amp, phase

    except:
        return None


# ===================== 🔥 GRAPH SETUP =====================
plt.ion()
fig, axs = plt.subplots(2, 1, figsize=(10, 6))

raw_line, = axs[0].plot([], [], linewidth=2)
filtered_line, = axs[1].plot([], [], linewidth=2)

axs[0].set_title("Raw CSI (Amplitude Avg)")
axs[1].set_title("Filtered Motion Signal")

axs[0].grid(True)
axs[1].grid(True)


# 🔥 Y SCALE MEMORY (smooth movement)
raw_min, raw_max = 0, 200
filt_min, filt_max = -5, 5

SMOOTH = 0.2
MARGIN = 0.5


# ===================== MAIN LOOP =====================
print("🚀 CSI System Running (Full Graph + Smooth Axis)...\n")

frame_count = 0

while True:
    try:
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        result = parse_csi(line)

        if result:
            frame_count += 1
            ts, rssi, amp, phase = result

            amp, phase = preprocess_frame(amp, phase)

            amp_buffer.append(amp)
            phase_buffer.append(phase)

            print(f"\n📦 Frame: {frame_count}")
            print(f"⏱ TIME: {ts} | RSSI: {rssi}")

            # ===================== RAW =====================
            if len(amp_buffer) > 0:
                raw_signal = np.mean(np.array(amp_buffer), axis=1)

                raw_history.append(raw_signal[-1])

                if len(raw_history) > MAX_POINTS:
                    raw_history.pop(0)

                raw_line.set_data(range(len(raw_history)), raw_history)
                axs[0].set_xlim(0, MAX_POINTS)

                # 🔥 Smooth Y scaling
                new_min = np.min(raw_history) - MARGIN
                new_max = np.max(raw_history) + MARGIN

                raw_min = (1 - SMOOTH) * raw_min + SMOOTH * new_min
                raw_max = (1 - SMOOTH) * raw_max + SMOOTH * new_max

                axs[0].set_ylim(raw_min, raw_max)

            # ===================== FILTERED =====================
            if len(amp_buffer) >= WINDOW_SIZE:

                _, _, final_signal = filter_window(amp_buffer, phase_buffer)

                motion, score = detect_motion(final_signal)

                print("Motion Score:", round(score, 5))

                if motion:
                    print("🚶 MOTION DETECTED")
                else:
                    print("🟢 NO MOTION")

                filtered_history.append(final_signal[-1])

                if len(filtered_history) > MAX_POINTS:
                    filtered_history.pop(0)

                filtered_line.set_data(range(len(filtered_history)), filtered_history)
                axs[1].set_xlim(0, MAX_POINTS)

                # 🔥 Smooth Y scaling
                new_min = np.min(filtered_history) - MARGIN
                new_max = np.max(filtered_history) + MARGIN

                filt_min = (1 - SMOOTH) * filt_min + SMOOTH * new_min
                filt_max = (1 - SMOOTH) * filt_max + SMOOTH * new_max

                axs[1].set_ylim(filt_min, filt_max)

            # ===================== DRAW =====================
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

    except serial.SerialException:
        print("⚠️ Serial Error → Reconnecting...")
        ser.close()
        ser = connect_serial()

    except KeyboardInterrupt:
        print("\n🛑 Stopped")
        break

    except Exception as e:
        print("⚠️ Error:", e)