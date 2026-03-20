import serial
import re
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt

# 🔥 IMPORT YOUR MODULES
from filtering import preprocess_frame, filter_window
from features import extract_features


# ===================== CONFIG =====================
PORT = "COM5"
BAUD = 115200

WINDOW_SIZE = 20
MAX_POINTS = 300
FS = 20  # sampling rate


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
rssi_buffer = deque(maxlen=WINDOW_SIZE)

# ===================== HISTORY =====================
csi_history = []
rssi_history = []

feature_history = {
    "variance": [],
    "doppler_frequency": [],
    "spectral_energy": [],
    "breathing_period": []
}


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

            c = complex(real, imag)

            amp.append(abs(c))
            phase.append(np.angle(c))

        return ts, rssi, amp, phase

    except:
        return None


# ===================== GRAPH SETUP =====================
plt.ion()

fig, axs = plt.subplots(3, 2, figsize=(12, 8))

# Graphs
csi_line, = axs[0, 0].plot([], [], lw=2)
rssi_line, = axs[0, 1].plot([], [], lw=2)

var_line, = axs[1, 0].plot([], [], lw=2)
doppler_line, = axs[1, 1].plot([], [], lw=2)

energy_line, = axs[2, 0].plot([], [], lw=2)
breath_line, = axs[2, 1].plot([], [], lw=2)

# Titles
axs[0, 0].set_title("CSI Amplitude (Avg)")
axs[0, 1].set_title("RSSI")

axs[1, 0].set_title("Variance")
axs[1, 1].set_title("Doppler Frequency")

axs[2, 0].set_title("Spectral Energy")
axs[2, 1].set_title("Breathing Period")

for ax in axs.flatten():
    ax.grid(True)


# ===================== MAIN LOOP =====================
print("🚀 CSI + Feature System Running...\n")

frame = 0

while True:
    try:
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        result = parse_csi(line)

        if result:
            frame += 1
            ts, rssi, amp, phase = result

            # preprocess
            amp, phase = preprocess_frame(amp, phase)

            # buffer
            amp_buffer.append(amp)
            phase_buffer.append(phase)
            rssi_buffer.append(rssi)

            print(f"\n📦 Frame: {frame} | TIME: {ts} | RSSI: {rssi}")

            # ================= CSI GRAPH =================
            if len(amp_buffer) > 0:
                avg_csi = np.mean(np.array(amp_buffer), axis=1)
                csi_history.append(avg_csi[-1])

                if len(csi_history) > MAX_POINTS:
                    csi_history.pop(0)

                csi_line.set_data(range(len(csi_history)), csi_history)
                axs[0, 0].set_xlim(0, MAX_POINTS)

            # ================= RSSI GRAPH =================
            rssi_history.append(rssi)
            if len(rssi_history) > MAX_POINTS:
                rssi_history.pop(0)

            rssi_line.set_data(range(len(rssi_history)), rssi_history)
            axs[0, 1].set_xlim(0, MAX_POINTS)

            # ================= FEATURE EXTRACTION =================
            if len(amp_buffer) >= WINDOW_SIZE:

                _, _, final_signal = filter_window(amp_buffer, phase_buffer)

                features = extract_features(
                    final_signal,
                    phase_buffer[-1],
                    rssi_buffer,
                    FS
                )

                # PRINT
                print("Variance:", round(features["variance"], 4))
                print("Doppler:", round(features["doppler_frequency"], 4))
                print("Energy:", round(features["spectral_energy"], 4))
                print("Breathing:", round(features["breathing_period"], 4))

                # ================= STORE =================
                for key in feature_history:
                    feature_history[key].append(features[key])
                    if len(feature_history[key]) > MAX_POINTS:
                        feature_history[key].pop(0)

                # ================= UPDATE GRAPHS =================
                var_line.set_data(range(len(feature_history["variance"])), feature_history["variance"])
                doppler_line.set_data(range(len(feature_history["doppler_frequency"])), feature_history["doppler_frequency"])
                energy_line.set_data(range(len(feature_history["spectral_energy"])), feature_history["spectral_energy"])
                breath_line.set_data(range(len(feature_history["breathing_period"])), feature_history["breathing_period"])

                for ax in axs.flatten():
                    ax.relim()
                    ax.autoscale_view()

            # ================= DRAW =================
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