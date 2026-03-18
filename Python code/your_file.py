import serial
import numpy as np
import matplotlib.pyplot as plt

# 🔧 CHANGE PORT IF NEEDED
ser = serial.Serial('COM5', 115200)

plt.ion()
fig, ax = plt.subplots()

y_data = []

# 🔥 SETTINGS
window_size = 10
threshold_offset = 3   # sensitivity

# 🔥 BASELINE VARIABLES
baseline_values = []
baseline = 0
calibration_done = False

def parse_csi(line):
    try:
        if "CSI_DATA" not in line:
            return None

        data_str = line.split("DATA:[")[1].split("]")[0]
        values = list(map(int, data_str.split(",")))

        I = values[::2]
        Q = values[1::2]

        amp = np.sqrt(np.array(I)**2 + np.array(Q)**2)

        # 🔥 REMOVE STATIC COMPONENT
        amp = amp - np.mean(amp)

        # 🔥 MOTION FEATURE (BEST)
        value = np.mean(np.abs(np.diff(amp)))

        return value

    except:
        return None


while True:
    try:
        line = ser.readline().decode(errors='ignore')
        value = parse_csi(line)

        if value is None:
            continue

        # 🥇 CALIBRATION PHASE
        if not calibration_done:
            baseline_values.append(value)

            print(f"Calibrating... {len(baseline_values)}/50")

            if len(baseline_values) >= 50:
                baseline = np.mean(baseline_values)
                calibration_done = True
                print("\n✅ Baseline set:", round(baseline, 2))
                print("System Ready 🚀\n")

            continue

        # 🥇 DETECTION
        delta = abs(value - baseline)

        if delta > threshold_offset:
            status = "🚶 MOTION"
        else:
            status = "🟢 STILL"

        print(f"Value: {value:.2f} | Δ: {delta:.2f} | {status}")

        # 🥇 STORE DATA
        y_data.append(value)

        if len(y_data) > 100:
            y_data = y_data[-100:]

        # 🥇 SMOOTHING
        if len(y_data) >= window_size:
            smooth = np.convolve(y_data, np.ones(window_size)/window_size, mode='valid')
        else:
            smooth = y_data

        current = smooth[-1]

        # 🥇 PLOT
        ax.clear()
        ax.plot(smooth, label="CSI Signal")

        # baseline line
        ax.axhline(baseline, linestyle='--', label="Baseline")

        ax.set_title(f"CSI Motion Detection | {status}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Signal")

        ax.legend()

        plt.pause(0.01)

    except KeyboardInterrupt:
        print("Stopped")
        break