import serial
import re
import numpy as np
import csv
from collections import deque

# 🔥 YOUR MODULES
from filtering import preprocess_frame, filter_window
from features import extract_features

# ===================== CONFIG =====================
PORT1 = "COM5"
PORT2 = "COM7"
BAUD = 115200

WINDOW_SIZE = 30
FS = 20

DATASET_FILE = "dataset.csv"

# ===================== COUNTERS =====================
total_samples = 0
human_count = 0
empty_count = 0

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

# ===================== PARSER =====================
def parse_line(line):
    try:
        rssi = int(re.search(r"RSSI:(-?\d+)", line).group(1))
        data = re.search(r"DATA:\[(.*?)\]", line).group(1)
        data = list(map(int, data.split(",")))

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

# ===================== CSV INIT =====================
def init_csv():
    header = []

    keys = [
        "variance","rms","mean_abs_diff","peak_to_peak",
        "peak_freq","spectral_energy","doppler_frequency",
        "phase_std","phase_coherence","breathing_period",
        "rssi_var","rssi_diff"
    ]

    for k in keys:
        header.append(f"R1_{k}")
    for k in keys:
        header.append(f"R2_{k}")

    header.append("label")

    with open(DATASET_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# ===================== MAIN =====================
def main():
    global total_samples, human_count, empty_count

    print("🔥 Dual Receiver Dataset Collection Started\n")

    # ===== MODE SELECT =====
    print("Select Mode:")
    print("0 → EMPTY ROOM")
    print("1 → HUMAN PRESENT")

    mode = input("Enter mode: ")

    if mode not in ["0", "1"]:
        print("❌ Invalid mode")
        return

    mode = int(mode)

    if mode == 0:
        print("\n🟢 EMPTY MODE STARTED\n")
    else:
        print("\n🔴 HUMAN MODE STARTED\n")

    init_csv()

    try:
        while True:

            # ===== READ SERIAL =====
            line1 = ser1.readline().decode(errors='ignore')
            line2 = ser2.readline().decode(errors='ignore')

            amp1, phase1, rssi1 = parse_line(line1)
            amp2, phase2, rssi2 = parse_line(line2)

            if amp1 is None or amp2 is None:
                continue

            # ===== RECEIVER 1 =====
            amp1, phase1 = preprocess_frame(amp1, phase1)
            amp_window_1.append(amp1)
            phase_window_1.append(phase1)
            rssi_window_1.append(rssi1)

            # ===== RECEIVER 2 =====
            amp2, phase2 = preprocess_frame(amp2, phase2)
            amp_window_2.append(amp2)
            phase_window_2.append(phase2)
            rssi_window_2.append(rssi2)

            # ===== WAIT FOR FULL WINDOW =====
            if len(amp_window_1) < WINDOW_SIZE or len(amp_window_2) < WINDOW_SIZE:
                continue

            # ===== FILTER =====
            amp_f1, phase_f1, final_signal_1 = filter_window(amp_window_1, phase_window_1)
            amp_f2, phase_f2, final_signal_2 = filter_window(amp_window_2, phase_window_2)

            # ===== FIX: CONVERT PHASE TO 1D =====
            phase_signal_1 = np.mean(phase_f1, axis=0)  # (30,)
            phase_signal_2 = np.mean(phase_f2, axis=0)  # (30,)

            # ===== FEATURE EXTRACTION =====
            f1 = extract_features(final_signal_1, phase_signal_1, rssi_window_1, FS)
            f2 = extract_features(final_signal_2, phase_signal_2, rssi_window_2, FS)

            f1_selected = select_12_features(f1)
            f2_selected = select_12_features(f2)

            combined_features = f1_selected + f2_selected

            label = mode

            # ===== UPDATE COUNTS =====
            total_samples += 1

            if label == 1:
                human_count += 1
            else:
                empty_count += 1

            # ===== SAVE =====
            row = combined_features + [label]

            with open(DATASET_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # ===== LIVE DISPLAY =====
            print(f"\r📊 Samples: {total_samples} | Human: {human_count} | Empty: {empty_count}", end="")

    except KeyboardInterrupt:
        print("\n\n🛑 DATA COLLECTION STOPPED")

        print("\n📊 FINAL DATASET STATS")
        print(f"Total Samples : {total_samples}")
        print(f"Human Samples : {human_count}")
        print(f"Empty Samples : {empty_count}")

        if total_samples > 0:
            print(f"Human %       : {(human_count/total_samples)*100:.1f}%")
            print(f"Empty %       : {(empty_count/total_samples)*100:.1f}%")

        print(f"\n✅ Dataset saved as: {DATASET_FILE}")

        ser1.close()
        ser2.close()

# ===================== RUN =====================
if __name__ == "__main__":
    main()