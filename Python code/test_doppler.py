import serial
import re
import numpy as np
from collections import deque

# 🔥 IMPORT YOUR MODULES
from filtering import preprocess_frame, filter_window
from features import extract_features
from Doppler import DopplerMotionDetector


# ===================== CONFIG =====================
PORT = "COM5"
BAUD = 115200

WINDOW_SIZE = 30
FS = 20

# ===================== INIT =====================
ser = serial.Serial(PORT, BAUD, timeout=1)

iq_history = deque(maxlen=WINDOW_SIZE)
phase_history = deque(maxlen=WINDOW_SIZE)
doppler_smooth_buffer = deque(maxlen=5)

doppler_detector = DopplerMotionDetector(fs=FS, threshold=0.15)


# ===================== PARSER =====================
def parse_raw_line(line):
    try:
        rssi_match = re.search(r"RSSI:(-?\d+)", line)
        data_match = re.search(r"DATA:\[(.*?)\]", line)

        if not rssi_match or not data_match:
            return None, None

        rssi = int(rssi_match.group(1))
        data = list(map(int, data_match.group(1).split(',')))

        return rssi, data
    except:
        return None, None


# ===================== CSI → I/Q =====================
def extract_iq(data):
    iq = []
    for i in range(0, len(data) - 1, 2):
        iq.append([data[i], data[i + 1]])
    return np.array(iq)


# ===================== MAIN LOOP =====================
print("🚀 Doppler Debug Mode Started...\n")

while True:
    try:
        line = ser.readline().decode(errors='ignore').strip()

        if "DATA:" not in line:
            continue

        print("\n" + "="*70)
        print(f"📥 RAW INPUT:\n{line[:120]}...")

        # ================= STEP 1: PARSE =================
        rssi, data = parse_raw_line(line)

        if data is None:
            print("❌ Parsing failed")
            continue

        print("\n[STEP 1] PARSING")
        print(f"RSSI: {rssi}")
        print(f"CSI Length: {len(data)}")

        # ================= STEP 2: I/Q =================
        iq = extract_iq(data)

        print("\n[STEP 2] I/Q EXTRACTION")
        print(f"IQ Shape: {iq.shape}")
        print(f"First 5 IQ pairs:\n{iq[:5]}")

        if len(iq) < 10:
            print("⚠️ Not enough subcarriers")
            continue

        # ================= STEP 3: AMP + PHASE =================
        amp = np.sqrt(iq[:, 0]**2 + iq[:, 1]**2)
        phase = np.arctan2(iq[:, 1], iq[:, 0])

        print("\n[STEP 3] AMPLITUDE & PHASE")
        print(f"Amp sample: {amp[:5]}")
        print(f"Phase sample: {phase[:5]}")

        # ================= STEP 4: PREPROCESS =================
        amp, phase = preprocess_frame(amp, phase)

        print("\n[STEP 4] PREPROCESS")
        print("Phase unwrapped ✔")

        # ================= STEP 5: STORE =================
        iq_history.append(iq)
        phase_history.append(phase)

        print("\n[STEP 5] WINDOW BUFFER")
        print(f"Frames stored: {len(iq_history)}/{WINDOW_SIZE}")

        if len(iq_history) < WINDOW_SIZE:
            print("⏳ Waiting for full window...")
            continue

        iq_window = np.array(iq_history)

        # ================= STEP 6: DOPPLER =================
        print("\n[STEP 6] DOPPLER COMPUTATION")

        phase_full = doppler_detector.extract_phase(iq_window)
        print("Phase extracted ✔")

        phase_clean = doppler_detector.remove_trend(phase_full)
        print("Trend removed ✔")

        doppler_raw = doppler_detector.compute_doppler(phase_clean)
        print(f"Doppler raw: {doppler_raw:.5f}")

        # 🔥 SMOOTHING
        doppler_smooth_buffer.append(doppler_raw)
        doppler_smooth = np.mean(doppler_smooth_buffer)

        print(f"Doppler smoothed: {doppler_smooth:.5f}")

        # ================= STEP 7: FILTERING =================
        print("\n[STEP 7] FILTERING (Hampel + EMA)")

        amp_window = [np.sqrt(i[:, 0]**2 + i[:, 1]**2) for i in iq_history]
        phase_window = list(phase_history)

        amp_f, phase_f, final_signal = filter_window(amp_window, phase_window)

        print("Filtering complete ✔")

        # ================= STEP 8: FEATURES =================
        print("\n[STEP 8] FEATURE EXTRACTION")

        phase_window_np = np.array(phase_history)
        mean_phase = np.mean(phase_window_np, axis=1)

        features = extract_features(final_signal, mean_phase, [rssi], fs=FS)

        print(f"Variance: {features['variance']:.4f}")
        print(f"RMS: {features['rms']:.4f}")
        print(f"Doppler (features.py): {features['doppler_frequency']:.4f}")
        print(f"Phase Coherence: {features['phase_coherence']:.4f}")

        # ================= STEP 9: FINAL DECISION =================
        print("\n[STEP 9] FINAL DECISION")

        print(f"Final Doppler: {doppler_smooth:.5f}")

        # 🔥 FINAL LOGIC (STABLE)
        if doppler_smooth > 0.15 and features["phase_coherence"] < 0.9:
            print("🟢 HUMAN DETECTED (MOTION)")
        else:
            print("⚫ NO MOTION")

        print("="*70)

    except Exception as e:
        print(f"❌ ERROR: {e}")