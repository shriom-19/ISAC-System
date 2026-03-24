import serial
import numpy as np
import re
import time
from keras.models import load_model
from scipy.signal import spectrogram
from collections import deque

# =========================
# ⚙️ CONFIG
# =========================
PORT1 = "COM5"
PORT2 = "COM7"
BAUD = 115200

WINDOW_SIZE = 100              # 🔥 faster testing
SUBCARRIER_STEP = 8           # 🔥 MUST match dataset builder
FS = 20                       # sampling freq

# =========================
# 🧠 LABELS
# =========================
LABELS = {
    0: "EMPTY 🟢",
    1: "STANDING 🧍",
    2: "MOVING 🚶"
}

# =========================
# 📦 LOAD MODEL
# =========================
print("📦 Loading model...")

model_path = "D:/Mini Project Code/Python code/deeplearning/ruview_csi_model_grouped.h5"

import os
print("📁 Checking path:", model_path)
print("📁 Exists:", os.path.exists(model_path))

model = load_model(model_path)

print("✅ Model Loaded")
print("🧠 Model Input Shape:", model.input_shape)

# =========================
# 🔌 SERIAL INIT
# =========================
print("🔌 Opening Serial Ports...")
ser1 = serial.Serial(PORT1, BAUD, timeout=1)
ser2 = serial.Serial(PORT2, BAUD, timeout=1)

time.sleep(2)  # allow connection

print("🚀 Real-Time DL Detection Started\n")

# =========================
# 📦 BUFFERS
# =========================
buffer = []
pred_buffer = deque(maxlen=5)

frame_count = 0
start_time = time.time()

# =========================
# 🧠 PARSE RAW CSI
# =========================
def parse_raw(line):
    try:
        data_match = re.search(r"DATA:\[(.*?)\]", line)
        if not data_match:
            return None

        data = np.array(list(map(int, data_match.group(1).split(","))))

        I = data[0::2]
        Q = data[1::2]

        amp = np.sqrt(I**2 + Q**2)
        phase = np.arctan2(Q, I)

        return amp, phase

    except Exception as e:
        print("❌ Parse error:", e)
        return None

# =========================
# 🔥 PROCESSING
# =========================
def sanitize_phase(phase):
    return np.unwrap(phase)

def remove_static(signal):
    return signal - np.mean(signal)

def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6)

# =========================
# 🔥 DOPPLER (MATCH DATASET BUILDER)
# =========================
def compute_doppler(sample):
    doppler_maps = []

    nper = min(32, sample.shape[0])   # 🔥 fix

    for i in range(sample.shape[1]):
        f, t, Sxx = spectrogram(
            sample[:, i],
            fs=FS,
            nperseg=nper,
            noverlap=nper // 2
        )
        doppler_maps.append(Sxx)

    doppler_maps = np.array(doppler_maps)
    doppler_maps = np.transpose(doppler_maps, (2, 1, 0))
    doppler_maps = np.log1p(doppler_maps)

    return doppler_maps

# =========================
# 🔥 MAIN LOOP
# =========================
try:
    while True:

        # =========================
        # 📡 SERIAL READ
        # =========================
        line1 = ser1.readline().decode(errors='ignore').strip()
        line2 = ser2.readline().decode(errors='ignore').strip()

        # DEBUG RAW
        if frame_count % 20 == 0:
            print("\n📡 RAW1:", line1[:80])
            print("📡 RAW2:", line2[:80])

        if "DATA:[" not in line1:
            print("⚠️ No DATA in PORT1")
            continue

        if "DATA:[" not in line2:
            print("⚠️ No DATA in PORT2")
            continue

        # =========================
        # 🧠 PARSE
        # =========================
        parsed1 = parse_raw(line1)
        if parsed1 is None:
            print("❌ Parse1 failed")
            continue

        parsed2 = parse_raw(line2)
        if parsed2 is None:
            print("❌ Parse2 failed")
            continue

        amp1, phase1 = parsed1
        amp2, phase2 = parsed2

        if len(amp1) != len(amp2):
            print("⚠️ Length mismatch")
            continue

        # =========================
        # 🔥 PREPROCESS
        # =========================
        phase1 = sanitize_phase(phase1)
        phase2 = sanitize_phase(phase2)

        amp1 = remove_static(amp1)
        amp2 = remove_static(amp2)

        amp1 = amp1[::SUBCARRIER_STEP]
        amp2 = amp2[::SUBCARRIER_STEP]

        timestep = np.concatenate([amp1, amp2])
        buffer.append(timestep)

        frame_count += 1

        # =========================
        # 🧠 WINDOW READY
        # =========================
        if len(buffer) >= WINDOW_SIZE:

            sample = np.array(buffer[-WINDOW_SIZE:])

            # DEBUG SHAPE
            print("\n📊 Window Shape:", sample.shape)

            doppler = compute_doppler(sample)
            doppler = normalize(doppler)

            doppler = np.expand_dims(doppler, axis=0)

            print("📊 Final Input Shape:", doppler.shape)

            # =========================
            # 🔥 PREDICTION
            # =========================
            pred = model.predict(doppler, verbose=0)

            label = np.argmax(pred)
            confidence = np.max(pred)

            pred_buffer.append(label)

            # Smooth prediction
            final_label = max(set(pred_buffer), key=pred_buffer.count)

        else:
            final_label = None
            confidence = 0

        # =========================
        # 📊 DISPLAY
        # =========================
        elapsed = time.time() - start_time
        fps = frame_count / (elapsed + 1e-6)

        if final_label is not None:
            state = LABELS[final_label]
        else:
            state = "Collecting..."

        print(
            f"\r📡 Frames: {frame_count} | "
            f"⏳ Window: {min(len(buffer), WINDOW_SIZE)}/{WINDOW_SIZE} | "
            f"🎯 State: {state} | "
            f"🔍 Conf: {confidence:.2f} | "
            f"⚡ FPS: {fps:.2f}",
            end=""
        )

# =========================
# 🛑 STOP
# =========================
except KeyboardInterrupt:
    print("\n\n🛑 Stopped")
    ser1.close()
    ser2.close()