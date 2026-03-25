import serial
import numpy as np
import re
import time
from keras.models import load_model
from scipy.signal import spectrogram
from collections import deque
import os

# =========================
# ⚙️ CONFIG
# =========================
PORT1 = "COM5"
PORT2 = "COM6"
BAUD = 115200

WINDOW_SIZE = 100
SUBCARRIER_STEP = 8
FS = 20

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

model_path = r"D:\Mini Project Code\Core\Python_code\deeplearning\ruview_model.keras"

print("📁 Checking path:", model_path)
print("📁 Exists:", os.path.exists(model_path))

model = load_model(model_path)

print("✅ Model Loaded")
print("🧠 Model Input Shape:", model.input_shape)

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
# 🔥 DOPPLER COMPUTATION
# =========================
def compute_doppler(sample, fs=FS):
    doppler_maps = []

    nper = min(32, sample.shape[0])

    for i in range(sample.shape[1]):
        f, t, Sxx = spectrogram(
            sample[:, i],
            fs=fs,
            nperseg=nper,
            noverlap=nper // 2
        )
        doppler_maps.append(Sxx)

    doppler_maps = np.array(doppler_maps)
    doppler_maps = np.transpose(doppler_maps, (2, 1, 0))
    doppler_maps = np.log1p(doppler_maps)

    return doppler_maps

# =========================
# 🚀 MAIN LOOP (ONLY RUN WHEN FILE EXECUTED)
# =========================
if __name__ == "__main__":

    print("🔌 Opening Serial Ports...")
    ser1 = serial.Serial(PORT1, BAUD, timeout=1)
    ser2 = serial.Serial(PORT2, BAUD, timeout=1)

    time.sleep(2)

    print("🚀 Real-Time DL Detection Started\n")

    buffer = []
    pred_buffer = deque(maxlen=5)

    frame_count = 0
    start_time = time.time()

    try:
        while True:

            line1 = ser1.readline().decode(errors='ignore').strip()
            line2 = ser2.readline().decode(errors='ignore').strip()

            if "DATA:[" not in line1 or "DATA:[" not in line2:
                continue

            parsed1 = parse_raw(line1)
            parsed2 = parse_raw(line2)

            if parsed1 is None or parsed2 is None:
                continue

            amp1, phase1 = parsed1
            amp2, phase2 = parsed2

            phase1 = sanitize_phase(phase1)
            phase2 = sanitize_phase(phase2)

            amp1 = remove_static(amp1)
            amp2 = remove_static(amp2)

            amp1 = amp1[::SUBCARRIER_STEP]
            amp2 = amp2[::SUBCARRIER_STEP]

            timestep = np.concatenate([amp1, amp2])
            buffer.append(timestep)

            frame_count += 1

            if len(buffer) >= WINDOW_SIZE:

                sample = np.array(buffer[-WINDOW_SIZE:])

                doppler = compute_doppler(sample)
                doppler = normalize(doppler)
                doppler = np.expand_dims(doppler, axis=0)

                pred = model.predict(doppler, verbose=0)

                label = np.argmax(pred)
                confidence = np.max(pred)

                pred_buffer.append(label)
                final_label = max(set(pred_buffer), key=pred_buffer.count)

            else:
                final_label = None
                confidence = 0

            if final_label is not None:
                state = LABELS[final_label]
            else:
                state = "Collecting..."

            print(
                f"\r📡 Frames: {frame_count} | "
                f"🎯 State: {state} | "
                f"🔍 Conf: {confidence:.2f}",
                end=""
            )

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped")
        ser1.close()
        ser2.close()