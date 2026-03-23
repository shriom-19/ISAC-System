import serial
import numpy as np
import re
import os
import time
from scipy.signal import spectrogram

# =========================
# ⚙️ CONFIG
# =========================
PORT1 = "COM5"
PORT2 = "COM7"
BAUD = 115200

WINDOW_SIZE = 100
LABEL = 1  # 0=empty, 1=standing, 2=moving

LABEL_NAMES = {
    0: "EMPTY 🟢",
    1: "STANDING 🧍",
    2: "MOVING 🚶"
}

SAVE_PATH = "data/ruview"
os.makedirs(SAVE_PATH, exist_ok=True)

# 🔥 FIX 2: Faster processing
SUBCARRIER_STEP = 8   # reduced (256 → 32)

# =========================
# 🔌 SERIAL
# =========================
ser1 = serial.Serial(PORT1, BAUD, timeout=1)
ser2 = serial.Serial(PORT2, BAUD, timeout=1)

print("🚀 RuView Dataset Builder Started")
print(f"🎯 Label: {LABEL_NAMES[LABEL]}")
print("Press Ctrl+C to stop\n")

# =========================
# 📦 BUFFERS
# =========================
buffer = []
dataset_X = []
dataset_y = []

frame_count = 0
start_time = time.time()

# =========================
# 🧠 PARSE
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
    except:
        return None

# =========================
# 🔥 PROCESSING
# =========================
def sanitize_phase(phase):
    return np.unwrap(phase)

def remove_static(signal):
    return signal - np.mean(signal)

# =========================
# 🔥 FIX 1: CORRECT DOPPLER
# =========================
def compute_doppler(sample):
    doppler_maps = []

    for i in range(sample.shape[1]):
        f, t, Sxx = spectrogram(
            sample[:, i],
            fs=20,
            nperseg=32,     # ✅ smaller window
            noverlap=16     # ✅ overlap for time dimension
        )

        doppler_maps.append(Sxx)

    doppler_maps = np.array(doppler_maps)

    # (subcarriers, freq, time) → (time, freq, subcarriers)
    doppler_maps = np.transpose(doppler_maps, (2, 1, 0))

    doppler_maps = np.log1p(doppler_maps)

    return doppler_maps

# =========================
# 🔥 MOTION INDICATOR
# =========================
def motion_score(window):
    diff = np.diff(window, axis=0)
    return np.mean(np.abs(diff))

# =========================
# 🚀 MAIN LOOP
# =========================
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

        if len(amp1) != len(amp2):
            continue

        # =========================
        # 🔥 RUVIEW PROCESSING
        # =========================
        phase1 = sanitize_phase(phase1)
        phase2 = sanitize_phase(phase2)

        amp1 = remove_static(amp1)
        amp2 = remove_static(amp2)

        # 🔥 reduced subcarriers
        amp1 = amp1[::SUBCARRIER_STEP]
        amp2 = amp2[::SUBCARRIER_STEP]

        timestep = np.concatenate([amp1, amp2])
        buffer.append(timestep)

        frame_count += 1

        # =========================
        # 🧠 WINDOW
        # =========================
        if len(buffer) >= WINDOW_SIZE:

            sample = np.array(buffer[-WINDOW_SIZE:])
            doppler = compute_doppler(sample)

            dataset_X.append(doppler)
            dataset_y.append(LABEL)

        # =========================
        # 🔥 LIVE INDICATIONS
        # =========================
        progress = min(len(buffer), WINDOW_SIZE)

        if len(buffer) > 20:
            recent = np.array(buffer[-20:])
            motion = motion_score(recent)
        else:
            motion = 0

        # Interpret motion
        if motion < 0.02:
            state = "🟢 Stable"
        elif motion < 0.08:
            state = "🟡 Slight Motion"
        else:
            state = "🔴 Strong Motion"

        elapsed = time.time() - start_time
        fps = frame_count / (elapsed + 1e-6)

        print(
            f"\r📡 Frames: {frame_count} | "
            f"📦 Samples: {len(dataset_X)} | "
            f"⏳ Window: {progress}/{WINDOW_SIZE} | "
            f"🎯 Label: {LABEL_NAMES[LABEL]} | "
            f"🏃 Motion: {motion:.4f} ({state}) | "
            f"⚡ FPS: {fps:.1f}",
            end=""
        )

# =========================
# 💾 SAVE
# =========================
except KeyboardInterrupt:

    print("\n\n🛑 Saving dataset...")

    X = np.array(dataset_X)
    y = np.array(dataset_y)

    print("\n📊 Final Dataset Shape:", X.shape)

    np.save(os.path.join(SAVE_PATH, f"X_label_{LABEL}.npy"), X)
    np.save(os.path.join(SAVE_PATH, f"y_label_{LABEL}.npy"), y)

    print("✅ Dataset saved successfully!")