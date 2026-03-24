import numpy as np


# ===================== 🔥 HAMPEL (SPIKE REMOVAL) =====================
class HampelFilter:
    def __init__(self, window_length=5, threshold=3.0):
        if window_length % 2 == 0:
            window_length += 1
        self.window_length = window_length
        self.threshold = threshold
        self.half = window_length // 2

    def filter(self, signal):
        signal = np.array(signal, dtype=float)
        filtered = signal.copy()

        for i in range(len(signal)):
            start = max(0, i - self.half)
            end = min(len(signal), i + self.half + 1)

            window = signal[start:end]
            median = np.median(window)
            mad = np.median(np.abs(window - median))

            if mad > 1e-6:
                if abs(signal[i] - median) / mad > self.threshold:
                    filtered[i] = median

        return filtered


# ===================== 🔥 EMA (LIGHT NOISE SMOOTHING) =====================
def ema(signal, alpha=0.2):
    signal = np.array(signal, dtype=float)

    out = np.zeros_like(signal)
    out[0] = signal[0]

    for i in range(1, len(signal)):
        out[i] = alpha * signal[i] + (1 - alpha) * out[i - 1]

    return out


# ===================== 🔥 PREPROCESS =====================
def preprocess_frame(amp, phase):
    amp = np.array(amp, dtype=float)
    phase = np.array(phase, dtype=float)

    # Only unwrap phase (important)
    phase = np.unwrap(phase)

    return amp, phase


# ===================== 🔥 NOISE REMOVAL PIPELINE =====================
def process_signal(window_data):

    window_data = np.array(window_data)   # (N, 128)
    time_series = window_data.T           # (128, N)

    hampel = HampelFilter(window_length=5, threshold=3.0)

    filtered_all = []

    for sub in time_series:

        s = np.array(sub, dtype=float)

        # ✅ Step 1: Remove spikes
        s = hampel.filter(s)

        # ✅ Step 2: Light smoothing
        s = ema(s, alpha=0.2)

        filtered_all.append(s)

    return np.array(filtered_all)   # (128, N)


# ===================== 🔥 MAIN FILTER =====================
def filter_window(amp_window, phase_window):

    amp_filtered = process_signal(amp_window)
    phase_filtered = process_signal(phase_window)

    # ✅ KEEP ALL SUBCARRIERS (NO SELECTION)
    final_signal = (
        np.mean(amp_filtered, axis=0)
        + 0.5 * np.mean(phase_filtered, axis=0)
    )

    return amp_filtered, phase_filtered, final_signal


# ===================== 🔥 MOTION =====================
def detect_motion(signal, threshold=0.05):
    diff = np.diff(signal)
    score = np.mean(np.abs(diff))
    return score > threshold, score