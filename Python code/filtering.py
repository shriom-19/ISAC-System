import numpy as np


# ===================== 🔥 HAMPEL FILTER =====================
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


# ===================== 🔥 EMA (LIGHT SMOOTHING) =====================
def ema(signal, alpha=0.2):
    signal = np.array(signal, dtype=float)
    out = np.zeros_like(signal)
    out[0] = signal[0]

    for i in range(1, len(signal)):
        out[i] = alpha * signal[i] + (1 - alpha) * out[i - 1]

    return out


# ===================== 🔥 PREPROCESS FRAME =====================
def preprocess_frame(amp, phase):
    """
    Preserve raw CSI (no normalization)
    Fix phase wrapping
    """
    amp = np.array(amp, dtype=float)
    phase = np.array(phase, dtype=float)

    # 🔥 unwrap phase (VERY IMPORTANT)
    phase = np.unwrap(phase)

    return amp, phase


# ===================== 🔥 CORE FILTER =====================
def process_signal(window_data):
    """
    Process each subcarrier independently
    """

    window_data = np.array(window_data)   # (N, 128)
    time_series = window_data.T           # (128, N)

    hampel = HampelFilter(window_length=5, threshold=3.0)

    filtered_all = []

    for sub in time_series:

        s = np.array(sub, dtype=float)

        # 🔥 Step 1: Remove spikes
        s = hampel.filter(s)

        # 🔥 Step 2: Light smoothing (retain info)
        s = ema(s, alpha=0.2)

        # 🔥 Step 3: Remove DC component
        s = s - np.mean(s)

        filtered_all.append(s)

    return np.array(filtered_all)   # (128, N)


# ===================== 🔥 MAIN FILTER =====================
# ===================== 🔥 MAIN FILTER =====================
def filter_window(amp_window, phase_window):
    """
    Full CSI filtering pipeline (Amplitude + Phase)
    Uses ALL subcarriers (NO DATA LOSS)
    """

    # 🔥 Process amplitude and phase separately
    amp_filtered = process_signal(amp_window)
    phase_filtered = process_signal(phase_window)

    # ===================== 🔥 WEIGHTED FUSION =====================

    # Variance = importance
    amp_var = np.var(amp_filtered, axis=1)
    phase_var = np.var(phase_filtered, axis=1)

    score = amp_var + phase_var

    # Normalize weights (VERY IMPORTANT)
    weights = score / (np.sum(score) + 1e-6)

    # Combine amplitude + phase
    combined = amp_filtered + 0.5 * phase_filtered

    # Weighted sum across ALL 128 subcarriers
    final_signal = np.sum(combined.T * weights, axis=1)

    return amp_filtered, phase_filtered, final_signal

# ===================== 🔥 MOTION DETECTION =====================
def detect_motion(signal, threshold=0.2):
    """
    Motion detection using temporal variation
    """

    diff = np.diff(signal)
    score = np.mean(np.abs(diff))

    return score > threshold, score