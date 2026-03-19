import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt


# ===================== 🔥 EMA FILTER =====================
def ema(signal, alpha=0.3):
    """
    Exponential Moving Average (temporal smoothing)
    """
    signal = np.array(signal, dtype=float)

    out = np.zeros_like(signal)
    out[0] = signal[0]

    for i in range(1, len(signal)):
        out[i] = alpha * signal[i] + (1 - alpha) * out[i - 1]

    return out


# ===================== 🔥 HAMPEL FILTER =====================
def hampel_filter(signal, window_size=5, threshold=3):
    """
    Robust outlier removal
    """
    signal = np.array(signal, dtype=float)
    filtered = signal.copy()

    half_window = window_size // 2

    for i in range(len(signal)):
        start = max(0, i - half_window)
        end = min(len(signal), i + half_window + 1)

        window = signal[start:end]

        median = np.median(window)
        mad = np.median(np.abs(window - median))

        if mad == 0:
            continue

        if abs(signal[i] - median) > threshold * mad:
            filtered[i] = median

    return filtered


# ===================== 🔥 BANDPASS FILTER =====================
def bandpass_filter(signal, low=0.5, high=2.0, fs=20, order=3):
    """
    Extract motion frequency band
    """
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq

    b, a = butter(order, [low, high], btype='band')

    try:
        return filtfilt(b, a, signal)
    except:
        return signal  # safe fallback


# ===================== 🔥 ZERO HANDLING =====================
def handle_zeros(csi):
    """
    Replace zeros with interpolation (DO NOT REMOVE)
    """
    csi = np.array(csi, dtype=float)

    for i in range(len(csi)):
        if csi[i] == 0:
            if i == 0:
                csi[i] = csi[i + 1]
            elif i == len(csi) - 1:
                csi[i] = csi[i - 1]
            else:
                csi[i] = (csi[i - 1] + csi[i + 1]) / 2

    return csi


# ===================== 🔥 NORMALIZATION =====================
def normalize(signal):
    signal = np.array(signal, dtype=float)

    min_val = np.min(signal)
    max_val = np.max(signal)

    if max_val - min_val == 0:
        return signal

    return (signal - min_val) / (max_val - min_val)


# ===================== 🔥 SINGLE FRAME CLEANING =====================
def preprocess_frame(csi):
    """
    Basic cleaning on single CSI frame (128 values)
    """
    csi = handle_zeros(csi)
    csi = normalize(csi)

    return csi


# ===================== 🔥 SLIDING WINDOW FILTER =====================
def filter_window(csi_window, fs=20):
    """
    Apply filtering on sliding window

    Input:
        csi_window: shape (N_frames, 128)

    Output:
        filtered_signals: shape (128, N_frames)
        final_signal: 1D motion signal
    """

    window_data = np.array(csi_window)          # (N, 128)
    time_series = window_data.T                 # (128, N)

    filtered_signals = []

    for subcarrier_signal in time_series:

        # Step 1: Hampel (remove spikes)
        signal = hampel_filter(subcarrier_signal)

        # Step 2: Savitzky-Golay (smooth)
        if len(signal) >= 7:
            signal = savgol_filter(signal, 7, 2)

        # Step 3: EMA (temporal smoothing)
        signal = ema(signal, alpha=0.3)

        # Step 4: Bandpass (motion extraction)
        signal = bandpass_filter(signal, fs=fs)

        filtered_signals.append(signal)

    filtered_signals = np.array(filtered_signals)   # (128, N)

    # Combine all subcarriers → single motion signal
    final_signal = np.mean(filtered_signals, axis=0)

    return filtered_signals, final_signal


# ===================== 🔥 MOTION DETECTION =====================
def detect_motion(final_signal, threshold=0.01):
    """
    Simple motion detection using variance
    """
    motion_score = np.var(final_signal)

    if motion_score > threshold:
        return True, motion_score
    else:
        return False, motion_score