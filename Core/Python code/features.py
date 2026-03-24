import numpy as np
from scipy.signal import correlate


# ===================== 🔥 RSSI FEATURES =====================
def rssi_features(rssi_window):
    rssi = np.array(rssi_window)

    if len(rssi) < 2:
        return {
            "rssi_mean": 0.0,
            "rssi_var": 0.0,
            "rssi_diff": 0.0
        }

    return {
        "rssi_mean": float(np.mean(rssi)),
        "rssi_var": float(np.var(rssi)),
        "rssi_diff": float(np.mean(np.abs(np.diff(rssi))))
    }


# ===================== 🔥 SAFE ARRAY =====================
def safe_array(signal):
    return np.array(signal, dtype=float) if len(signal) > 0 else np.zeros(1)


# ===================== 🔥 TIME DOMAIN =====================
def time_features(signal):
    signal = safe_array(signal)

    if len(signal) < 2:
        return {
            "variance": 0.0,
            "rms": 0.0,
            "mean_abs_diff": 0.0,
            "peak_to_peak": 0.0,
        }

    diff = np.diff(signal)

    # 🔥 FIX: NO CLIPPING
    variance = float(np.var(signal))

    return {
        "variance": variance,
        "rms": float(np.sqrt(np.mean(signal**2))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "peak_to_peak": float(np.max(signal) - np.min(signal)),
    }


# ===================== 🔥 FREQUENCY DOMAIN =====================
def frequency_features(signal, fs=20):
    signal = safe_array(signal)

    if len(signal) < 4:
        return {
            "peak_freq": 0.0,
            "spectral_energy": 0.0
        }

    # Remove DC
    signal = signal - np.mean(signal)

    # Apply window
    window = np.hamming(len(signal))
    signal = signal * window

    fft_vals = np.fft.fft(signal)
    fft_mag = np.abs(fft_vals)

    freqs = np.fft.fftfreq(len(signal), d=1/fs)

    mask = freqs > 0
    freqs = freqs[mask]
    fft_mag = fft_mag[mask]

    if len(fft_mag) == 0:
        return {
            "peak_freq": 0.0,
            "spectral_energy": 0.0
        }

    peak_freq = freqs[np.argmax(fft_mag)]
    spectral_energy = np.sum(fft_mag**2)  # 🔥 FIXED

    return {
        "peak_freq": float(peak_freq),
        "spectral_energy": float(spectral_energy)
    }


# ===================== 🔥 PHASE DOMAIN =====================
def phase_features(phase_signal, fs=20):
    phase = safe_array(phase_signal)

    if len(phase) < 2:
        return {
            "doppler_frequency": 0.0,
            "phase_std": 0.0,
            "phase_coherence": 0.0
        }

    # 🔥 unwrap
    phase = np.unwrap(phase)

    # 🔥 remove trend
    x = np.arange(len(phase))
    coeffs = np.polyfit(x, phase, 1)
    phase = phase - (coeffs[0]*x + coeffs[1])

    diff = np.diff(phase)

    # 🔥 robust doppler
    doppler = np.median(np.abs(diff)) * fs

    # 🔥 coherence
    coherence = np.abs(np.mean(np.exp(1j * phase)))

    return {
        "doppler_frequency": float(doppler),
        "phase_std": float(np.std(phase)),
        "phase_coherence": float(coherence)
    }


# ===================== 🔥 BREATHING FEATURE =====================
def breathing_feature(signal, fs=20):
    signal = safe_array(signal)

    if len(signal) < 15:
        return {"breathing_period": 0.0}

    signal = signal - np.mean(signal)

    # 🔥 smoothing (important)
    signal = np.convolve(signal, np.ones(5)/5, mode='same')

    acf = correlate(signal, signal, mode='full')
    acf = acf[len(acf)//2:]

    acf = acf / (acf[0] + 1e-10)

    peaks = []
    for i in range(2, len(acf) - 2):
        if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.2:
            peaks.append(i)

    if len(peaks) < 2:
        return {"breathing_period": 0.0}

    period_samples = np.mean(np.diff(peaks))
    period_seconds = period_samples / fs

    # 🔥 realistic breathing limits
    if period_seconds < 0.5 or period_seconds > 10:
        return {"breathing_period": 0.0}

    return {"breathing_period": float(period_seconds)}


# ===================== 🔥 MAIN FEATURE EXTRACTOR =====================
def extract_features(final_signal, phase_signal, rssi_window, fs=20):
    """
    FINAL ROBUST FEATURE SET FOR PRESENCE DETECTION
    """

    features = {}

    # 🔥 TIME
    features.update(time_features(final_signal))

    # 🔥 FREQUENCY
    features.update(frequency_features(final_signal, fs))

    # 🔥 PHASE
    features.update(phase_features(phase_signal, fs))

    # 🔥 BREATHING
    features.update(breathing_feature(final_signal, fs))

    # 🔥 RSSI
    features.update(rssi_features(rssi_window))

    return features