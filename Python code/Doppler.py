import numpy as np


class DopplerMotionDetector:
    """
    Improved Doppler Motion Detection (CSI-Based)

    ✔ Uses ALL subcarriers
    ✔ Stable (mean aggregation + median diff)
    ✔ Noise resistant
    ✔ Works with your pipeline
    """

    def __init__(self, fs=20, threshold=0.2):
        self.fs = fs
        self.threshold = threshold

    # ===================== 🔥 EXTRACT PHASE =====================
    def extract_phase(self, iq_window):
        """
        iq_window: (N_frames, N_subcarriers, 2)
        """
        I = iq_window[:, :, 0]
        Q = iq_window[:, :, 1]

        phase = np.arctan2(Q, I)

        # ✅ unwrap across time
        phase = np.unwrap(phase, axis=0)

        return phase

    # ===================== 🔥 REMOVE TREND =====================
    def remove_trend(self, phase):
        """
        Remove linear phase drift
        """
        detrended = np.zeros_like(phase)

        for i in range(phase.shape[1]):
            x = np.arange(len(phase[:, i]))
            coeffs = np.polyfit(x, phase[:, i], 1)
            detrended[:, i] = phase[:, i] - (coeffs[0]*x + coeffs[1])

        return detrended

    # ===================== 🔥 COMPUTE DOPPLER =====================
    def compute_doppler(self, phase):
        """
        Stable Doppler computation using ALL subcarriers
        """

        # ✅ Step 1: Normalize each subcarrier
        phase = phase - np.mean(phase, axis=0)

        # ✅ Step 2: Combine ALL subcarriers (mean)
        mean_phase = np.mean(phase, axis=1)

        # ✅ Step 3: Smooth (reduces noise)
        mean_phase = np.convolve(mean_phase, np.ones(3)/3, mode='same')

        # ✅ Step 4: Phase difference
        diff = np.diff(mean_phase)

        # ✅ Step 5: Clamp extreme spikes (important)
        diff = np.clip(diff, -1.0, 1.0)

        # ✅ Step 6: Robust Doppler
        doppler = np.median(np.abs(diff)) * self.fs

        return doppler

    # ===================== 🔥 MAIN DETECTOR =====================
    def detect(self, iq_window):
        """
        iq_window: (N_frames, N_subcarriers, 2)
        """

        if len(iq_window) < 5:
            return {
                "human_detected": False,
                "doppler": 0.0
            }

        # Step 1: Extract phase
        phase = self.extract_phase(iq_window)

        # Step 2: Remove drift
        phase = self.remove_trend(phase)

        # Step 3: Compute Doppler
        doppler = self.compute_doppler(phase)

        # Step 4: Detection
        human = doppler > self.threshold

        return {
            "human_detected": human,
            "doppler": float(doppler)
        }