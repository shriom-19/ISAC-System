import numpy as np
from collections import deque

class PresenceDetector:
    def __init__(self):
        self.base_var = None
        self.base_energy = None
        self.base_diff = None

        # 🔥 Stability buffer (VERY IMPORTANT)
        self.history = deque(maxlen=5)

    # ================= CALIBRATION =================
    def calibrate(self, signals):
        all_signals = np.concatenate(signals)

        self.base_var = np.var(all_signals)
        self.base_energy = np.sum(all_signals**2)
        self.base_diff = np.mean(np.abs(np.diff(all_signals)))

        print("\n✅ CALIBRATION DONE")
        print(f"Base VAR: {self.base_var:.4f}")
        print(f"Base ENERGY: {self.base_energy:.2f}")
        print(f"Base DIFF: {self.base_diff:.4f}\n")

    # ================= DETECTION =================
    def detect(self, signal, sensitivity="medium"):

        # 🔥 SAFETY CHECK
        if self.base_var is None:
            return {
                "presence": False,
                "confidence": 0,
                "score": 0,
                "var_ratio": 1,
                "energy_ratio": 1,
                "diff_ratio": 1
            }

        # ================= FEATURES =================
        var = np.var(signal)
        energy = np.sum(signal**2)
        diff = np.mean(np.abs(np.diff(signal)))

        # ================= SYMMETRIC RATIOS =================
        var_ratio = max(var / (self.base_var + 1e-6),
                        self.base_var / (var + 1e-6))

        energy_ratio = max(energy / (self.base_energy + 1e-6),
                           self.base_energy / (energy + 1e-6))

        diff_ratio = max(diff / (self.base_diff + 1e-6),
                         self.base_diff / (diff + 1e-6))

        # 🔥 CLAMP EXTREME VALUES (VERY IMPORTANT)
        var_ratio = min(var_ratio, 5)
        energy_ratio = min(energy_ratio, 5)
        diff_ratio = min(diff_ratio, 5)

        # ================= SCORE =================
        score = (
            0.4 * var_ratio +
            0.3 * energy_ratio +
            0.3 * diff_ratio
        )

        # ================= THRESHOLD =================
        thresholds = {
            "low": 1.2,
            "medium": 1.1,
            "high": 1.05
        }

        threshold = thresholds.get(sensitivity, 1.1)

        raw_presence = score > threshold

        # 🔥 STABILITY (ANTI-FLICKER)
        self.history.append(1 if raw_presence else 0)
        presence = sum(self.history) >= 3

        # ================= CONFIDENCE =================
        confidence = (score - 1) * 100

        # 🔥 Clamp 0–100
        confidence = max(0, min(100, confidence))

        return {
            "presence": presence,
            "confidence": float(confidence),
            "score": float(score),
            "var_ratio": float(var_ratio),
            "energy_ratio": float(energy_ratio),
            "diff_ratio": float(diff_ratio)
        }