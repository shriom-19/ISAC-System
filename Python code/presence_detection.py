import numpy as np

class PresenceDetector:
    def __init__(self):
        self.baseline_rms = None
        self.baseline_var = None

    # ================= CALIBRATION =================
    def calibrate(self, signal_history):
        """
        signal_history: list of final_signal arrays (empty room)
        """

        # Combine all signals
        full_signal = np.concatenate(signal_history)

        self.baseline_rms = np.sqrt(np.mean(full_signal ** 2))
        self.baseline_var = np.var(full_signal)

        print("\n✅ CALIBRATION COMPLETE")
        print(f"Baseline RMS: {self.baseline_rms:.4f}")
        print(f"Baseline VAR: {self.baseline_var:.4f}\n")

    # ================= DETECTION =================
    def detect(self, signal, sensitivity="medium"):

         current_rms = np.sqrt(np.mean(signal ** 2))
         current_var = np.var(signal)
     
         rms_ratio = current_rms / (self.baseline_rms + 1e-10)
         var_ratio = current_var / (self.baseline_var + 1e-10)
     
         # 🔥 NEW LOGIC
         score = (rms_ratio + var_ratio) / 2
         doppler = np.mean(np.abs(np.diff(signal)))
     
         presence = (
             score > 1.25 or
             var_ratio > 1.5 or
             doppler > 0.3
         )
     
         confidence = max(0, min(100, (score - 1.0) * 100))
     
         return {
             "presence": presence,
             "confidence": confidence,
             "rms_ratio": rms_ratio,
             "var_ratio": var_ratio,
             "doppler": doppler
         }