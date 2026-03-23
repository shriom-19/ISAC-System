# =========================
# 📥 LOAD DATA
# =========================
import numpy as np


base_path = r"D:\Mini Project Code\data\processed"

X0 = np.load(base_path + r"\X_label_0.npy")
y0 = np.load(base_path + r"\y_label_0.npy")

X1 = np.load(base_path + r"\X_label_1.npy")
y1 = np.load(base_path + r"\y_label_1.npy")

X2 = np.load(base_path + r"\X_label_2.npy")
y2 = np.load(base_path + r"\y_label_2.npy")

print("Loaded successfully ✅")

print("Loaded shapes:")
print("X0:", X0.shape)
print("X1:", X1.shape)
print("X2:", X2.shape)


# =========================
# 🔗 COMBINE ALL DATA (NO TRIMMING)
# =========================
X = np.concatenate([X0, X1, X2], axis=0)
y = np.concatenate([y0, y1, y2], axis=0)

print("\nCombined shape:")
print("X:", X.shape)
print("y:", y.shape)


# =========================
# 🔀 SHUFFLE (VERY IMPORTANT)
# =========================
from sklearn.utils import shuffle

X, y = shuffle(X, y, random_state=42)


# =========================
# 📊 CHECK LABEL DISTRIBUTION
# =========================
print("\nLabel distribution:")
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))


# =========================
# 🔄 CONVERT LABELS (ONE-HOT)
# =========================
from tensorflow.keras.utils import to_categorical

y = to_categorical(y, num_classes=3)


# =========================
# 💾 SAVE FINAL DATASET
# =========================
np.save("X_final.npy", X)
np.save("y_final.npy", y)

print("\n✅ FINAL DATASET READY")
print("X_final shape:", X.shape)
print("y_final shape:", y.shape)