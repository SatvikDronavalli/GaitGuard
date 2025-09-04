# sensor_importance_updated_cnn.py
# Analyzes channel/sensor importance for your 101x8 CNN trained as 'updated_fall_risk.keras'

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

# === Load the same arrays you trained with ===
# Your training code imports these from 'visualization_test'
from visualization_test import X_train, y_train  # shape ~ (N, 101, 8), (N,)

MODEL_PATH = "updated_fall_risk.keras"
OUT_DIR = Path("AI_Models/CNN/_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Prep data -----
X = np.array(X_train, dtype=np.float32)
y = np.asarray(y_train).reshape(-1)

# Split a validation set from your train data (keeps test set pristine if you have one elsewhere)
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

N, T, C = X_val.shape
CHANNEL_NAMES = [f"ch{i}" for i in range(C)]  # edit if you know real names

# Mean per channel (constant over time) for neutral masking
mean_per_channel = np.tile(X_tr.mean(axis=(0, 1), keepdims=True), (1, T, 1))[0]

# ----- Load model -----
model = load_model(MODEL_PATH)

def predict_proba(m, X_):
    return m.predict(X_, verbose=0).ravel()

def score_auc_or_f1(m, X_, y_):
    # Prefer AUC; fall back to F1 if AUC not defined (e.g., single-class edge case)
    try:
        p = predict_proba(m, X_)
        return roc_auc_score(y_, p)
    except Exception:
        p = (predict_proba(m, X_) > 0.5).astype(int)
        return f1_score(y_, p)

base = score_auc_or_f1(model, X_val, y_val)
print(f"Baseline metric on validation (AUC or F1 fallback): {base:.4f}")

# =========== 1) Channel permutation importance ===========
print("\n[1/3] Channel permutation importance...")
rng = np.random.default_rng(42)
drops = np.zeros(C)

for c in range(C):
    Xp = X_val.copy()
    perm = rng.permutation(N)
    Xp[:, :, c] = X_val[perm, :, c]  # shuffle channel across samples
    drops[c] = base - score_auc_or_f1(model, Xp, y_val)

order = np.argsort(-drops)

# Save CSV + bar chart
csv_path = OUT_DIR / "channel_importance.csv"
with open(csv_path, "w") as f:
    f.write("rank,channel,drop\n")
    for r, idx in enumerate(order, 1):
        f.write(f"{r},{CHANNEL_NAMES[idx]},{drops[idx]:.6f}\n")
print("Saved:", csv_path)

plt.figure()
plt.bar([CHANNEL_NAMES[i] for i in range(C)], drops)
plt.ylabel("Performance drop (importance)")
plt.title("Channel permutation importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "channel_importance.png", dpi=200)
plt.close()
print("Saved:", OUT_DIR / "channel_importance.png")

print("\nRanked channels (bigger drop = more important):")
for r, idx in enumerate(order, 1):
    print(f"{r:2d}. {CHANNEL_NAMES[idx]:<8} | drop = {drops[idx]:.4f}")

# =========== 2) Temporal occlusion (when in gait cycle) ===========
print("\n[2/3] Temporal occlusion (which timesteps matter)...")
window = max(5, T // 10)  # ~10% of the gait cycle
time_importance = np.zeros(T)
coverage = np.zeros(T)

for t0 in range(0, T - window + 1):
    t1 = t0 + window
    Xo = X_val.copy()
    # Replace this time window with neutral per-channel means
    Xo[:, t0:t1, :] = mean_per_channel
    drop = base - score_auc_or_f1(model, Xo, y_val)
    time_importance[t0:t1] += drop
    coverage[t0:t1] += 1

time_importance /= np.maximum(coverage, 1e-9)

plt.figure()
plt.plot(np.arange(T), time_importance)
plt.xlabel("Timestep in gait cycle (0 … T-1)")
plt.ylabel("Importance (drop when occluded)")
plt.title("Temporal importance over gait cycle")
plt.tight_layout()
plt.savefig(OUT_DIR / "temporal_importance.png", dpi=200)
plt.close()
print("Saved:", OUT_DIR / "temporal_importance.png")

# =========== 3) Greedy minimal sensor set (no retraining) ===========
print("\n[3/3] Greedy selection of minimal sensor set (no retraining)...")
target = base - max(0.01 * base, 0.005)  # within ~1–2% absolute of baseline
chosen = []
remaining = list(range(C))
best_so_far = 0.0

while remaining:
    best_add, best_score = None, -1
    for c in remaining:
        keep = chosen + [c]
        Xk = np.zeros_like(X_val)
        Xk[:, :, keep] = X_val[:, :, keep]
        s = score_auc_or_f1(model, Xk, y_val)
        if s > best_score:
            best_score, best_add = s, c
    chosen.append(best_add)
    remaining.remove(best_add)
    print(f" -> add {CHANNEL_NAMES[best_add]} | score = {best_score:.4f} "
          f"| chosen = {[CHANNEL_NAMES[i] for i in chosen]}")
    if best_score >= target:
        break

print("\nRecommended minimal sensor set (near full performance):")
for i, c in enumerate(chosen, 1):
    print(f"{i}. {CHANNEL_NAMES[c]}")

# Save the pick
np.save(OUT_DIR / "recommended_channels_idx.npy", np.array(chosen, dtype=int))
with open(OUT_DIR / "recommended_channels.txt", "w") as f:
    for i, c in enumerate(chosen, 1):
        f.write(f"{i}. {CHANNEL_NAMES[c]}\n")
print("Saved:", OUT_DIR / "recommended_channels.txt")

# ---- Optional: If you have a separate test set, use it instead of a split ----
# from Force_Data.force_data import final_x_test, final_y_test
# X_val = np.array(final_x_test, dtype=np.float32); y_val = np.asarray(final_y_test).reshape(-1)
# (then rerun the three blocks above using this X_val/y_val)
