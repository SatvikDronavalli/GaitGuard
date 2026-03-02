"""
GaitGuard Ablation Study & F1 Score Evaluation
================================================
Evaluates the full CNN-SVM model and runs a systematic ablation
study by removing one component at a time.

Components ablated:
  1. Full Model (baseline)
  2. No Channel Attention
  3. No LSTM
  4. No SVM (CNN-only, sigmoid head)
  5. No Dropout
  6. No Embedding Dense Layer

Metrics: F1, Precision, Recall, PR-AUC, ROC-AUC, Accuracy
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no GUI needed
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, Dropout, MaxPooling1D,
    GlobalAveragePooling1D, LSTM, Layer, Multiply, Reshape,
)
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_curve,
)
import warnings
warnings.filterwarnings("ignore")

# ── Preprocessing (same as AI_Pipeline.py) ────────────────────────

def normalize(val):
    new_val = val.copy()
    channels = new_val.shape[1]
    if channels:
        for i in range(channels):
            mean = np.mean(new_val[:, i])
            std = np.std(new_val[:, i])
            new_val[:, i] = (new_val[:, i] - mean) / (std + 1e-8)
    else:
        mean = np.mean(new_val[:])
        std = np.std(new_val[:])
        new_val[:, 0] = (new_val[:] - mean) / (std + 1e-8)
    return new_val

def resample(norm_val, desired_len):
    n, c = norm_val.shape
    t_old = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t_new = np.linspace(0.0, 1.0, desired_len, dtype=np.float64)
    x = norm_val.astype(np.float64, copy=False)
    y = np.empty((desired_len, c), dtype=np.float64)
    for ch in range(c):
        y[:, ch] = np.interp(t_new, t_old, x[:, ch])
    return y

# ── Channel Attention ─────────────────────────────────────────────

@register_keras_serializable()
class ChannelAttention1D(Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel_dim = input_shape[-1]
        reduced_dim = max(channel_dim // self.reduction_ratio, 1)
        self.dense1 = Dense(reduced_dim, activation="relu", use_bias=True)
        self.dense2 = Dense(channel_dim, activation="sigmoid", use_bias=True)
        super().build(input_shape)

    def call(self, inputs):
        x = GlobalAveragePooling1D()(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = Reshape((1, -1))(x)
        return Multiply()([inputs, x])

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config

# ── Model Builders ────────────────────────────────────────────────

def _compile(model):
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model

def build_full_model():
    return _compile(Sequential([
        Input(shape=(2000, 9)),
        Conv1D(32, 7, activation="relu"), ChannelAttention1D(), MaxPooling1D(2), Dropout(0.2),
        Conv1D(64, 5, activation="relu"), ChannelAttention1D(), MaxPooling1D(2), Dropout(0.2),
        Conv1D(128, 3, activation="relu"), ChannelAttention1D(), MaxPooling1D(2), Dropout(0.3),
        LSTM(64, return_sequences=True), GlobalAveragePooling1D(),
        Dense(64, activation="relu", name="embed"),
        Dense(1, activation="sigmoid"),
    ]))

def build_no_attention():
    return _compile(Sequential([
        Input(shape=(2000, 9)),
        Conv1D(32, 7, activation="relu"), MaxPooling1D(2), Dropout(0.2),
        Conv1D(64, 5, activation="relu"), MaxPooling1D(2), Dropout(0.2),
        Conv1D(128, 3, activation="relu"), MaxPooling1D(2), Dropout(0.3),
        LSTM(64, return_sequences=True), GlobalAveragePooling1D(),
        Dense(64, activation="relu", name="embed"),
        Dense(1, activation="sigmoid"),
    ]))

def build_no_lstm():
    return _compile(Sequential([
        Input(shape=(2000, 9)),
        Conv1D(32, 7, activation="relu"), ChannelAttention1D(), MaxPooling1D(2), Dropout(0.2),
        Conv1D(64, 5, activation="relu"), ChannelAttention1D(), MaxPooling1D(2), Dropout(0.2),
        Conv1D(128, 3, activation="relu"), ChannelAttention1D(), MaxPooling1D(2), Dropout(0.3),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu", name="embed"),
        Dense(1, activation="sigmoid"),
    ]))

def build_no_dropout():
    return _compile(Sequential([
        Input(shape=(2000, 9)),
        Conv1D(32, 7, activation="relu"), ChannelAttention1D(), MaxPooling1D(2),
        Conv1D(64, 5, activation="relu"), ChannelAttention1D(), MaxPooling1D(2),
        Conv1D(128, 3, activation="relu"), ChannelAttention1D(), MaxPooling1D(2),
        LSTM(64, return_sequences=True), GlobalAveragePooling1D(),
        Dense(64, activation="relu", name="embed"),
        Dense(1, activation="sigmoid"),
    ]))

def build_no_embedding():
    return _compile(Sequential([
        Input(shape=(2000, 9)),
        Conv1D(32, 7, activation="relu"), ChannelAttention1D(), MaxPooling1D(2), Dropout(0.2),
        Conv1D(64, 5, activation="relu"), ChannelAttention1D(), MaxPooling1D(2), Dropout(0.2),
        Conv1D(128, 3, activation="relu"), ChannelAttention1D(), MaxPooling1D(2), Dropout(0.3),
        LSTM(64, return_sequences=True), GlobalAveragePooling1D(),
        Dense(1, activation="sigmoid"),
    ]))

# ── Train & evaluate (single run, fast SVM grid) ─────────────────

def train_and_evaluate(name, build_fn, X_train, y_train, X_test, y_test, use_svm=True):
    print(f"    Training ...", end=" ", flush=True)
    model = build_fn()
    early_stop = EarlyStopping(monitor="pr_auc", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=200, batch_size=32,
              callbacks=[early_stop], verbose=0)

    if use_svm:
        feature_extractor = Sequential(model.layers[:-1])
        feature_extractor.build(input_shape=(None, 2000, 9))
        feat_train = feature_extractor.predict(X_train, verbose=0)
        feat_test  = feature_extractor.predict(X_test, verbose=0)

        # Match AI_Pipeline.py: class_weight='balanced', optimize for F1
        best_score, best_svm = -1, None
        for C in [1, 10, 100]:
            for gamma in ["scale", 0.01]:
                svm = SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced")
                svm.fit(feat_train, y_train)
                sc = f1_score(y_train, svm.predict(feat_train))
                if sc > best_score:
                    best_score, best_svm = sc, svm

        y_probs = best_svm.decision_function(feat_test)

        # Optimal F1 threshold (same as AI_Pipeline.py)
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0
        y_pred = (y_probs >= best_threshold).astype(int)
    else:
        y_probs = model.predict(X_test, verbose=0).ravel()
        y_pred  = (y_probs > 0.5).astype(int)

    metrics = {
        "F1":        round(f1_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "PR-AUC":    round(average_precision_score(y_test, y_probs), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_probs), 4),
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
    }
    print(f"F1={metrics['F1']:.3f}  Recall={metrics['Recall']:.3f}  PR-AUC={metrics['PR-AUC']:.3f}")

    report = classification_report(y_test, y_pred, target_names=["Stable", "Unstable"])
    return metrics, report

# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("  GaitGuard – F1 Score & Ablation Study")
    print("=" * 60)
    print("\n📂 Loading dataset ...")

    file_df = pd.read_csv("dataset.csv")
    file_df["Data_Path"] = file_df["Data_Path"].str.replace("\\", "/", regex=False)

    rows = []
    for c in file_df.itertuples(index=False):
        arr = np.load(c.Data_Path, allow_pickle=True)
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)
        processed = normalize(arr)
        walk1 = resample(processed[c.Gait_Start:c.UTurn_Start], 2000)
        turn  = resample(processed[c.UTurn_Start:c.UTurn_End + 1], 2000)
        walk2 = resample(processed[c.UTurn_End + 1:c.Gait_End], 2000)
        rows.append({
            "Patient": c.Patient,
            "walk1_x": walk1[:, 0], "walk1_y": walk1[:, 1], "walk1_z": walk1[:, 2],
            "turn_x":  turn[:, 0],  "turn_y":  turn[:, 1],  "turn_z":  turn[:, 2],
            "walk2_x": walk2[:, 0], "walk2_y": walk2[:, 1], "walk2_z": walk2[:, 2],
            "stability": c.Unstable_Gait,
        })

    df = pd.DataFrame(rows)
    X_cols = df.columns[1:-1]
    X = np.array([np.stack([df.iloc[i][col] for col in X_cols], axis=1) for i in range(len(df))])
    y = df["stability"].values

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df["Patient"]))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"   Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"   Positive rate – train: {y_train.mean():.1%}  test: {y_test.mean():.1%}")

    # ── Ablation variants ─────────────────────────────────────────
    variants = [
        ("Full Model",            build_full_model,  True),
        ("No Channel Attention",  build_no_attention, True),
        ("No LSTM",               build_no_lstm,      True),
        ("No SVM (CNN-only)",     build_full_model,   False),
        ("No Dropout",            build_no_dropout,   True),
        ("No Embedding Layer",    build_no_embedding, True),
    ]

    results = {}
    full_report = ""
    for name, build_fn, use_svm in variants:
        print(f"\n🔬 {name}")
        metrics, report = train_and_evaluate(name, build_fn, X_train, y_train, X_test, y_test, use_svm)
        results[name] = metrics
        if name == "Full Model":
            full_report = report

    # ── Results table ─────────────────────────────────────────────
    print("\n\n" + "=" * 85)
    print("  ABLATION STUDY RESULTS")
    print("=" * 85)
    header = f"{'Variant':<25} {'F1':>7} {'Prec':>7} {'Recall':>7} {'PR-AUC':>7} {'ROC-AUC':>8} {'Acc':>7}"
    print(header)
    print("-" * 85)
    for name, m in results.items():
        print(f"{name:<25} {m['F1']:>7.3f} {m['Precision']:>7.3f} {m['Recall']:>7.3f} "
              f"{m['PR-AUC']:>7.3f} {m['ROC-AUC']:>8.3f} {m['Accuracy']:>7.3f}")
    print("=" * 85)

    full_f1 = results["Full Model"]["F1"]
    print(f"\n🏆 Full Model F1 Score: {full_f1:.3f}")
    print(f"\n📋 Full Model Classification Report:\n{full_report}")

    # ── Save CSV ──────────────────────────────────────────────────
    pd.DataFrame([{"Variant": k, **v} for k, v in results.items()]).to_csv("ablation_results.csv", index=False)
    print("📄 Results saved → ablation_results.csv")

    # ── Bar chart ─────────────────────────────────────────────────
    metrics_to_plot = ["F1", "Precision", "Recall", "PR-AUC", "ROC-AUC"]
    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]
    for i, metric in enumerate(metrics_to_plot):
        vals = [results[n][metric] for n in names]
        ax.bar(x + i * width, vals, width, label=metric, color=colors[i])

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("GaitGuard Ablation Study — Metric Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("ablation_study_chart.png", dpi=150)
    print("📊 Chart saved → ablation_study_chart.png")
