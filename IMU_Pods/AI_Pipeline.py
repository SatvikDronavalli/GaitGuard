import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Dropout, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report,make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


curr_path = Path.cwd() / "Processed_Data"

total = len(list(curr_path.iterdir()))

bad = 0

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


file_df = pd.read_csv("dataset.csv")

walk1 = np.array([])


walk1_vals = []
turn_vals = []
walk2_vals = []

rows = []
full_walks = []

for c in file_df.itertuples(index=False):
    name = c.Patient
    arr = np.load(c.Data_Path,allow_pickle=True)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    processed_arr = normalize(arr)
    walk1 = resample(processed_arr[c.Gait_Start:c.UTurn_Start],2000)
    turn = resample(processed_arr[c.UTurn_Start:c.UTurn_End+1],2000)
    walk2 = resample(processed_arr[c.UTurn_End+1:c.Gait_End],2000)
    six_k_resample = np.concatenate((walk1,turn,walk2))
    full_walks.append(six_k_resample)
    rows.append({
        "Patient": name,
        "walk1_x": walk1[:, 0],
        "walk1_y": walk1[:, 1],
        "walk1_z": walk1[:, 2],
        "turn_x": turn[:, 0],
        "turn_y": turn[:, 1],
        "turn_z": turn[:, 2],
        "walk2_x": walk2[:, 0],
        "walk2_y": walk2[:, 1],
        "walk2_z": walk2[:, 2],
        "stability": c.Unstable_Gait})

    # lb_gyr_x = processed_arr[:, 0]


df = pd.DataFrame(rows)

X = df.iloc[:, 1:-1]
X = np.array([np.stack(tuple(df.iloc[i][col] for col in X.columns), axis=1) for i in range(len(X))])
y = df.iloc[:, -1]

gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['Patient']))
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y.iloc[train_idx].to_numpy(), y.iloc[test_idx].to_numpy()

'''
X = X.to_numpy()
print(X.shape)
X = np.stack([np.stack((i[0],i[1],i[2]),axis=1) for i in X], axis=0)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
'''

print(y_train.shape)

# Define the model for GRF data
'''
model = Sequential([ # Modify architecture, more convolution
    Input(shape=(2000,9)),
    Conv1D(filters=32, kernel_size=11, activation='relu'),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output
]) '''

model = Sequential([
    Input(shape=(2000, 9)),

    Conv1D(32, kernel_size=7, activation='relu'),  # Broad patterns
    MaxPooling1D(2),
    Dropout(0.2),

    Conv1D(64, kernel_size=5, activation='relu'),  # Medium patterns
    MaxPooling1D(2),
    Dropout(0.2),

    Conv1D(128, kernel_size=3, activation='relu'),  # Fine details
    MaxPooling1D(2),
    Dropout(0.3),
    GlobalAveragePooling1D(),
    Dense(1, activation='sigmoid')
    # Flatten(),
    # Dense(128, activation='relu'),
    # Dropout(0.5),
    # Dense(64, activation='relu'),
    # Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
    ])

# Define checkpoints and early stopping
checkpoint = ModelCheckpoint('updated_fall_risk.keras', save_best_only=True, monitor='pr_auc', mode='max')
early_stopping = EarlyStopping(monitor='pr_auc', patience=5, restore_best_weights=True)

# Train the model with GRF data and labels
history = model.fit(
    X_train,
    y_train,  # Use actual labels here
    epochs=200,
    batch_size=32,
    callbacks=[checkpoint, early_stopping]
)

feature_extractor = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[-2].output
)

features_train = feature_extractor.predict(X_train, verbose=0)
features_test = feature_extractor.predict(X_test, verbose=0)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='recall')
grid_search.fit(features_train, y_train)

svm = grid_search.best_estimator_  # handles imbalance
svm.fit(features_train, y_train)

# --- Step 6: Evaluate ---
y_pred = svm.predict(features_test)
y_true = y_test
print(roc_auc_score(y_test, y_pred))
print(classification_report(y_true=y_true,y_pred=y_pred))

'''
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=['accuracy'])

# Define checkpoints and early stopping
checkpoint = ModelCheckpoint('updated_fall_risk.keras', save_best_only=True, monitor='loss', mode='min')
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Train the model with GRF data and labels
history = model.fit(
    X_train,
    y_train,  # Use actual labels here
    epochs=200,
    batch_size=32,
    callbacks=[checkpoint, early_stopping]
) '''

'''
y_pred_probs = model.predict(X_test)  # Get probabilities (output of sigmoid layer)
print(roc_auc_score(y_test, y_pred_probs))
y_pred = (y_pred_probs > 0.5).astype(int)  # Convert probabilities to binary (0 or 1) using a threshold of 0.5
y_true = y_test  # Ground truth labels
print(len(y_true))
# Calculate metrics
print(classification_report(y_true=y_true,y_pred=y_pred)) '''

'''
v1, WearGait-PD CNN architecture (AUC: 0.615, True recall: 0.42)
v2, WearGait-PD CNN with kernel size 5 (AUC: 0.642, True recall: 0.45)
- For fun, I tried splits without subject leakage handling, AUC:  0.749, True recall: 0.58 for these
v3, made gait-instability class trial-based instead of subject-based (AUC: 0.650, True recall: 0.42)
v4, WearGait-PD CNN with kernel size 7 (AUC: 0.683, True recall: 0.55 )
v5, WearGait-PD CNN with kernel size 61 (AUC: 0.755, True recall: 0.55) # Can start tuning threshold now
v6, same CNN but with gyr-y and gyr-z data added, kernel size 7 (AUC: 0.733, True recall: 0.52)
v7, kept 3 axis gyroscope data and changed kernel size to 61 (AUC: 0.696, True recall: 0.29)
v8, kept everything else but changed kernel size to 3 (AUC: 0.714, True recall: 0.5)
v9, Tried kernel size 5 (AUC: 0.730, True recall: 0.48)
v10, kernel size 9 (AUC: 0.749, True recall: 0.58)
v11, new CNN architecture with three convolutional layers, instead of one (AUC: 0.711, True recall: 0.41) # I think it overfit
- Using dense layers make the CNN's overfit, since they both have around 4 million parameters
v12, kept the same three conv architecture, removed dense layers and attached svm to it (AUC: 0.690, True recall: 0.52)
'''




'''
Kernel Size 61
              precision    recall  f1-score   support

           0       0.71      0.77      0.74       143
           1       0.62      0.55      0.59       100

    accuracy                           0.68       243
   macro avg       0.67      0.66      0.66       243
weighted avg       0.67      0.68      0.68       243


'''