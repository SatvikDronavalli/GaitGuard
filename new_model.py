import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,make_scorer, recall_score
import numpy as np
from visualization_test import new_x, y_vals

# --- Configuration ---
INPUT_SHAPE = (101, 8)  # (timesteps, features)
NUM_CLASSES = 2         # Adjust based on your data

# --- Step 1: CNN Feature Extractor for 1D Time Series ---
class CNN1DFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNN1DFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.flattened_size = self._get_flattened_size()

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 8, 101)  # (batch, channels, time)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return x

# --- Step 2: Simulate or Load Data ---
num_samples = 1000
X = torch.tensor(new_x, dtype=torch.float32)  # (batch, timesteps, features)
y = y_vals # imbalanced label

# Reshape for CNN1D: (batch, channels, sequence_length)
X_cnn = X.permute(0, 2, 1)  # (1000, 8, 101)

# --- Step 3: Extract Features ---
cnn = CNN1DFeatureExtractor()
cnn.eval()

with torch.no_grad():
    features = cnn(X_cnn).numpy()  # shape: (1000, flattened_size)

# --- Step 4: Scale Features ---
scaler = StandardScaler()
features = scaler.fit_transform(features)

# --- Step 5: Split and Train SVM ---
X_train, X_test, y_train, y_test = train_test_split(
    features, y_vals, test_size=0.2, random_state=42, stratify=y_vals
)

param_grid = {
    'C': [0.1, 1, 10, 100], 
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='recall')
grid_search.fit(X_train, y_train)

svm = grid_search.best_estimator_  # handles imbalance
svm.fit(X_train, y_train)

# --- Step 6: Evaluate ---
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))
