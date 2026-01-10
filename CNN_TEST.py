from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from visualization_test import X_test, y_test
# Load the best model after training
best_model = load_model('updated_fall_risk.keras')

# Evaluate the best model on the testing data
y_pred_probs = best_model.predict(X_test)  # Get probabilities (output of sigmoid layer)
y_pred = (y_pred_probs > 0.5).astype(int)  # Convert probabilities to binary (0 or 1) using a threshold of 0.5
y_true = y_test  # Ground truth labels
print(len(y_true))
# Calculate metrics
print(classification_report(y_true=y_true,y_pred=y_pred))


  
