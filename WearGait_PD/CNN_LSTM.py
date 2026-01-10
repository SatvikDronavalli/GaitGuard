from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from visualization_test import X_train, y_train
from tensorflow.keras.utils import plot_model

# Define the CNN-LSTM model for GRF data
model = Sequential([
    Input(shape=(101, 8)),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    Dropout(0.2),
    LSTM(64),  # LSTM after CNN
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('new_cnn_lstm.keras', save_best_only=True, monitor='loss', mode='min')
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=32,
    callbacks=[checkpoint, early_stopping]
)

# Save training history
pd.DataFrame(history.history).to_csv('../AI_Models/CNN/training_history_cnn_lstm.csv', index=False)

# Visualize model
#plot_model(model, to_file='cnn_lstm_model.png', show_shapes=True, show_layer_names=True)
