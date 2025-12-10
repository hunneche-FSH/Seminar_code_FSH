import os
# Force Intel oneDNN optimizations for your Ultra 7 CPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt

# 1. Load Data
# NOTE: Ensure the path is correct for your machine
file_path = r"C:\Users\hunne\OneDrive - University of Copenhagen\9. Semester\Seminar financial economtri\ML_ready.xlsx"

try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    # Fallback for demonstration if file doesn't exist in environment
    print("File not found. Generatng dummy data for demonstration...")
    dates = pd.date_range(start='2020-01-01', periods=1000)
    df = pd.DataFrame({
        'Date': dates,
        'Target': np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000),
        'Feat1': np.random.rand(1000),
        'Feat2': np.random.rand(1000)
    })

# 2. Preprocessing
y = df.iloc[:, 1].values      # Target
X_raw = df.iloc[:, 2:].values # Features

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# --- NEW: LSTM DATA PREPARATION ---
# LSTMs need 3D input: [Samples, Time_Steps, Features]
# We create a sliding window of N_PAST days to predict the next day

N_PAST = 30  # Lookback window (e.g., look at past 60 days to predict today)

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_lstm, y_lstm = create_sequences(X_scaled, y, N_PAST)

# 3. Splitting (80% Train, 10% Val, 10% Test)
# We utilize the new sequence arrays
n = len(X_lstm)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

X_train = X_lstm[:train_end]
y_train = y_lstm[:train_end]

X_val = X_lstm[train_end:val_end]
y_val = y_lstm[train_end:val_end]

X_test = X_lstm[val_end:]
y_test = y_lstm[val_end:]

# Input shape for LSTM: (Time Steps, Number of Features)
input_shape = (X_train.shape[1], X_train.shape[2])

# --- 4. DEFINE HYPERPARAMETER SEARCH SPACE ---

def build_lstm_model(hp):
    model = keras.Sequential()
    
    # Tune number of LSTM layers
    num_layers = hp.Int('num_layers', 1, 4)
    
    for i in range(num_layers):
        # Determine if we need to return sequences
        # We must return sequences if it is NOT the last LSTM layer
        is_last_layer = (i == num_layers - 1)
        
        model.add(layers.LSTM(
            units=hp.Int(f'units_{i}', min_value=8, max_value=32, step=4),
            activation='tanh', # tanh is standard for LSTM
            return_sequences=not is_last_layer, 
            input_shape=input_shape if i == 0 else None
        ))
        
        # Optional: Add Dropout to prevent overfitting
        if hp.Boolean(f'dropout_{i}'):
            model.add(layers.Dropout(rate=0.2))
    
    # Output Layer
    model.add(layers.Dense(1, activation='linear'))
    
    # Tune the learning rate
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    return model

# --- 5. INITIALIZE TUNER ---

tuner = kt.Hyperband(
    build_lstm_model,
    objective='val_loss',
    max_epochs=20, # Reduced slightly as LSTMs take longer to train
    factor=3,
    directory='lstm_dir',
    project_name='lstm_tuning'
)

stop_early = EarlyStopping(monitor='val_loss', patience=5)

print("\n--- Starting LSTM Hyperparameter Search ---")
tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. 
The optimal number of layers is {best_hps.get('num_layers')}.
The optimal learning rate is {best_hps.get('learning_rate')}.
""")

# --- 6. TRAIN THE BEST MODEL ---

print("\n--- Training the Best LSTM Model ---")
model = tuner.hypermodel.build(best_hps)

callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=50, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=20, min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_train, 
    y_train, 
    epochs=200, # Adjusted for LSTM training time
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    shuffle=False, # IMPORTANT: Often kept False for time series to maintain order, though debatable depending on sequence length
    verbose=1
)

# --- 7. VISUALIZATION ---

pred_train = model.predict(X_train).flatten()
pred_val = model.predict(X_val).flatten()
pred_test = model.predict(X_test).flatten()

# 6. Calculate Metrics (To Compare with Neural Networks)
train_mse = mean_squared_error(y_train, pred_train)
val_mse = mean_squared_error(y_val, pred_val)
test_mse = mean_squared_error(y_test, pred_test)

sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(3, 2)

# Graph 1: Learning Curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('1. LSTM Learning Curve (MSE)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Graph 2: Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, pred_test, alpha=0.5, color='green')
max_val = max(max(y_test), max(pred_test))
min_val = min(min(y_test), min(pred_test))
ax2.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)
ax2.set_title('2. Prediction Accuracy (Actual vs Predicted)')
ax2.set_xlabel('Actual Returns')
ax2.set_ylabel('Predicted Returns')

N_VIEW = 100 

# Graph 3: Train Zoom
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(y_train[-N_VIEW:], label='Actual', color='black', alpha=0.7)
ax3.plot(pred_train[-N_VIEW:], label='Predicted', color='blue', linestyle='--')
ax3.set_title(f'3. Training Data (Last {N_VIEW} Days)')
ax3.legend()

# Graph 4: Val Zoom
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(y_val[-N_VIEW:], label='Actual', color='black', alpha=0.7)
ax4.plot(pred_val[-N_VIEW:], label='Predicted', color='orange', linestyle='--')
ax4.set_title(f'4. Validation Data (Last {N_VIEW} Days)')
ax4.legend()

# Graph 5: Test Zoom
ax5 = fig.add_subplot(gs[2, :]) 
ax5.plot(y_test[:N_VIEW], label='Actual', color='black', linewidth=2)
ax5.plot(pred_test[:N_VIEW], label='Predicted', color='green', linestyle='--', linewidth=2)
ax5.set_title(f'5. Test Data (First {N_VIEW} Days)')
ax5.set_xlabel('Time Step')
ax5.set_ylabel('Log-Return')
ax5.legend()

plt.tight_layout()
plt.show()