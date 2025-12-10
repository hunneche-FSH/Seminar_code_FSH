import os
# Force Intel oneDNN optimizations for your Ultra 7 CPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout # Added LSTM and Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Load Data
file_path = r"C:\Users\hunne\OneDrive - University of Copenhagen\9. Semester\Seminar financial economtri\ML_ready_LSTM.xlsx"
df = pd.read_excel(file_path)

# 2. Preprocessing
y = df.iloc[:, 1].values      # Target
X_raw = df.iloc[:, 2:].values # Features

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 3. Splitting (80% Train, 10% Val, 10% Test)
n = len(df)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

X_train_raw = X_scaled[:train_end]
y_train_raw = y[:train_end]

X_val_raw = X_scaled[train_end:val_end]
y_val_raw = y[train_end:val_end]

X_test_raw = X_scaled[val_end:]
y_test_raw = y[val_end:]

# --- CRITICAL STEP FOR LSTM: SLIDING WINDOW ---
# We convert (Samples, Features) -> (Samples, TimeSteps, Features)
def create_sequences(X, y, time_steps=30):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        # Grab the previous 'time_steps' days of features
        Xs.append(X[i:(i + time_steps)])
        # Grab the target for the next day
        ys.append(y[i + time_steps]) 
    return np.array(Xs), np.array(ys)

# Define how far back the model looks (e.g., 10 days)
TIME_STEPS = 30 

X_train, y_train = create_sequences(X_train_raw, y_train_raw, TIME_STEPS)
X_val, y_val = create_sequences(X_val_raw, y_val_raw, TIME_STEPS)
X_test, y_test = create_sequences(X_test_raw, y_test_raw, TIME_STEPS)

# Info
# Shape is now: (Number of Samples, Time Steps, Number of Features)
print("-" * 30)
print(f"LSTM Data Split Summary:")
print(f"Input Shape: {X_train.shape}") 
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print("-" * 30)

# 4. Build LSTM Model
model = Sequential()

# Layer 1: LSTM
# return_sequences=True is needed if you stack LSTMs (pass sequence to next layer)
# input_shape=(Time Steps, Features)
model.add(LSTM(1280, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.5)) # Helps prevent overfitting

# Layer 2: LSTM
# return_sequences=False because the next layer is Dense (it needs a flat vector, not a sequence)
model.add(LSTM(640, return_sequences=False))
model.add(Dropout(0.5))

# Layer 3: Dense (Standard connection)
model.add(Dense(320, activation='relu'))

# Output Layer
model.add(Dense(1, activation='linear'))

# 5. Compile
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 6. Define Callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss", 
        mode="min", 
        patience=50, # Slightly reduced for LSTM as they can overfit faster
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss", 
        mode="min", 
        factor=0.5, 
        patience=20, 
        min_lr=1e-6, 
        verbose=1
    ),
]

# 7. Train
print("\nStarting training...")
history = model.fit(
    X_train, 
    y_train, 
    epochs=50, 
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks, 
    shuffle=False, # Important: For time series, often better to keep shuffle=False or handle carefully
    verbose=1
)

# 8. Predictions for Graphing
pred_train = model.predict(X_train).flatten()
pred_val = model.predict(X_val).flatten()
pred_test = model.predict(X_test).flatten()

# 6. Calculate Metrics (To Compare with Neural Networks)
train_mse = mean_squared_error(y_train, pred_train)
val_mse = mean_squared_error(y_val, pred_val)
test_mse = mean_squared_error(y_test, pred_test)

# --- 9. VISUALIZATION (5 GRAPHS) ---

sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(3, 2)

# Graph 1: Learning Curve (Loss)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('1. LSTM Learning Curve (MSE)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Graph 2: Scatter Plot (Accuracy)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, pred_test, alpha=0.5, color='green')
max_val = max(max(y_test), max(pred_test))
min_val = min(min(y_test), min(pred_test))
ax2.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)
ax2.set_title('2. Prediction Accuracy (Actual vs Predicted)')
ax2.set_xlabel('Actual Returns')
ax2.set_ylabel('Predicted Returns')

# Setting: How many points to zoom in on?
N_VIEW = 100 

# Graph 3: Training Performance (Zoomed)
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(y_train[-N_VIEW:], label='Actual', color='black', alpha=0.7)
ax3.plot(pred_train[-N_VIEW:], label='Predicted', color='blue', linestyle='--')
ax3.set_title(f'3. Training Data (Last {N_VIEW} Days)')
ax3.legend()

# Graph 4: Validation Performance (Zoomed)
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(y_val[-N_VIEW:], label='Actual', color='black', alpha=0.7)
ax4.plot(pred_val[-N_VIEW:], label='Predicted', color='orange', linestyle='--')
ax4.set_title(f'4. Validation Data (Last {N_VIEW} Days)')
ax4.legend()

# Graph 5: Test Performance (Zoomed)
ax5 = fig.add_subplot(gs[2, :]) 
ax5.plot(y_test[:N_VIEW], label='Actual', color='black', linewidth=2)
ax5.plot(pred_test[:N_VIEW], label='Predicted', color='green', linestyle='--', linewidth=2)
ax5.set_title(f'5. Test Data (First {N_VIEW} Days - The Future)')
ax5.set_xlabel('Time Step')
ax5.set_ylabel('Log-Return')
ax5.legend()

plt.tight_layout()
plt.show()