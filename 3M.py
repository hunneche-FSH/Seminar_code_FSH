import os
# Force Intel oneDNN optimizations for your Ultra 7 CPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Keras / Tensorflow imports
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler

# 1. Load Data
file_path = r"C:\Users\hunne\OneDrive - University of Copenhagen\9. Semester\Seminar financial economtri\ML_ready_LSTM.xlsx"
df = pd.read_excel(file_path)

# 2. Preprocessing
y = df.iloc[:, 1].values      # Target
X_raw = df.iloc[:, 2:].values # Features

# Scaling
scaler = RobustScaler()
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

# --- SLIDING WINDOW (Same as LSTM) ---
def create_sequences(X, y, time_steps=30):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 30 

X_train, y_train = create_sequences(X_train_raw, y_train_raw, TIME_STEPS)
X_val, y_val = create_sequences(X_val_raw, y_val_raw, TIME_STEPS)
X_test, y_test = create_sequences(X_test_raw, y_test_raw, TIME_STEPS)

print("-" * 30)
print(f"Transformer Data Split Summary:")
print(f"Input Shape: {X_train.shape}") 
print("-" * 30)

# --- 4. BUILD TRANSFORMER MODEL ---

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # 1. Attention and Normalization
    x = LayerNormalization(epsilon=1e-6)(inputs)
    
    # MultiHeadAttention: The core of the Transformer
    # It allows the model to focus on different parts of the history simultaneously
    x = MultiHeadAttention(
        key_dim=head_size, 
        num_heads=num_heads, 
        dropout=dropout
    )(x, x)
    
    x = Dropout(dropout)(x)
    res = x + inputs # Residual Connection (Skip connection)

    # 2. Feed Forward Part (Conv1D is often used in Time Series Transformers)
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    
    return x + res # Residual Connection

# Define Hyperparameters
input_shape = (TIME_STEPS, X_train.shape[2])
head_size = 128
num_heads = 3
ff_dim = 128
dropout = 0.2

# Build via Functional API
inputs = Input(shape=input_shape)

# Transformer Blocks
x = transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout)
x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout) # Stacking a second block

# Output Head
x = GlobalAveragePooling1D(data_format="channels_last")(x) # Flatten time dimension
x = Dense(64, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(1, activation="linear")(x)

model = Model(inputs, outputs)

# 5. Compile
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 6. Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=50, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=20, min_lr=1e-6, verbose=1),
]

# 7. Train
print("\nStarting Transformer training...")
history = model.fit(
    X_train, 
    y_train, 
    epochs=30, 
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks, 
    shuffle=True,
    verbose=1
)

# 8. Predictions
pred_train = model.predict(X_train).flatten()
pred_val = model.predict(X_val).flatten()
pred_test = model.predict(X_test).flatten()


# --- 9. VISUALIZATION ---

sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(3, 2)

# Graph 1: Learning Curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('1. Transformer Learning Curve (MSE)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Graph 2: Scatter Plot
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, pred_test, alpha=0.5, color='purple') # Changed color for variety
max_val = max(max(y_test), max(pred_test))
min_val = min(min(y_test), min(pred_test))
ax2.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)
ax2.set_title('2. Prediction Accuracy (Actual vs Predicted)')
ax2.set_xlabel('Actual Returns')
ax2.set_ylabel('Predicted Returns')

N_VIEW = 100 

# Graph 3: Training
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(y_train[-N_VIEW:], label='Actual', color='black', alpha=0.7)
ax3.plot(pred_train[-N_VIEW:], label='Predicted', color='blue', linestyle='--')
ax3.set_title(f'3. Training Data (Last {N_VIEW} Days)')
ax3.legend()

# Graph 4: Validation
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(y_val[-N_VIEW:], label='Actual', color='black', alpha=0.7)
ax4.plot(pred_val[-N_VIEW:], label='Predicted', color='orange', linestyle='--')
ax4.set_title(f'4. Validation Data (Last {N_VIEW} Days)')
ax4.legend()

# Graph 5: Test
ax5 = fig.add_subplot(gs[2, :]) 
ax5.plot(y_test[:N_VIEW], label='Actual', color='black', linewidth=2)
ax5.plot(pred_test[:N_VIEW], label='Predicted', color='purple', linestyle='--', linewidth=2)
ax5.set_title(f'5. Test Data (First {N_VIEW} Days - The Future)')
ax5.set_xlabel('Time Step')
ax5.set_ylabel('Log-Return')
ax5.legend()

plt.tight_layout()
plt.show()