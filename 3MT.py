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
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
import keras_tuner as kt

# 1. Load Data
file_path = r"C:\Users\hunne\OneDrive - University of Copenhagen\9. Semester\Seminar financial economtri\ML_ready.xlsx"

try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print("File not found. Generating dummy data for demonstration...")
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
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_raw)

# --- DATA PREPARATION ---
N_PAST = 60 

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_trans, y_trans = create_sequences(X_scaled, y, N_PAST)

# 3. Splitting (80% Train, 10% Val, 10% Test)
n = len(X_trans)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

X_train = X_trans[:train_end]
y_train = y_trans[:train_end]

X_val = X_trans[train_end:val_end]
y_val = y_trans[train_end:val_end]

X_test = X_trans[val_end:]
y_test = y_trans[val_end:]

# Input shape: (Time Steps, Number of Features)
input_shape = (X_train.shape[1], X_train.shape[2])

# --- 4. DEFINE TRANSFORMER HYPERPARAMETER SEARCH SPACE ---

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # 1. Attention and Normalization
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs # Residual Connection

    # 2. Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer_model(hp):
    inputs = Input(shape=input_shape)
    x = inputs

    # Tune: Number of Transformer Blocks (stacking layers)
    num_blocks = hp.Int('num_blocks', 1, 3) 
    
    # Tune: Hyperparameters inside the block
    head_size = hp.Int('head_size', min_value=32, max_value=128, step=32)
    num_heads = hp.Int('num_heads', min_value=2, max_value=6, step=2)
    ff_dim = hp.Int('ff_dim', min_value=32, max_value=128, step=32)
    dropout = hp.Float('dropout', min_value=0.1, max_value=0.3, step=0.1)

    # Create the stack of Transformer blocks
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Output Head
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    
    # Tune: Size of the dense layer after pooling
    dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    
    outputs = layers.Dense(1, activation="linear")(x)

    model = Model(inputs, outputs)

    # Tune learning rate
    lr = hp.Choice('learning_rate', values=[1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mse"],
    )
    return model

# --- 5. INITIALIZE TUNER ---

# Using Hyperband (efficient for finding good params quickly)
tuner = kt.Hyperband(
    build_transformer_model,
    objective='val_loss',
    max_epochs=50,      # Keep this low for the search phase on CPU
    factor=3,
    directory='transformer_dir',
    project_name='transformer_tuning_v1'
)

stop_early = EarlyStopping(monitor='val_loss', patience=3)

print("\n--- Starting Transformer Hyperparameter Search ---")
tuner.search(X_train, y_train, epochs=15, validation_data=(X_val, y_val), callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. 
Optimal Blocks: {best_hps.get('num_blocks')}
Optimal Heads: {best_hps.get('num_heads')}
Optimal Head Size: {best_hps.get('head_size')}
Optimal Learning Rate: {best_hps.get('learning_rate')}
""")

# --- 6. TRAIN THE BEST MODEL ---

print("\n--- Training the Best Transformer Model ---")
model = tuner.hypermodel.build(best_hps)

callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=30, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_train, 
    y_train, 
    epochs=100,  # Increased for final training
    batch_size=64, # Larger batch size helps CPU efficiency
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    shuffle=True,
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
ax1.set_title('1. Transformer Learning Curve (MSE)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Graph 2: Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, pred_test, alpha=0.5, color='purple')
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
ax5.plot(pred_test[:N_VIEW], label='Predicted', color='purple', linestyle='--', linewidth=2)
ax5.set_title(f'5. Test Data (First {N_VIEW} Days)')
ax5.set_xlabel('Time Step')
ax5.set_ylabel('Log-Return')
ax5.legend()

plt.tight_layout()
plt.show()