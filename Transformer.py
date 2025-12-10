# transformer_tuner_v1.py
# -----------------------------------------------------------
# Transformer (Encoder) forecast for "Target" using KerasTuner
# - Uses KerasTuner (Hyperband) to find optimal hyperparameters
# - Searches for embed_dim, num_heads, ff_dim, dropout, and learning_rate
# - Chronological split: 10% validation, 10% test (of TOTAL)
# - Metrics + visualizations (training curves, test plots)
# -----------------------------------------------------------

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D, Embedding
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt

# -------------------
# Configure
# -------------------
FILE_PATH = r"C:\Users\hunne\Desktop\Python projects\seminar Kode\LTSM.xlsx"
SHEET_NAME = 0          # change if needed
DATE_COL   = "date 2"     # will be dropped/ignored as a feature
TARGET_COL = "Target"     # the thing we forecast

LOOKBACK   = 12           # number of past rows per sample (e.g., months)
VAL_RATIO  = 0.10         # 10% of TOTAL data used for validation
TEST_RATIO = 0.10         # 10% of TOTAL data used for test
EPOCHS     = 20           # Max epochs for final training
BATCH_SIZE = 10

MODEL_DIR  = Path("transformer_tuner_artifacts") # New directory
MODEL_DIR.mkdir(exist_ok=True)

# -------------------
# Load & clean
# (This section is identical to the previous script)
# -------------------
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Excel not found: {FILE_PATH}")

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
df.columns = [str(c).strip() for c in df.columns]

# Keep a time index if possible (for plots)
dates_available = False
if DATE_COL in df.columns:
    try:
        _dt = pd.to_datetime(df[DATE_COL], errors="coerce")
        if _dt.notna().sum() > len(df) * 0.5:
            # Sort by date and keep a clean dates Series aligned to df
            tmp = df.assign(__date__=_dt).sort_values("__date__")
            dates = tmp["__date__"].reset_index(drop=True)
            df = tmp.drop(columns="__date__").reset_index(drop=True)
            dates_available = True
        else:
            # Not reliably parseable
            df = df.reset_index(drop=True)
    except Exception:
        df = df.reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

# Drop date col from features regardless
df = df.drop(columns=[DATE_COL], errors="ignore")

# Verify Target
if TARGET_COL not in df.columns:
    raise KeyError(f'"{TARGET_COL}" column not found. Columns present: {list(df.columns)}')

# Determine features (all except Target)
feature_cols = [c for c in df.columns if c != TARGET_COL]

# Force numeric to avoid ragged arrays; clean NaNs/infs
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
df[TARGET_COL]   = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)

# If we have dates, trim the dates series to match cleaned df length
if dates_available and len(dates) != len(df):
    # Simple fallback: disable dates if lengths diverge (e.g., if rows dropped)
    dates_available = False

# -------------------
# Arrays
# (This section is identical to the previous script)
# -------------------
X_raw = df[feature_cols].to_numpy(dtype=np.float32)
y_raw = df[[TARGET_COL]].to_numpy(dtype=np.float32)

n_total = len(df)
if n_total < (LOOKBACK + 50):
    raise RuntimeError(f"Not enough rows ({n_total}) for lookback={LOOKBACK}. "
                       f"Add more data or reduce LOOKBACK.")

# -------------------
# Chronological split
# (This section is identical to the previous script)
# -------------------
test_size = int(np.floor(n_total * TEST_RATIO))
val_size  = int(np.floor(n_total * VAL_RATIO))
train_end = n_total - val_size - test_size
val_end   = n_total - test_size

if train_end <= LOOKBACK:
    raise RuntimeError(
        f"Not enough train rows ({train_end}) after splits for LOOKBACK={LOOKBACK}. "
        f"Reduce LOOKBACK or ratios."
    )

X_train_raw = X_raw[:train_end]
y_train_raw = y_raw[:train_end]

X_val_raw   = X_raw[train_end:val_end]
y_val_raw   = y_raw[train_end:val_end]

X_test_raw  = X_raw[val_end:]
y_test_raw  = y_raw[val_end:]

# -------------------
# Scale
# (This section is identical to the previous script)
# -------------------
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train_raw)
X_val   = x_scaler.transform(X_val_raw)
X_test  = x_scaler.transform(X_test_raw)

y_train = y_scaler.fit_transform(y_train_raw)
y_val   = y_scaler.transform(y_val_raw)
y_test  = y_scaler.transform(y_test_raw)

# -------------------
# Sequence maker
# (This section is identical to the previous script)
# -------------------
def make_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        xi = X[i - lookback:i, :]
        yi = y[i, :]
        if xi.ndim != 2:
            raise ValueError(f"Bad window rank: got {xi.ndim}, expected 2; window end index {i}")
        Xs.append(xi)
        ys.append(yi)
    Xs = np.stack(Xs, axis=0).astype(np.float32)  # (n_samples, lookback, n_features)
    ys = np.stack(ys, axis=0).astype(np.float32)  # (n_samples, 1)
    return Xs, ys

def can_make_sequences(X, lookback):
    return len(X) > lookback

if not (can_make_sequences(X_train, LOOKBACK) and
        can_make_sequences(X_val,   LOOKBACK) and
        can_make_sequences(X_test,  LOOKBACK)):
    raise RuntimeError(
        "One of train/val/test is too short for the chosen LOOKBACK. "
        "Reduce LOOKBACK or adjust split ratios."
    )

Xtr, ytr = make_sequences(X_train, y_train, LOOKBACK)
Xva, yva = make_sequences(X_val,   y_val,   LOOKBACK)
Xte, yte = make_sequences(X_test,  y_test,  LOOKBACK)

# -------------------
# Sanity checks
# (This section is identical to the previous script)
# -------------------
# (Omitted for brevity, but it's the same as before)
print("\n=== Dataset split (chronological) ===")
print(f"Train shapes -> X:{Xtr.shape} y:{ytr.shape}")
print(f"Val shapes   -> X:{Xva.shape} y:{yva.shape}")
print(f"Test shapes  -> X:{Xte.shape} y:{yte.shape}")


# -------------------
# Model (CHANGED TO TRANSFORMER ENCODER)
# -------------------
tf.random.set_seed(42)
input_shape = (LOOKBACK, Xtr.shape[-1])

# --- Define Transformer Encoder Block as a function ---
def transformer_encoder(inputs, embed_dim, num_heads, ff_dim, dropout_rate):
    # 1. Multi-Head Attention
    attn = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=embed_dim // num_heads 
    )(query=inputs, value=inputs, key=inputs)
    attn = Dropout(dropout_rate)(attn)
    out1 = LayerNormalization()(inputs + attn)

    # 2. Feed Forward Network
    ffn = Sequential(
        [Dense(ff_dim, activation="relu"), Dense(embed_dim)], name="ffn"
    )
    ffn_output = ffn(out1)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    return LayerNormalization()(out1 + ffn_output)

# --- Define Model-Building Function for KerasTuner ---
def build_model(hp):
    # --- Tunable Hyperparameters ---
    embed_dim = hp.Int('embed_dim', min_value=16, max_value=64, step=16)
    num_heads = hp.Int('num_heads', min_value=2, max_value=4, step=2)
    ff_dim = hp.Int('ff_dim', min_value=16, max_value=64, step=16)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.3, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    # ---

    inputs = Input(shape=input_shape)
    
    # 1. Input Projection
    x = Dense(embed_dim)(inputs)

    # 2. Add Positional Encoding
    positions = tf.range(start=0, limit=LOOKBACK, delta=1)
    pos_embedding = Embedding(input_dim=LOOKBACK, output_dim=embed_dim)(positions)
    x = x + pos_embedding

    # 3. Transformer Encoder Block
    x = transformer_encoder(x, embed_dim, num_heads, ff_dim, dropout_rate)

    # 4. Output Head
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x) # Use same dropout
    x = Dense(hp.Int('dense_out', min_value=8, max_value=24, step=8), activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse",
                  metrics=["mae"])
    return model

# -------------------
# KerasTuner Search
# -------------------

# Note: You may need to run: pip install keras-tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=EPOCHS, # Use full epochs as max_epochs
    factor=3,
    directory='keras_tuner_dir',
    project_name='transformer_forecast_v1'
)

# Create a simple callback to stop search trials early
search_callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=10)
]

print("\n\n=== Starting Hyperparameter Search ===")
tuner.search(
    Xtr, ytr,
    validation_data=(Xva, yva),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=search_callbacks,
    shuffle=False
)
print("=== Hyperparameter Search Finished ===")

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n=== Best Hyperparameters Found ===")
print(f"Embed Dim: {best_hps.get('embed_dim')}")
print(f"Num Heads: {best_hps.get('num_heads')}")
print(f"FF Dim: {best_hps.get('ff_dim')}")
print(f"Dropout Rate: {best_hps.get('dropout_rate'):.2f}")
print(f"Dense Out: {best_hps.get('dense_out')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")

# -------------------
# Fit Best Model
# -------------------

# Build the model with the best HPs
model = tuner.hypermodel.build(best_hps)

# Define final training callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=200, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=50, min_lr=1e-5, verbose=1),
]

# Fit (no shuffle for time series)
effective_batch = min(BATCH_SIZE, max(1, len(Xtr)))
history = model.fit(
    Xtr, ytr,
    validation_data=(Xva, yva),
    epochs=EPOCHS, # Train for the full epochs
    batch_size=effective_batch,
    verbose=1,
    shuffle=False,
    callbacks=callbacks
)

# -------------------
# Evaluation
# -------------------
y_pred_test_scaled = model.predict(Xte)
y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled)
y_true_test = y_scaler.inverse_transform(yte)

mae  = mean_absolute_error(y_true_test, y_pred_test)
rmse = float(np.sqrt(mean_squared_error(y_true_test, y_pred_test)))
ss_res = np.sum((y_true_test - y_pred_test) ** 2)
ss_tot = np.sum((y_true_test - np.mean(y_true_test)) ** 2)
r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

print("\n=== Test Performance (Best Model) ===")
print(f"MAE : {mae:,.6f}")
print(f"RMSE: {rmse:,.6f}")
print(f"R^2 : {r2:,.6f}")

# -------------------
# One-step-ahead forecast from the latest window
# -------------------
X_all_scaled = np.concatenate([X_train, X_val, X_test], axis=0)
if len(X_all_scaled) <= LOOKBACK:
    raise RuntimeError("Not enough samples for a final prediction window.")
last_window = X_all_scaled[-LOOKBACK:, :]
last_window = np.expand_dims(last_window, axis=0) # Shape: (1, LOOKBACK, n_features)

next_pred_scaled = model.predict(last_window)
next_pred = y_scaler.inverse_transform(next_pred_scaled)[0, 0]
print(f"\nNext 1-step ahead forecast for '{TARGET_COL}': {next_pred:,.6f}")

# -------------------
# Save artifacts
# -------------------
model_path = MODEL_DIR / "transformer_best_forecaster.keras" # Changed filename
xsc_path   = MODEL_DIR / "x_scaler.joblib"
ysc_path   = MODEL_DIR / "y_scaler.joblib"

model.save(model_path)
dump(x_scaler, xsc_path)
dump(y_scaler, ysc_path)

print(f"\nSaved model to: {model_path}")
print(f"Saved X scaler to: {xsc_path}")
print(f"Saved y scaler to: {ysc_path}")

# ===========================================================
# Visualizations
# (This section is identical to the previous script)
# ===========================================================
# 1) Training history
hist = history.history
plt.figure()
plt.plot(hist.get("loss", []), label="train loss")
plt.plot(hist.get("val_loss", []), label="val loss")
plt.title("Training history (loss) - Best Model")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(MODEL_DIR / "training_history_loss.png", dpi=150)
plt.show()

# 2) Test set: Actual vs Predicted over time
y_true_1d = y_true_test.reshape(-1)
y_pred_1d = y_pred_test.reshape(-1)

# Build a time axis for test sequences (each label corresponds to a row index in test block starting at LOOKBACK)
if dates_available:
    # Global indices for test labels: from val_end + LOOKBACK ... n_total-1
    start_idx = val_end + LOOKBACK
    end_idx   = n_total  # exclusive for iloc slicing semantics
    if (end_idx - start_idx) == len(y_true_1d):
        x_time = pd.to_datetime(dates.iloc[start_idx:end_idx])
    else:
        # Fallback if mismatch due to earlier drops
        x_time = pd.RangeIndex(len(y_true_1d))
else:
    x_time = pd.RangeIndex(len(y_true_1d))

plt.figure()
plt.plot(x_time, y_true_1d, label="Actual")
plt.plot(x_time, y_pred_1d, label="Predicted")
plt.title("Test set: Actual vs. Predicted (Best Model)")
plt.xlabel("Time")
plt.ylabel(TARGET_COL)
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(MODEL_DIR / "test_actual_vs_pred.png", dpi=150)
plt.show()

# 3) Train set: Actual vs Predicted over time (NEW PLOT)
y_pred_train_scaled = model.predict(Xtr)
y_pred_train = y_scaler.inverse_transform(y_pred_train_scaled)
y_true_train = y_scaler.inverse_transform(ytr)

y_true_train_1d = y_true_train.reshape(-1)
y_pred_train_1d = y_pred_train.reshape(-1)

# Build a time axis for train sequences
if dates_available:
    # Global indices for train labels: from LOOKBACK ... train_end-1
    start_idx = LOOKBACK
    end_idx   = train_end  # exclusive for iloc slicing semantics
    if (end_idx - start_idx) == len(y_true_train_1d):
        x_time_train = pd.to_datetime(dates.iloc[start_idx:end_idx])
    else:
        # Fallback if mismatch due to earlier drops
        x_time_train = pd.RangeIndex(len(y_true_train_1d))
else:
    x_time_train = pd.RangeIndex(len(y_true_train_1d))

plt.figure()
plt.plot(x_time_train, y_true_train_1d, label="Actual (Train)")
plt.plot(x_time_train, y_pred_train_1d, label="Predicted (Train)", alpha=0.8)
plt.title("Train set: Actual vs. Predicted (Best Model)")
plt.xlabel("Time")
plt.ylabel(TARGET_COL)
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(MODEL_DIR / "train_actual_vs_pred.png", dpi=150)
plt.show()


# 4) Scatter: y_true vs y_pred with 45° line (was #3)
plt.figure()
plt.scatter(y_true_1d, y_pred_1d, alpha=0.8)
min_v = float(min(np.min(y_true_1d), np.min(y_pred_1d)))
max_v = float(max(np.max(y_true_1d), np.max(y_pred_1d)))
plt.plot([min_v, max_v], [min_v, max_v])
plt.title("Test set: y_true vs y_pred (Best Model)")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(MODEL_DIR / "test_scatter_true_vs_pred.png", dpi=150)
plt.show()

# 5) Residuals histogram (was #4)
residuals = y_true_1d - y_pred_1d
plt.figure()
plt.hist(residuals, bins=12)
plt.title("Test set residuals (Best Model)")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(MODEL_DIR / "test_residuals_hist.png", dpi=150)
plt.show()

# -------------------
# Optional: print model summary
# -------------------
print("\n\n=== Best Model Summary ===")
model.summary()