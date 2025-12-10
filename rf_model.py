# rf_from_excel_v1.py
# -----------------------------------------------------------
# Random Forest forecast for "Target" from your Excel sheet.
# - Uses all columns except "date 2" and "Target" as features
# - Flattens the LOOKBACK window to create 2D tabular data
# - Chronological split: 10% validation, 10% test (of TOTAL)
# - Metrics + visualizations (feature importance, test plots)
# -----------------------------------------------------------

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import matplotlib.pyplot as plt

# -------------------
# Configure
# -------------------
FILE_PATH = r"C:\Users\hunne\Desktop\Python projects\seminar Kode\LTSM.xlsx"
SHEET_NAME = 0          # change if needed
DATE_COL   = "date 2"   # will be dropped/ignored as a feature
TARGET_COL = "Target"   # the thing we forecast

LOOKBACK   = 12          # number of past rows per sample (e.g., months)
VAL_RATIO  = 0.10       # 10% of TOTAL data used for validation
TEST_RATIO = 0.20       # 10% of TOTAL data used for test
# Note: EPOCHS and BATCH_SIZE are not needed for Random Forest

MODEL_DIR  = Path("rf_artifacts") # New directory for RF model
MODEL_DIR.mkdir(exist_ok=True)

# -------------------
# Load & clean
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
# -------------------
X_raw = df[feature_cols].to_numpy(dtype=np.float32)
y_raw = df[[TARGET_COL]].to_numpy(dtype=np.float32)

n_total = len(df)
if n_total < (LOOKBACK + 50):
    raise RuntimeError(f"Not enough rows ({n_total}) for lookback={LOOKBACK}. "
                       f"Add more data or reduce LOOKBACK.")

# -------------------
# Chronological split: exact 10% val + 10% test of TOTAL
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
# Scale (fit on TRAIN only)
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
# Sequence maker (safe)
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

# These are 3D (X) and 2D (y)
Xtr_3d, ytr_2d = make_sequences(X_train, y_train, LOOKBACK)
Xva_3d, yva_2d = make_sequences(X_val,   y_val,   LOOKBACK)
Xte_3d, yte_2d = make_sequences(X_test,  y_test,  LOOKBACK)

# -------------------
# *** NEW STEP: Flatten data for Random Forest ***
# -------------------
# RF needs 2D input: (n_samples, n_features * lookback)
# and 1D target: (n_samples,)

def flatten_data(X_3d, y_2d):
    n_samples = X_3d.shape[0]
    # Reshape X from (samples, lookback, features) to (samples, lookback * features)
    X_2d = X_3d.reshape(n_samples, -1) 
    # Reshape y from (samples, 1) to (samples,)
    y_1d = y_2d.ravel() 
    return X_2d, y_1d

Xtr, ytr = flatten_data(Xtr_3d, ytr_2d)
Xva, yva = flatten_data(Xva_3d, yva_2d)
Xte, yte = flatten_data(Xte_3d, yte_2d)

# -------------------
# Sanity checks
# -------------------
def _chk(name, Xs, ys):
    assert Xs.ndim == 2, f"{name} X ndim {Xs.ndim}" # Must be 2D
    assert ys.ndim == 1, f"{name} y ndim {ys.ndim}" # Must be 1D
    assert Xs.shape[1] == LOOKBACK * X_raw.shape[1], f"{name} feature mismatch {Xs.shape}"
    assert Xs.dtype == np.float32 and ys.dtype == np.float32, f"{name} dtypes {Xs.dtype}, {ys.dtype}"
    print(f"{name} shapes -> X:{Xs.shape} y:{ys.shape}")

print("\n=== Dataset split (chronological) ===")
print(f"Total rows : {n_total}")
print(f"Train rows : {train_end}  ({train_end/n_total:.1%})")
print(f"Val rows   : {val_size}   ({val_size/n_total:.1%})")
print(f"Test rows  : {test_size}  ({test_size/n_total:.1%})")
print(f"LOOKBACK   : {LOOKBACK}")
print(f"Raw features : {X_raw.shape[1]}")
print(f"Flattened features: {Xtr.shape[1]} (Lookback * Raw features)")

_chk("Train", Xtr, ytr)
_chk("Val",   Xva, yva)
_chk("Test",  Xte, yte)
print("Any NaNs in Xtr?", np.isnan(Xtr).any(), " | ytr?", np.isnan(ytr).any())


# -------------------
# Model (Random Forest)
# -------------------
# We combine Train and Validation sets, as RF doesn't need
# a validation set for early stopping. We can use the Test set for final eval.
# Or, for a more robust setup, we could use the validation set
# for hyperparameter tuning (GridSearch), but this is a simple replacement.
print("\nCombining Train and Validation sets for RF training...")
X_train_full = np.concatenate([Xtr, Xva], axis=0)
y_train_full = np.concatenate([ytr, yva], axis=0)
print(f"Full Train shapes -> X:{X_train_full.shape} y:{y_train_full.shape}")

print("\nTraining Random Forest model...")
model = RandomForestRegressor(
    n_estimators=10,       # Number of trees
    random_state=42,        # For reproducibility
    n_jobs=-1,              # Use all available CPU cores
    max_features='sqrt',    # A good default for regression
    oob_score=True          # Use out-of-bag samples for a validation score
)

# Fit the model
model.fit(X_train_full, y_train_full)

print("Training complete.")
print(f"Model OOB Score (R^2 on unseen training data): {model.oob_score_:.6f}")


# -------------------
# Evaluation
# -------------------
# yte is scaled, shape (n_samples,)
# y_pred_test_scaled is scaled, shape (n_samples,)
y_pred_test_scaled = model.predict(Xte)

# Reshape both to (n_samples, 1) for y_scaler
y_true_for_scaler = yte.reshape(-1, 1)
y_pred_for_scaler = y_pred_test_scaled.reshape(-1, 1)

# Inverse transform to get original values
y_true_test = y_scaler.inverse_transform(y_true_for_scaler)
y_pred_test = y_scaler.inverse_transform(y_pred_for_scaler)

mae  = mean_absolute_error(y_true_test, y_pred_test)
rmse = float(np.sqrt(mean_squared_error(y_true_test, y_pred_test)))
ss_res = np.sum((y_true_test - y_pred_test) ** 2)
ss_tot = np.sum((y_true_test - np.mean(y_true_test)) ** 2)
r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

print("\n=== Test Performance ===")
print(f"MAE : {mae:,.6f}")
print(f"RMSE: {rmse:,.6f}")
print(f"R^2 : {r2:,.6f}")

# -------------------
# One-step-ahead forecast from the latest window
# -------------------
X_all_scaled = np.concatenate([X_train, X_val, X_test], axis=0)
if len(X_all_scaled) <= LOOKBACK:
    raise RuntimeError("Not enough samples for a final prediction window.")
    
# Get the last (lookback, features) window
last_window = X_all_scaled[-LOOKBACK:, :] 

# Flatten it to (1, lookback * features) for RF
last_window_flat = last_window.reshape(1, -1)

next_pred_scaled = model.predict(last_window_flat) # shape (1,)
next_pred = y_scaler.inverse_transform(next_pred_scaled.reshape(-1, 1))[0, 0]
print(f"\nNext 1-step ahead forecast for '{TARGET_COL}': {next_pred:,.6f}")


# -------------------
# Save artifacts
# -------------------
model_path = MODEL_DIR / "rf_target_forecaster.joblib" # Save as joblib
xsc_path   = MODEL_DIR / "x_scaler.joblib"
ysc_path   = MODEL_DIR / "y_scaler.joblib"

dump(model, model_path)
dump(x_scaler, xsc_path)
dump(y_scaler, ysc_path)

print(f"\nSaved model to: {model_path}")
print(f"Saved X scaler to: {xsc_path}")
print(f"Saved y scaler to: {ysc_path}")

# ===========================================================
# Visualizations
# ===========================================================
# 1) Feature Importance
#    We need to create the flattened feature names
original_feature_names = feature_cols
flattened_feature_names = []
for i in range(LOOKBACK, 0, -1): # e.g., 6, 5, ..., 1
    for feat in original_feature_names:
        flattened_feature_names.append(f"{feat}_t-{i}")

importances = model.feature_importances_
N_TOP_FEATURES = 25 # Show top 25 features
indices = np.argsort(importances)[-N_TOP_FEATURES:]

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [flattened_feature_names[i] for i in indices])
plt.title(f"Top {N_TOP_FEATURES} Feature Importances")
plt.xlabel("Importance")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(MODEL_DIR / "feature_importance.png", dpi=150)
plt.show()


# 2) Test set: Actual vs Predicted over time
y_true_1d = y_true_test.ravel()
y_pred_1d = y_pred_test.ravel()

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
plt.plot(x_time, y_pred_1d, label="Predicted", linestyle="--", alpha=0.8)
plt.title("Test set: Actual vs. Predicted")
plt.xlabel("Time")
plt.ylabel(TARGET_COL)
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(MODEL_DIR / "test_actual_vs_pred.png", dpi=150)
plt.show()

# 3) Scatter: y_true vs y_pred with 45° line
plt.figure()
plt.scatter(y_true_1d, y_pred_1d, alpha=0.8)
min_v = float(min(np.min(y_true_1d), np.min(y_pred_1d)))
max_v = float(max(np.max(y_true_1d), np.max(y_pred_1d)))
plt.plot([min_v, max_v], [min_v, max_v], 'r--')
plt.title("Test set: y_true vs y_pred")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(MODEL_DIR / "test_scatter_true_vs_pred.png", dpi=150)
plt.show()

# 4) Residuals histogram
residuals = y_true_1d - y_pred_1d
plt.figure()
plt.hist(residuals, bins=12)
plt.title("Test set residuals")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(MODEL_DIR / "test_residuals_hist.png", dpi=150)
plt.show()

print("\nScript complete. Artifacts saved to 'rf_artifacts' directory.")