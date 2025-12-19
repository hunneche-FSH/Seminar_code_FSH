import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random

# ==========================================
# --- USER CONFIGURATION (HYPERPARAMETERS) ---
# ==========================================
# 1. Data
FILE_PATH = r"C:\Users\hunne\Desktop\Seminar kode\ML_ready.xlsx"
TEST_SPLIT_PCT = 0.1
VAL_SPLIT_PCT = 0.1

# 2. Model Architecture
MODEL_NAME = "Linear Regression (OLS)"
# OLS has no hidden layers or neurons to configure

# 3. Settings
N_VIEW = 100            # How many days to zoom in on graphs
SEED = 42               # Reproducibility Seed

# ==========================================
# --- 1. SETUP SEED ---
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

set_seed(SEED)

print("-" * 30)
print(f"Running: {MODEL_NAME}")
print("-" * 30)

# --- 2. LOAD DATA ---
df = pd.read_excel(FILE_PATH)

# --- 3. PREPROCESSING (LEAKAGE FIX) ---
y = df.iloc[:, 1].values      # Target (2nd column)
X_raw = df.iloc[:, 2:].values # Features (3rd column onwards)

# A. Calculate Split Indices
n = len(df)
test_len = int(n * TEST_SPLIT_PCT)
val_len = int(n * VAL_SPLIT_PCT)
train_end = n - val_len - test_len
val_end = n - test_len

# B. Split Raw Data FIRST
X_train_raw = X_raw[:train_end]
y_train = y[:train_end]

X_val_raw = X_raw[train_end:val_end]
y_val = y[train_end:val_end]

X_test_raw = X_raw[val_end:]
y_test = y[val_end:]

# C. Scaling: Fit ONLY on Train, Transform others
# NOTE: OLS coefficients change with scaling, but predictions/MSE stay relative.
# We keep scaling to maintain strict comparability with the Neural Network inputs.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

# Info
print(f"OLS Benchmark Split Summary:")
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print("-" * 30)

# --- 4. BUILD & TRAIN MODEL ---
# OLS is a closed-form solution (instant calculation), so no epochs needed.
model = LinearRegression()

print("\nFitting OLS Model...")
model.fit(X_train, y_train)

# --- 5. PREDICTIONS ---
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)
# PRED TEST IS CALCULATED HERE - ONLY USED FOR FINAL METRICS
pred_test = model.predict(X_test)

# --- 6. PERFORMANCE EVALUATION TABLE (STRICTLY TEST DATA) ---
print("\n" + "="*50)
print("FINAL MODEL EVALUATION (TEST DATA ONLY)")
print("="*50)

# A. Standard Metrics
mse_score = mean_squared_error(y_test, pred_test)
mae_score = mean_absolute_error(y_test, pred_test)
r2_score_val = r2_score(y_test, pred_test)

# B. Directional Accuracy (DA) Logic
actual_signs = np.sign(y_test)
pred_signs = np.sign(pred_test)

# Total DA
correct_directions = (actual_signs == pred_signs)
da_total = np.mean(correct_directions)

# Positive DA
pos_mask = y_test > 0
if np.sum(pos_mask) > 0:
    da_pos = np.mean(correct_directions[pos_mask])
else:
    da_pos = np.nan

# Negative DA
neg_mask = y_test < 0
if np.sum(neg_mask) > 0:
    da_neg = np.mean(correct_directions[neg_mask])
else:
    da_neg = np.nan

# C. Create Table
results_data = {
    "Model": [MODEL_NAME],
    "Layers": ["0"], 
    "Neurons": ["N/A (Linear)"], 
    "MSE": [f"{mse_score:.5f}"],
    "MAE": [f"{mae_score:.5f}"],
    "R^2": [f"{r2_score_val:.4f}"],
    "DA Total": [f"{da_total:.2%}"],
    "DA Pos": [f"{da_pos:.2%}"],
    "DA Neg": [f"{da_neg:.2%}"]
}

results_df = pd.DataFrame(results_data)

# Display the table
print(results_df.to_string(index=False))
print("="*50 + "\n")


# --- 7. VISUALIZATION ---
sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(3, 2)

# Graph 1: Residual Distribution (Replaces Learning Curve)
residuals = y_test - pred_test
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(residuals, kde=True, ax=ax1, color='gray')
ax1.set_title('1. Residual Distribution (Test Set Errors)')
ax1.set_xlabel('Error (Actual - Predicted)')
ax1.set_ylabel('Frequency')

# Graph 2: Scatter Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, pred_test, alpha=0.5, color='green')
max_val = max(max(y_test), max(pred_test))
min_val = min(min(y_test), min(pred_test))
ax2.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)
ax2.set_title('2. Prediction Accuracy (Actual vs Predicted)')
ax2.set_xlabel('Actual Returns')
ax2.set_ylabel('Predicted Returns')

# Graph 3: Train Forecast
ax3 = fig.add_subplot(gs[1, 0])
if len(y_train) > N_VIEW:
    ax3.plot(y_train[-N_VIEW:], label='Actual', color='black', alpha=0.7)
    ax3.plot(pred_train[-N_VIEW:], label='Predicted', color='blue', linestyle='--')
else:
    ax3.plot(y_train, label='Actual', color='black', alpha=0.7)
    ax3.plot(pred_train, label='Predicted', color='blue', linestyle='--')
ax3.set_title(f'3. Training Data (Last {N_VIEW} Days)')
ax3.legend()

# Graph 4: Validation Forecast
ax4 = fig.add_subplot(gs[1, 1])
if len(y_val) > N_VIEW:
    ax4.plot(y_val[-N_VIEW:], label='Actual', color='black', alpha=0.7)
    ax4.plot(pred_val[-N_VIEW:], label='Predicted', color='orange', linestyle='--')
else:
    ax4.plot(y_val, label='Actual', color='black', alpha=0.7)
    ax4.plot(pred_val, label='Predicted', color='orange', linestyle='--')
ax4.set_title(f'4. Validation Data (Last {N_VIEW} Days)')
ax4.legend()

# Graph 5: Test Forecast
ax5 = fig.add_subplot(gs[2, :]) 
if len(y_test) > N_VIEW:
    ax5.plot(y_test[:N_VIEW], label='Actual', color='black', linewidth=2)
    ax5.plot(pred_test[:N_VIEW], label='Predicted', color='green', linestyle='--', linewidth=2)
else:
    ax5.plot(y_test, label='Actual', color='black', linewidth=2)
    ax5.plot(pred_test, label='Predicted', color='green', linestyle='--', linewidth=2)
ax5.set_title(f'5. Test Data (First {N_VIEW} Days - The Future)')
ax5.set_xlabel('Time Step')
ax5.set_ylabel('Log-Return')
ax5.legend()

plt.tight_layout()
plt.show()