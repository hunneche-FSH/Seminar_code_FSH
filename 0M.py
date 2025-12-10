import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
y = df.iloc[:, 1].values      # Target (2nd column)
X_raw = df.iloc[:, 2:].values # Features (3rd column onwards)

# Scaling
# NOTE: While OLS coefficients change with scaling, the predictions/MSE remain the same.
# We keep scaling to maintain strict comparability with the Neural Network inputs.
X_scaled = X_raw

# 3. Splitting (80% Train, 10% Val, 10% Test)
n = len(df)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

X_train = X_scaled[:train_end]
y_train = y[:train_end]

X_val = X_scaled[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X_scaled[val_end:]
y_test = y[val_end:]

# Info
print("-" * 30)
print(f"OLS Benchmark Split Summary:")
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print("-" * 30)

# 4. Build & Train Model
# OLS is a closed-form solution (instant calculation), so no epochs needed.
model = LinearRegression()

print("\nFitting OLS Model...")
model.fit(X_train, y_train)

# 5. Predictions
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)
pred_test = model.predict(X_test)

# 6. Calculate Metrics (To Compare with Neural Networks)
train_mse = mean_squared_error(y_train, pred_train)
val_mse = mean_squared_error(y_val, pred_val)
test_mse = mean_squared_error(y_test, pred_test)

print("\n" + "="*30)
print("BENCHMARK RESULTS (MSE)")
print(f"Train MSE: {train_mse:.6f}")
print(f"Val MSE:   {val_mse:.6f}")
print(f"Test MSE:  {test_mse:.6f}")
print("="*30 + "\n")

# --- 9. VISUALIZATION (5 GRAPHS) ---

sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(3, 2)

# Graph 1: Residual Distribution (Replaces Learning Curve)
# In Econometrics, we want this to look like a Bell Curve (Normal Distribution)
residuals = y_test - pred_test
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(residuals, kde=True, ax=ax1, color='gray')
ax1.set_title('1. Residual Distribution (Test Set Errors)')
ax1.set_xlabel('Error (Actual - Predicted)')
ax1.set_ylabel('Frequency')

# Graph 2: Scatter Plot (Accuracy)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, pred_test, alpha=0.5, color='green')
max_val = max(max(y_test), max(pred_test))
min_val = min(min(y_test), min(pred_test))
ax2.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)
ax2.set_title(f'2. Prediction Accuracy (Test MSE: {test_mse:.5f})')
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