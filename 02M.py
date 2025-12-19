import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# 1. Load Data
file_path = r"C:\Users\hunne\OneDrive - University of Copenhagen\9. Semester\Seminar financial economtri\ML_ready.xlsx"



# 2. Select Target and "Naive" Prediction
# Target is usually Column 2 (index 1)
y = df.iloc[:, 1].values 

# User Request: "Column 59" is the prediction. 
# In 0-based indexing, Column 59 is index 58.
# We assume this column contains the "Long Term Average" or similar metric.
try:
    predictions_raw = df.iloc[:, 57].values
    print(f"Benchmark Column Selected: {df.columns[57]}")
except IndexError:
    print("Error: Dataset does not have 59 columns. Using last column as fallback.")
    predictions_raw = df.iloc[:, -1].values

# 3. Splitting (80% Train, 10% Val, 10% Test)
# We split to maintain the exact same comparison windows as the ML/OLS models
n = len(df)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

# Target Splits
y_train = y[:train_end]
y_val = y[train_end:val_end]
y_test = y[val_end:]

# Prediction Splits (Just slicing the column)
pred_train = predictions_raw[:train_end]
pred_val = predictions_raw[train_end:val_end]
pred_test = predictions_raw[val_end:]

# 4. Calculate Metrics
train_mse = mean_squared_error(y_train, pred_train)
val_mse = mean_squared_error(y_val, pred_val)
test_mse = mean_squared_error(y_test, pred_test)

print("\n" + "="*30)
print("NAIVE BENCHMARK RESULTS (MSE)")
print(f"Train MSE: {train_mse:.6f}")
print(f"Val MSE:   {val_mse:.6f}")
print(f"Test MSE:  {test_mse:.6f}")
print("="*30 + "\n")

# --- 5. VISUALIZATION (5 GRAPHS) ---

sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(3, 2)

# Graph 1: Residual Distribution
residuals = y_test - pred_test
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(residuals, kde=True, ax=ax1, color='gray')
ax1.set_title('1. Residual Distribution (Test Set Errors)')
ax1.set_xlabel('Error (Actual - Benchmark)')
ax1.set_ylabel('Frequency')

# Graph 2: Scatter Plot (Accuracy)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, pred_test, alpha=0.5, color='brown') # Brown for "Plain/Raw" data
max_val = max(max(y_test), max(pred_test))
min_val = min(min(y_test), min(pred_test))
ax2.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)
ax2.set_title(f'2. Prediction Accuracy (Test MSE: {test_mse:.5f})')
ax2.set_xlabel('Actual Returns')
ax2.set_ylabel('Benchmark (Col 59)')

# Setting: How many points to zoom in on?
N_VIEW = 100 

# Graph 3: Training Performance (Zoomed)
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(y_train[-N_VIEW:], label='Actual', color='black', alpha=0.7)
ax3.plot(pred_train[-N_VIEW:], label='Benchmark', color='blue', linestyle='--')
ax3.set_title(f'3. Training Data (Last {N_VIEW} Days)')
ax3.legend()

# Graph 4: Validation Performance (Zoomed)
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(y_val[-N_VIEW:], label='Actual', color='black', alpha=0.7)
ax4.plot(pred_val[-N_VIEW:], label='Benchmark', color='orange', linestyle='--')
ax4.set_title(f'4. Validation Data (Last {N_VIEW} Days)')
ax4.legend()

# Graph 5: Test Performance (Zoomed)
ax5 = fig.add_subplot(gs[2, :]) 
ax5.plot(y_test[:N_VIEW], label='Actual', color='black', linewidth=2)
ax5.plot(pred_test[:N_VIEW], label='Benchmark', color='brown', linestyle='--', linewidth=2)
ax5.set_title(f'5. Test Data (First {N_VIEW} Days - The Future)')
ax5.set_xlabel('Time Step')
ax5.set_ylabel('Log-Return')
ax5.legend()

plt.tight_layout()
plt.show()