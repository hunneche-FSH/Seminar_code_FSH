
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# Import the new Callback functions
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Load Data
file_path = r"C:\Users\hunne\Desktop\Seminar kode\ML_ready.xlsx"
df = pd.read_excel(file_path)

# 2. Preprocessing
y = df.iloc[:, 1].values      # Target (2nd column)
X_raw = df.iloc[:, 2:].values # Features (3rd column onwards)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

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
input_neurons = X_train.shape[1]
print("-" * 30)
print(f"Data Split Summary:")
print(f"Input Neurons: {input_neurons}")
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print("-" * 30)

# 4. Build Model
model = Sequential()
model.add(Dense(1, input_dim=input_neurons, activation='relu'))
model.add(Dense(1, activation='linear'))

# 5. Compile
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 6. Define Callbacks
callbacks = [
    # Stop training if validation loss doesn't improve for 200 epochs
    EarlyStopping(
        monitor="val_loss", 
        mode="min", 
        patience=100, 
        restore_best_weights=True
    ),
    # Reduce Learning Rate by 50% if validation loss stagnates for 50 epochs
    ReduceLROnPlateau(
        monitor="val_loss", 
        mode="min", 
        factor=0.5, 
        patience=50, 
        min_lr=1e-5, 
        verbose=1
    ),
]

# 7. Train
# NOTE: Increased epochs to 500 so the callbacks have time to work
print("\nStarting training...")
history = model.fit(
    X_train, 
    y_train, 
    epochs=200,          # Increased from 50 to 500
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks, # Added callbacks here
    shuffle=True,
    verbose=1
)



# 8. Predictions for Graphing
pred_train = model.predict(X_train).flatten()
pred_val = model.predict(X_val).flatten()
pred_test = model.predict(X_test).flatten()



# --- 9. VISUALIZATION (5 GRAPHS) ---

sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(3, 2)

# Graph 1: Learning Curve (Loss)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('1. Model Learning Curve (MSE)')
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