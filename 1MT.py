import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt  # Import Keras Tuner

# 1. Load Data
file_path = r"C:\Users\hunne\OneDrive - University of Copenhagen\9. Semester\Seminar financial economtri\ML_ready.xlsx"
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

X_train = X_scaled[:train_end]
y_train = y[:train_end]

X_val = X_scaled[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X_scaled[val_end:]
y_test = y[val_end:]

input_neurons = X_train.shape[1]

# --- 4. DEFINE HYPERPARAMETER SEARCH SPACE ---

def build_model(hp):
    model = keras.Sequential()
    
    # Tune the number of hidden layers (Choose between 1 and 5 layers)
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(layers.Dense(
            # Tune number of neurons in each layer (Choose between 32 and 256)
            units=hp.Int(f'units_{i}', min_value=8, max_value=64, step=8),
            activation='relu'
        ))
    
    # Output Layer
    model.add(layers.Dense(1, activation='linear'))
    
    # Tune the learning rate (Choose between 0.01, 0.001, or 0.0001)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    return model

# --- 5. INITIALIZE TUNER ---

# Hyperband is an efficient algorithm that quickly discards bad models
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=50,
    factor=3,
    directory='my_dir',       # Where to save logs
    project_name='intro_to_kt'
)

# Define Callbacks for the Search Phase
stop_early = EarlyStopping(monitor='val_loss', patience=10)

print("\n--- Starting Hyperparameter Search ---")
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. 
The optimal number of layers is {best_hps.get('num_layers')}.
The optimal learning rate is {best_hps.get('learning_rate')}.
""")

# --- 6. TRAIN THE BEST MODEL ---

print("\n--- Training the Best Model ---")
# Build the model with the best hyperparameters found
model = tuner.hypermodel.build(best_hps)

# Define Callbacks for the Final Training (Longer patience)
callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=200, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=50, min_lr=1e-6, verbose=1),
]

# Train
history = model.fit(
    X_train, 
    y_train, 
    epochs=500,
    batch_size=64,
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
ax1.set_title('1. Model Learning Curve (MSE)')
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