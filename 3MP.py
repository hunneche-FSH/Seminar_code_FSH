import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy 
import math
import random

# ==========================================
# --- USER CONFIGURATION (HYPERPARAMETERS) ---
# ==========================================
# 1. Data
FILE_PATH = r"C:\Users\hunne\Desktop\Seminar kode\ML_ready_LSTM.xlsx"
TEST_SPLIT_PCT = 0.1
VAL_SPLIT_PCT = 0.1
SEQUENCE_LENGTH = 30 

# 2. Model Architecture
MODEL_NAME = "Transformer (Time Series)"
D_MODEL = 72            # Transformer model dimension
N_HEAD = 12             # Number of attention heads
NUM_LAYERS = 3          # Number of encoder layers
DROPOUT_RATE = 0.2

# 3. Training Settings
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 50           # Early stopping patience
FACTOR = 0.1            # Learning rate reduction factor
MIN_LR = 1e-5
N_VIEW = 100            # How many days to zoom in on graphs
SEED = 42               # Reproducibility Seed

# ==========================================
# --- 1. SETUP DEVICE & SEED ---
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-" * 30)
print(f"Running on: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("-" * 30)

# --- 2. LOAD DATA ---
df = pd.read_excel(FILE_PATH)

# --- 3. PREPROCESSING (LEAKAGE FIX) ---
# Target (Returns)
y = df.iloc[:, 1].values.reshape(-1, 1) 
# Features
X_raw = df.iloc[:, 2:].values

# A. Calculate Split Indices on RAW Data
n = len(df)
test_len = int(n * TEST_SPLIT_PCT)
val_len = int(n * VAL_SPLIT_PCT)
train_end = n - val_len - test_len
val_end = n - test_len

# B. Split Raw Data FIRST
X_train_raw = X_raw[:train_end]
y_train_raw = y[:train_end]

X_val_raw = X_raw[train_end:val_end]
y_val_raw = y[train_end:val_end]

X_test_raw = X_raw[val_end:]
y_test_raw = y[val_end:]

# C. Scale Features (Fit only on Train)
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train_raw)
X_val_scaled = scaler_x.transform(X_val_raw)
X_test_scaled = scaler_x.transform(X_test_raw)

# D. Scale Target (Fit only on Train)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_raw)
y_val_scaled = scaler_y.transform(y_val_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# --- SEQUENCE CREATION ---
def create_sequences(input_data, target_data, seq_length):
    xs, ys = [], []
    if len(input_data) <= seq_length:
        return np.array([]), np.array([])
        
    for i in range(len(input_data) - seq_length):
        x = input_data[i:(i + seq_length)]
        y = target_data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# E. Create Sequences Independently
X_train_np, y_train_np = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
X_val_np, y_val_np = create_sequences(X_val_scaled, y_val_scaled, SEQUENCE_LENGTH)
X_test_np, y_test_np = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)

# F. Convert to Tensors and move to GPU
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_np, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val_np, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val_np, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_np, dtype=torch.float32).to(device)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

input_dim = X_train.shape[2] 

print(f"Data Split Summary:")
print(f"Sequence Length: {SEQUENCE_LENGTH}")
print(f"Input Features: {input_dim}")
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print("-" * 30)

# --- 4. BUILD TRANSFORMER MODEL ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_linear(src) 
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        last_time_step = output[:, -1, :]
        prediction = self.output_linear(last_time_step)
        return prediction

model = TimeSeriesTransformer(
    input_dim=input_dim,
    d_model=D_MODEL,
    nhead=N_HEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT_RATE
).to(device)

# --- 5. COMPILE ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=FACTOR, patience=PATIENCE, min_lr=MIN_LR
)

# --- 6. TRAINING LOOP ---
best_val_loss = float('inf')
patience_counter = 0
best_model_weights = None

history = {'loss': [], 'val_loss': []}

for epoch in range(EPOCHS):
    # Training
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        val_loss_val = val_loss.item()

    history['loss'].append(epoch_loss)
    history['val_loss'].append(val_loss_val)

    scheduler.step(val_loss_val)

    # Early Stopping
    if val_loss_val < best_val_loss:
        best_val_loss = val_loss_val
        best_model_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0 
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.6f} - Val Loss: {val_loss_val:.6f}")

if best_model_weights:
    model.load_state_dict(best_model_weights)
    print("Restored best model weights.")

# --- 7. PREDICTIONS ---
model.eval()
with torch.no_grad():
    # Get raw scaled predictions from GPU
    pred_train_scaled = model(X_train).cpu().numpy()
    pred_val_scaled = model(X_val).cpu().numpy()
    pred_test_scaled = model(X_test).cpu().numpy()

# Inverse Transform Predictions (Scale back to real units)
pred_train = scaler_y.inverse_transform(pred_train_scaled).flatten()
pred_val = scaler_y.inverse_transform(pred_val_scaled).flatten()
pred_test = scaler_y.inverse_transform(pred_test_scaled).flatten()

# Inverse Transform Actuals (Scale back to real units)
y_train_plot = scaler_y.inverse_transform(y_train_np).flatten()
y_val_plot = scaler_y.inverse_transform(y_val_np).flatten()
y_test_plot = scaler_y.inverse_transform(y_test_np).flatten()

# --- 8. PERFORMANCE EVALUATION TABLE (STRICTLY TEST DATA) ---
print("\n" + "="*50)
print("FINAL MODEL EVALUATION (TEST DATA ONLY)")
print("="*50)

# A. Standard Metrics
mse_score = mean_squared_error(y_test_plot, pred_test)
mae_score = mean_absolute_error(y_test_plot, pred_test)
r2_score_val = r2_score(y_test_plot, pred_test)

# B. Directional Accuracy (DA) Logic
actual_signs = np.sign(y_test_plot)
pred_signs = np.sign(pred_test)

# Total DA
correct_directions = (actual_signs == pred_signs)
da_total = np.mean(correct_directions)

# Positive DA
pos_mask = y_test_plot > 0
if np.sum(pos_mask) > 0:
    da_pos = np.mean(correct_directions[pos_mask])
else:
    da_pos = np.nan

# Negative DA
neg_mask = y_test_plot < 0
if np.sum(neg_mask) > 0:
    da_neg = np.mean(correct_directions[neg_mask])
else:
    da_neg = np.nan

# C. Create Table
results_data = {
    "Model": [MODEL_NAME],
    "Layers": [NUM_LAYERS], 
    "Neurons": [f"{D_MODEL} dim / {N_HEAD} heads"], 
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

# --- 9. VISUALIZATION ---
sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(3, 2)

# Graph 1: Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history['loss'], label='Training Loss')
ax1.plot(history['val_loss'], label='Validation Loss')
ax1.set_title('1. Model Learning Curve (MSE - Scaled)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Graph 2: Scatter Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test_plot, pred_test, alpha=0.5, color='green')
if len(y_test_plot) > 0:
    max_val = max(np.max(y_test_plot), np.max(pred_test))
    min_val = min(np.min(y_test_plot), np.min(pred_test))
    ax2.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)
ax2.set_title('2. Prediction Accuracy (Actual vs Predicted)')
ax2.set_xlabel('Actual Returns')
ax2.set_ylabel('Predicted Returns')

# Graph 3: Train Forecast
ax3 = fig.add_subplot(gs[1, 0])
if len(y_train_plot) > N_VIEW:
    ax3.plot(y_train_plot[-N_VIEW:], label='Actual', color='black', alpha=0.7)
    ax3.plot(pred_train[-N_VIEW:], label='Predicted', color='blue', linestyle='--')
else:
    ax3.plot(y_train_plot, label='Actual', color='black', alpha=0.7)
    ax3.plot(pred_train, label='Predicted', color='blue', linestyle='--')
ax3.set_title(f'3. Training Data (Last {N_VIEW} Days)')
ax3.legend()

# Graph 4: Validation Forecast
ax4 = fig.add_subplot(gs[1, 1])
if len(y_val_plot) > N_VIEW:
    ax4.plot(y_val_plot[-N_VIEW:], label='Actual', color='black', alpha=0.7)
    ax4.plot(pred_val[-N_VIEW:], label='Predicted', color='orange', linestyle='--')
else:
    ax4.plot(y_val_plot, label='Actual', color='black', alpha=0.7)
    ax4.plot(pred_val, label='Predicted', color='orange', linestyle='--')
ax4.set_title(f'4. Validation Data (Last {N_VIEW} Days)')
ax4.legend()

# Graph 5: Test Forecast
ax5 = fig.add_subplot(gs[2, :]) 
if len(y_test_plot) > N_VIEW:
    ax5.plot(y_test_plot[:N_VIEW], label='Actual', color='black', linewidth=2)
    ax5.plot(pred_test[:N_VIEW], label='Predicted', color='green', linestyle='--', linewidth=2)
else:
    ax5.plot(y_test_plot, label='Actual', color='black', linewidth=2)
    ax5.plot(pred_test, label='Predicted', color='green', linestyle='--', linewidth=2)
ax5.set_title(f'5. Test Data (First {N_VIEW} Days - The Future)')
ax5.set_xlabel('Time Step')
ax5.set_ylabel('Log-Return')
ax5.legend()

plt.tight_layout()
plt.show()