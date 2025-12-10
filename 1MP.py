import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy 

# --- 1. SETUP DEVICE (GPU Check) ---
# This will find your RTX 5080 automatically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-" * 30)
print(f"Running on: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("-" * 30)

# --- 2. LOAD DATA ---
file_path = r"C:\Users\hunne\Desktop\Seminar kode\ML_ready.xlsx"
df = pd.read_excel(file_path)

# --- 3. PREPROCESSING ---
y = df.iloc[:, 1].values.reshape(-1, 1) 
X_raw = df.iloc[:, 2:].values

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Splitting
n = len(df)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

X_train_np = X_scaled[:train_end]
y_train_np = y[:train_end]
X_val_np = X_scaled[train_end:val_end]
y_val_np = y[train_end:val_end]
X_test_np = X_scaled[val_end:]
y_test_np = y[val_end:]

# Convert to Tensors and move to GPU
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_np, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val_np, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val_np, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

input_neurons = X_train.shape[1]
print(f"Data Split Summary:")
print(f"Input Neurons: {input_neurons}")
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print("-" * 30)

# --- 4. BUILD MODEL (Updated for 64 Neurons) ---
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        # Hidden Layer 1: 64 Neurons
        self.layer1 = nn.Linear(input_dim, 2000) 
        # Hidden Layer 2: 64 Neurons
        self.layer2 = nn.Linear(2000, 2000)
        # Output Layer: 1 Neuron (Linear prediction)
        self.output_layer = nn.Linear(2000, 1) 
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x)) 
        x = self.output_layer(x)
        return x

model = NeuralNet(input_neurons).to(device)

# --- 5. COMPILE ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5
)

# --- 6. TRAINING LOOP ---
patience = 200
best_val_loss = float('inf')
patience_counter = 0
best_model_weights = None

history = {'loss': [], 'val_loss': []}

epochs = 2000 # You can increase this if the model needs more time to converge
for epoch in range(epochs):
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
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f} - Val Loss: {val_loss_val:.6f}")

if best_model_weights:
    model.load_state_dict(best_model_weights)
    print("Restored best model weights.")

# --- 7. PREDICTIONS ---
model.eval()
with torch.no_grad():
    pred_train = model(X_train).cpu().numpy().flatten()
    pred_val = model(X_val).cpu().numpy().flatten()
    pred_test = model(X_test).cpu().numpy().flatten()

y_train_plot = y_train_np.flatten()
y_val_plot = y_val_np.flatten()
y_test_plot = y_test_np.flatten()

# --- 8. VISUALIZATION ---
sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(3, 2)

# Graph 1
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history['loss'], label='Training Loss')
ax1.plot(history['val_loss'], label='Validation Loss')
ax1.set_title('1. Model Learning Curve (MSE)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Graph 2
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test_plot, pred_test, alpha=0.5, color='green')
max_val = max(max(y_test_plot), max(pred_test))
min_val = min(min(y_test_plot), min(pred_test))
ax2.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)
ax2.set_title('2. Prediction Accuracy (Actual vs Predicted)')
ax2.set_xlabel('Actual Returns')
ax2.set_ylabel('Predicted Returns')

N_VIEW = 100 

# Graph 3
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(y_train_plot[-N_VIEW:], label='Actual', color='black', alpha=0.7)
ax3.plot(pred_train[-N_VIEW:], label='Predicted', color='blue', linestyle='--')
ax3.set_title(f'3. Training Data (Last {N_VIEW} Days)')
ax3.legend()

# Graph 4
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(y_val_plot[-N_VIEW:], label='Actual', color='black', alpha=0.7)
ax4.plot(pred_val[-N_VIEW:], label='Predicted', color='orange', linestyle='--')
ax4.set_title(f'4. Validation Data (Last {N_VIEW} Days)')
ax4.legend()

# Graph 5
ax5 = fig.add_subplot(gs[2, :]) 
ax5.plot(y_test_plot[:N_VIEW], label='Actual', color='black', linewidth=2)
ax5.plot(pred_test[:N_VIEW], label='Predicted', color='green', linestyle='--', linewidth=2)
ax5.set_title(f'5. Test Data (First {N_VIEW} Days - The Future)')
ax5.set_xlabel('Time Step')
ax5.set_ylabel('Log-Return')
ax5.legend()

plt.tight_layout()
plt.show()