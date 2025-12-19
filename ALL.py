import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import math
import random

# ==========================================
# --- 1. CONFIGURATION & HYPERPARAMETERS ---
# ==========================================
CONFIG = {
    # File Paths for Datasets
    "PATH_FLAT": r"C:\..\ML_ready2.xlsx",       # Dataset for non-sequential models (OLS, FFNN)
    "PATH_SEQ":  r"C:\...\ML_ready_LSTM2.xlsx",  # Dataset for sequential models (LSTM, Transformer)
    
    # Data Splitting Parameters
    "TEST_SPLIT": 0.1,
    "VAL_SPLIT": 0.1,
    "SEQ_LENGTH": 30,        # Lookback window size (T) for time-series models
    "SEED": 123,             # Seed for reproducibility
    
    # Visualization Settings
    "ZOOM_VIEW": 200,        # Number of days to visualize in the final forecast plot
    
    # Training Hyperparameters
    "BATCH_SIZE": 128,
    "EPOCHS": 200,
    "PATIENCE": 150,         # Early stopping patience
    "LEARNING_RATE": 0.001,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
    # Architecture Specifics
    "FFNN": { "H1": 512, "H2": 256, "H3": 128 },
    "LSTM": { "HIDDEN_DIM":64, "LAYERS": 1, "DROPOUT": 0.5 },
    "TRANSFORMER": { "D_MODEL": 128, "N_HEAD": 8, "LAYERS": 2, "DROPOUT": 0.5 }
}

# ==========================================
# --- 2. REPRODUCIBILITY SETUP ---
# ==========================================
def set_seed(seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure 
    reproducible results across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(CONFIG["SEED"])
print(f"Running on Device: {CONFIG['DEVICE']}")

# ==========================================
# --- 3. DATA PREPROCESSING PIPELINE ---
# ==========================================

def get_split_indices(n_samples, config):
    """Calculates indices for Train/Validation/Test splits based on sample size."""
    test_len = int(n_samples * config["TEST_SPLIT"])
    val_len = int(n_samples * config["VAL_SPLIT"])
    train_end = n_samples - val_len - test_len
    val_end = n_samples - test_len
    return train_end, val_end

def process_flat_data(config):
    """
    Preprocesses data for OLS and FeedForward NN.
    Standardization is fitted on the training set only to strictly avoid look-ahead bias.
    """
    print(f"Loading Flat Data from: {config['PATH_FLAT']}")
    df = pd.read_excel(config["PATH_FLAT"])
    y = df.iloc[:, 1].values.reshape(-1, 1)
    X = df.iloc[:, 2:].values
    
    # Determine split indices
    train_end, val_end = get_split_indices(len(df), config)
    
    # Fit scaler ONLY on training data to prevent data leakage
    scaler_x = StandardScaler().fit(X[:train_end])
    scaler_y = StandardScaler().fit(y[:train_end])
    
    X_scaled = scaler_x.transform(X)
    y_scaled = scaler_y.transform(y) 
    
    data = {
        'train': (X_scaled[:train_end], y_scaled[:train_end]),
        'val':   (X_scaled[train_end:val_end], y_scaled[train_end:val_end]),
        'test':  (X_scaled[val_end:], y_scaled[val_end:])
    }
    return data, scaler_y, X.shape[1]

def process_seq_data(config):
    """
    Preprocesses data for LSTM and Transformer models.
    Generates sliding window sequences (Many-to-One).
    """
    print(f"Loading Sequence Data from: {config['PATH_SEQ']}")
    df = pd.read_excel(config["PATH_SEQ"])
    y = df.iloc[:, 1].values.reshape(-1, 1)
    X = df.iloc[:, 2:].values
    
    # Determine split indices
    train_end, val_end = get_split_indices(len(df), config)
    
    # Fit scaler ONLY on training data
    scaler_x = StandardScaler().fit(X[:train_end])
    scaler_y = StandardScaler().fit(y[:train_end])
    
    X_scaled = scaler_x.transform(X)
    y_scaled = scaler_y.transform(y)
    
    # Helper to generate sliding window sequences
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

    # Generate sequences globally
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, config["SEQ_LENGTH"])
    
    # Recalculate split indices to account for samples lost during windowing
    train_end_seq, val_end_seq = get_split_indices(len(X_seq), config)
    
    data = {
        'train': (X_seq[:train_end_seq], y_seq[:train_end_seq]),
        'val':   (X_seq[train_end_seq:val_end_seq], y_seq[train_end_seq:val_end_seq]),
        'test':  (X_seq[val_end_seq:], y_seq[val_end_seq:])
    }
    
    input_dim = X_seq.shape[2] if len(X_seq) > 0 else 0
    return data, scaler_y, input_dim

# ==========================================
# --- 4. MODEL ARCHITECTURES ---
# ==========================================

# --- A. Ordinary Least Squares (Baseline) ---
class OLSWrapper:
    def __init__(self):
        self.model = LinearRegression()
        self.name = "OLS (Linear)"
        self.color = "gray"
    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
        return {} # OLS does not have training history (epochs)
    def predict(self, X):
        return self.model.predict(X)

# --- B. FeedForward Neural Network ---
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, h1, h2, h3):
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, 1)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return self.out(x)

# --- C. Long Short-Term Memory (LSTM) ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # We use the output of the last time step for prediction
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

# --- D. Transformer (Encoder-Only) ---
class PositionalEncoding(nn.Module):
    """Injects sequence order information into the Transformer inputs."""
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

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)
    def forward(self, src):
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.output_linear(output[:, -1, :])

# --- Generic PyTorch Trainer Wrapper ---
class PyTorchTrainer:
    """
    Standardizes the training loop, validation, and early stopping 
    across all PyTorch models.
    """
    def __init__(self, model, name, color, config):
        self.model = model.to(config["DEVICE"])
        self.name = name
        self.color = color
        self.config = config
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        
    def train(self, X_train, y_train, X_val, y_val):
        # Prepare DataLoaders
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(self.config["DEVICE"]), 
                                 torch.tensor(y_train, dtype=torch.float32).to(self.config["DEVICE"]))
        train_loader = DataLoader(train_ds, batch_size=self.config["BATCH_SIZE"], shuffle=True)
        
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.config["DEVICE"])
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.config["DEVICE"])
        
        history = {'loss': [], 'val_loss': []}
        best_loss = float('inf')
        patience = 0
        best_weights = None
        
        print(f"Training {self.name}...")
        for epoch in range(self.config["EPOCHS"]):
            self.model.train()
            run_loss = 0.0
            for bx, by in train_loader:
                self.optimizer.zero_grad()
                out = self.model(bx)
                loss = self.criterion(out, by)
                loss.backward()
                self.optimizer.step()
                run_loss += loss.item() * bx.size(0)
            
            epoch_loss = run_loss / len(train_loader.dataset)
            
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val_t)
                val_loss = self.criterion(val_out, y_val_t).item()
            
            history['loss'].append(epoch_loss)
            history['val_loss'].append(val_loss)
            self.scheduler.step(val_loss)
            
            # Early Stopping Check
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= self.config["PATIENCE"]:
                    print(f"  -> Early stopping epoch {epoch+1}")
                    break
        
        # Restore best model weights
        if best_weights:
            self.model.load_state_dict(best_weights)
        return history

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.config["DEVICE"])
        with torch.no_grad():
            return self.model(X_t).cpu().numpy()

# ==========================================
# --- 5. EXECUTION & TRAINING PIPELINE ---
# ==========================================

# 1. Load and Process Datasets
flat_data, flat_scaler, flat_dim = process_flat_data(CONFIG)
seq_data, seq_scaler, seq_dim = process_seq_data(CONFIG)

# 2. Initialize Models
models = []

# Model 1: Ordinary Least Squares
models.append({
    'model': OLSWrapper(),
    'data': flat_data,
    'scaler': flat_scaler
})

# Model 2: FeedForward NN
ffnn_net = FeedForwardNN(flat_dim, CONFIG["FFNN"]["H1"], CONFIG["FFNN"]["H2"], CONFIG["FFNN"]["H3"])
models.append({
    'model': PyTorchTrainer(ffnn_net, "FeedForward NN", "blue", CONFIG),
    'data': flat_data,
    'scaler': flat_scaler
})

# Model 3: LSTM
lstm_net = LSTMModel(seq_dim, CONFIG["LSTM"]["HIDDEN_DIM"], CONFIG["LSTM"]["LAYERS"], CONFIG["LSTM"]["DROPOUT"])
models.append({
    'model': PyTorchTrainer(lstm_net, "LSTM", "orange", CONFIG),
    'data': seq_data,
    'scaler': seq_scaler
})

# Model 4: Transformer
trans_net = TransformerModel(seq_dim, CONFIG["TRANSFORMER"]["D_MODEL"], CONFIG["TRANSFORMER"]["N_HEAD"], CONFIG["TRANSFORMER"]["LAYERS"], CONFIG["TRANSFORMER"]["DROPOUT"])
models.append({
    'model': PyTorchTrainer(trans_net, "Transformer", "purple", CONFIG),
    'data': seq_data,
    'scaler': seq_scaler
})

# 3. Training Loop
histories = {}
raw_predictions = {} 

print("\n" + "="*50)
print("STARTING TRAINING PHASES")
print("="*50)

for m in models:
    name = m['model'].name
    data = m['data']
    
    # Train the model
    h = m['model'].train(data['train'][0], data['train'][1], data['val'][0], data['val'][1])
    histories[name] = h
    
    # Generate predictions on the test set
    preds = m['model'].predict(data['test'][0])
    
    # Inverse transform predictions to original scale (returns) for interpretation
    preds_real = m['scaler'].inverse_transform(preds).flatten()
    actual_real = m['scaler'].inverse_transform(data['test'][1]).flatten()
    
    raw_predictions[name] = {
        'pred': preds_real,
        'actual': actual_real
    }

# ==========================================
# --- 6. ALIGNMENT & METRIC EVALUATION ---
# ==========================================
# IMPORTANT: Sequential models (LSTM/Transformer) consume the first N samples 
# for window creation. This creates a length mismatch with OLS/FFNN test sets.
# We determine the minimum common length and slice all test sets from the END
# to ensure we are comparing predictions for the exact same calendar days.

min_len = min([len(v['actual']) for v in raw_predictions.values()])

print(f"\nAligning Test Sets to last {min_len} samples for fair comparison...")

results = []
aligned_preds = {}
final_actual = None

for name, res in raw_predictions.items():
    # Slice to enforce temporal alignment
    p_aligned = res['pred'][-min_len:]
    a_aligned = res['actual'][-min_len:]
    
    aligned_preds[name] = p_aligned
    
    # Set ground truth (derived from the aligned actuals)
    if final_actual is None:
        final_actual = a_aligned
    
    # Calculate Performance Metrics
    mse = mean_squared_error(a_aligned, p_aligned)
    mae = mean_absolute_error(a_aligned, p_aligned)
    r2 = r2_score(a_aligned, p_aligned)
    
    # Calculate Directional Accuracy (Hit Rate)
    actual_s = np.sign(a_aligned)
    pred_s = np.sign(p_aligned)
    correct = (actual_s == pred_s)
    da_total = np.mean(correct)
    
    pos_mask = a_aligned > 0
    neg_mask = a_aligned < 0
    da_pos = np.mean(correct[pos_mask]) if np.sum(pos_mask) > 0 else np.nan
    da_neg = np.mean(correct[neg_mask]) if np.sum(neg_mask) > 0 else np.nan
    
    results.append({
        "Model": name,
        "MSE": mse, 
        "MAE": mae, 
        "R2": r2, 
        "DA Total": da_total,
        "DA Pos": da_pos,
        "DA Neg": da_neg
    })

# ==========================================
# --- 7. REPORTING & VISUALIZATION ---
# ==========================================
res_df = pd.DataFrame(results)

# Format metrics for display
fmt_df = res_df.copy()
fmt_df["MSE"] = fmt_df["MSE"].apply(lambda x: f"{x:.5f}")
fmt_df["MAE"] = fmt_df["MAE"].apply(lambda x: f"{x:.5f}")
fmt_df["R2"] = fmt_df["R2"].apply(lambda x: f"{x:.4f}")
for col in ["DA Total", "DA Pos", "DA Neg"]:
    fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:.2%}")

print("\n" + "="*60)
print("FINAL MODEL LEADERBOARD (Aligned Test Data)")
print("="*60)
print(fmt_df.to_string(index=False))

# --- PLOTTING ---
sns.set_style("whitegrid")

# 1. Validation Loss Curves
plt.figure(figsize=(12, 6))
plt.title("Learning Curves (Validation Loss)")
for name, h in histories.items():
    if h and 'val_loss' in h:
        plt.plot(h['val_loss'], label=name)
plt.xlabel("Epoch")
plt.ylabel("MSE (Scaled)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Forecast Zoom (Visual Inspection)
plt.figure(figsize=(15, 8))
view = CONFIG["ZOOM_VIEW"]
plt.plot(final_actual[:view], label='Actual', color='black', linewidth=3, alpha=0.8)

colors = {'OLS (Linear)': 'gray', 'FeedForward NN': 'blue', 'LSTM': 'orange', 'Transformer': 'purple'}
for name, pred in aligned_preds.items():
    c = colors.get(name, 'green')
    plt.plot(pred[:view], label=name, linestyle='--', linewidth=1.5, color=c)

plt.title(f"Forecast Comparison (First {view} days of Test Set)")
plt.xlabel("Time Step")
plt.ylabel("Returns")
plt.legend()
plt.tight_layout()
plt.show()

# 3. Scatter Matrix (Prediction vs Actual)
cols = 2
rows = 2
fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
axes = axes.flatten()

for i, (name, pred) in enumerate(aligned_preds.items()):
    ax = axes[i]
    ax.scatter(final_actual, pred, alpha=0.3, color=colors.get(name, 'blue'))
    
    # 45-degree Identity Line (Perfect Prediction)
    mx = max(final_actual.max(), pred.max())
    mn = min(final_actual.min(), pred.min())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2)
    
    ax.set_title(f"{name}: Predicted vs Actual")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

plt.tight_layout()
plt.show()