import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# =====================================================================
# 1. SYNTHETIC OCEAN DATA GENERATOR (Optimized)
# =====================================================================
class OceanInstabilityGenerator:
    def __init__(self, num_years=30, seed=None):
        self.num_years = num_years
        self.total_months = num_years * 12
        self.time = np.arange(self.total_months)
        self.rng = np.random.default_rng(seed) 
        
        self.variables = ['SST', 'SSS', 'SSH', 'U_curr', 'V_curr', 'Wind_Curl']
        self.num_vars = len(self.variables)

    def generate_climatology(self):
        means =       np.array([12.0,  34.5,  0.1,  0.2,  0.1,  1e-7]) 
        amplitudes =  np.array([ 4.0,   0.4,  0.1,  0.1,  0.05, 0.5e-7])
        phases =      np.array([ 0.0,   1.5,  0.5, -0.5,  0.0,  1.0])
        
        climatology = np.zeros((self.total_months, self.num_vars))
        for i in range(self.num_vars):
            climatology[:, i] = means[i] + amplitudes[i] * np.sin(2 * np.pi * self.time / 12 + phases[i])
        return climatology

    def generate_correlated_noise(self, rho=0.7):
        noise = np.zeros((self.total_months, self.num_vars))
        epsilon = self.rng.normal(0, 0.2, size=(self.total_months, self.num_vars))
        
        for t in range(1, self.total_months):
            noise[t] = rho * noise[t-1] + (1 - rho) * epsilon[t]
        return noise

    def generate_collapse_event(self, collapse_year=15, severity=-3.0, lambda_rate=0.15):
        t_c = collapse_year * 12
        instability = np.zeros((self.total_months, self.num_vars))
        impact_vector = np.array([1.0, 0.8, -0.5, 0.6, 0.4, 0.2]) * severity
        
        for t in range(self.total_months):
            if t >= t_c:
                factor = 1 - np.exp(-lambda_rate * (t - t_c))
                instability[t] = impact_vector * factor
                
        return instability, t_c

    def make_recordings(self, has_collapse=True, collapse_year=15):
        climatology = self.generate_climatology()
        noise = self.generate_correlated_noise()
        
        if has_collapse:
            instability, t_c = self.generate_collapse_event(collapse_year)
        else:
            instability = np.zeros_like(climatology)
            t_c = None
            
        return climatology + noise + instability, t_c

# =====================================================================
# 2. FEATURE EXTRACTION & SCALING
# =====================================================================
def extract_climatological_anomalies(raw_data, baseline_years=10):
    total_months, num_vars = raw_data.shape
    baseline_months = baseline_years * 12
    baseline_period = raw_data[:baseline_months]
    
    monthly_climatology = np.zeros((12, num_vars))
    for m in range(12):
        monthly_climatology[m] = np.mean(baseline_period[m::12], axis=0)
        
    anomalies = np.zeros_like(raw_data)
    for t in range(total_months):
        m = t % 12
        anomalies[t] = raw_data[t] - monthly_climatology[m]
        
    return anomalies

# =====================================================================
# 3. PYTORCH DATASET
# =====================================================================
class OceanAnomalyDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=36, num_years=50, prediction_horizon=0):
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        self.sequences = []
        self.labels = []
        self.valid_indices = []
        
        generator = OceanInstabilityGenerator(num_years=num_years)
        
        print("Generating sequences...")
        all_anomalies = []
        for seq_idx in range(num_samples):
            has_collapse = np.random.rand() > 0.5
            col_year = np.random.randint(11, num_years - 5) if has_collapse else None
            
            raw_data, t_c = generator.make_recordings(has_collapse, col_year)
            anomalies = extract_climatological_anomalies(raw_data)
            all_anomalies.append(anomalies)
            
            for t in range(seq_len, len(anomalies) - prediction_horizon):
                self.valid_indices.append((seq_idx, t))
                
                target_t = t + prediction_horizon
                label = 1.0 if (has_collapse and target_t >= t_c) else 0.0
                self.labels.append(label)

        stacked_anomalies = np.vstack(all_anomalies)
        self.mean = np.mean(stacked_anomalies, axis=0)
        self.std = np.std(stacked_anomalies, axis=0) + 1e-8
        
        for i in range(num_samples):
            self.sequences.append((all_anomalies[i] - self.mean) / self.std)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        seq_idx, t = self.valid_indices[idx]
        window = self.sequences[seq_idx][t - self.seq_len:t]
        
        # Keep tensors on CPU here; DataLoader will fetch them, then we move to GPU in the training loop
        x = torch.tensor(window, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

# =====================================================================
# 4. TRANSFORMER ARCHITECTURE
# =====================================================================
class OceanInstabilityTransformer(nn.Module):
    def __init__(self, num_features=6, d_model=32, nhead=4, num_layers=2, seq_len=36):
        super().__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model)) 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.input_projection(x) + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1) 
        
        return self.fc_out(x).squeeze(-1)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# =====================================================================
# 5. TRAINING LOOP (CUDA + VAL + PLOTTING)
# =====================================================================
if __name__ == "__main__":
    # Parameters
    SEQ_LEN = 36
    NUM_SAMPLES = 500  # Increased slightly for better curve smoothing
    EPOCHS = 15        # Increased to see convergence over time
    BATCH_SIZE = 64
    
    # 1. Define Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Accelerator Status ---")
    print(f"Using device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"--------------------------\n")
    
    # Generate Dataset
    dataset = OceanAnomalyDataset(num_samples=NUM_SAMPLES, seq_len=SEQ_LEN, num_years=50)
    
    # Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type == 'cuda'))
    
    # Initialize Model, Loss, and Optimizer
    model = OceanInstabilityTransformer(num_features=6, seq_len=SEQ_LEN).to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5) # Added minor weight decay for regularization
    
    # Tracking lists for plots
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print(f"Total sliding windows: {len(dataset)} (Train: {train_size}, Val: {val_size})")
    print("Starting training...\n")
    
    for epoch in range(EPOCHS):
        # -- TRAINING PHASE --
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # -- VALIDATION PHASE --
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                
                # Calculate Accuracy
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct_preds += (preds == batch_y).sum().item()
                total_preds += batch_y.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_preds / total_preds
        
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}%")

    print("\nTraining Complete. Plotting learning curves...")

    # -- PLOTTING METRICS --
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Training vs Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), history['train_loss'], label='Train Loss', marker='o')
    plt.plot(range(1, EPOCHS + 1), history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), history['val_acc'], label='Validation Accuracy', color='green', marker='o')
    plt.title('Validation Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("test_training.png")
    plt.show()