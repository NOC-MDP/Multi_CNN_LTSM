import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional,Tuple
import pickle
import pandas as pd
import math
# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Recording:
    """
    A single (multi-parameter) time-series recording.
    Data shape is now (P, T). Depth is removed.
    """
    data: np.ndarray 
    bifurcation_t: Optional[int]
    recording_id: str

class BifurcationType(Enum):
    VARIANCE = auto()
    MEAN_SHIFT = auto()
    HOPF = auto()

class HardNegativeType(Enum):
    STORM = auto()         # large variance burst
    EDDY = auto()          # temporary mean shift
    SEASONAL = auto()      # slow drift
    OSCILLATOR = auto()    # temporary oscillation

@dataclass
class DatasetConfig:
    n_recordings: int = 800
    num_params: int = 5
    base_time_len: int = 2048
    window_sizes: list[int] = field(default_factory=lambda: [64,128, 256, 512])
    dt: float = 0.1
    seed: int = 42

# ──────────────────────────────────────────────────────────────────────────────
# Core Redesigned Signal Generator
# ──────────────────────────────────────────────────────────────────────────────

class OceanStateGenerator:
    def __init__(self, cfg: DatasetConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def generate_latent_system(self, T: int, is_positive: bool, 
                               bif_type: Optional[BifurcationType], 
                               null_type: Optional[HardNegativeType]) -> Tuple[np.ndarray, int]:
        dt = self.cfg.dt
        
        # ──────────────────────────────────────────────────────────────────────
        # OPTION A: Domain Randomization of Latent SDE Constants
        # ──────────────────────────────────────────────────────────────────────
        # Instead of fixed numbers (0.05, 0.1), each record gets distinct environmental properties
        sigma = self.rng.uniform(0.03, 0.07)       # Noise intensity (Default was 0.05)
        theta0 = self.rng.uniform(0.07, 0.13)      # Base restoring force / inertia (Default was 0.1)
        
        strat_decay = self.rng.uniform(0.03, 0.07) # Stratification dissipation rate (Default was 0.05)
        coupling_strength = self.rng.uniform(0.12, 0.18) # Circulation -> Stratification coupling (Default was 0.15)
        meso_decay = self.rng.uniform(0.07, 0.13)  # Mesoscale dissipation rate (Default was 0.10)
        # ──────────────────────────────────────────────────────────────────────
        
        # Latent state z = [circulation, stratification, mesoscale]
        z = np.zeros((3, T))
        
        # Transition mechanics
        t_axis = np.arange(T)
        bif_center = int(T * self.rng.uniform(0.4, 0.8))
        ramp = 1 / (1 + np.exp(-(t_axis - bif_center) / 20.0))  # Sigmoid ramp
        r = np.zeros(T) # For Hopf oscillator
        
        # Hard negative distractors
        distractor = np.zeros((3, T))
        if not is_positive and null_type:
            event_t = int(T * self.rng.uniform(0.3, 0.7))
            if null_type == HardNegativeType.STORM:
                distractor[0, event_t:event_t+100] = self.rng.normal(0, 0.3, 100)
            elif null_type == HardNegativeType.EDDY:
                distractor[2, event_t:event_t+150] = 0.4  # Mean shift
            elif null_type == HardNegativeType.SEASONAL:
                distractor[1, :] = np.linspace(-0.5, 0.5, T) # Slow drift
            elif null_type == HardNegativeType.OSCILLATOR:
                distractor[0, event_t:event_t+100] = 0.3 * np.sin(np.arange(100) * 0.5)

        # Evolve coupled SDEs
        for t in range(1, T):
            # Dynamic restoring force for Critical Slowing Down (Variance)
            theta_t = theta0
            if is_positive and bif_type == BifurcationType.VARIANCE:
                theta_t = theta0 * (1 - ramp[t])
                theta_t = max(theta_t, 1e-3) # Cap to avoid strict division by zero
                
            # Dynamic Hopf Oscillator
            if is_positive and bif_type == BifurcationType.HOPF:
                mu = -0.1 + 0.2 * ramp[t] # Transitions from <0 to >0
                r[t] = r[t-1] + (mu * r[t-1] - r[t-1]**3) * dt + sigma * self.rng.standard_normal()
            
            # Latent Equations incorporating randomized parameters
            z[0, t] = z[0, t-1] + dt * (-theta_t * z[0, t-1]) + sigma * self.rng.standard_normal()
            z[1, t] = z[1, t-1] + dt * (-strat_decay * z[1, t-1] + coupling_strength * z[0, t-1]) + sigma * self.rng.standard_normal()
            z[2, t] = z[2, t-1] + dt * (-meso_decay * z[2, t-1]) + sigma * self.rng.standard_normal()
            
            if is_positive and bif_type == BifurcationType.MEAN_SHIFT:
                z[0, t] += 0.5 * ramp[t]

        # Add physical expressions of states
        z += distractor
        if is_positive and bif_type == BifurcationType.HOPF:
            z[0, :] += r * np.sin(t_axis * 0.5)

        return z, bif_center

    def generate_observations(self, z: np.ndarray) -> np.ndarray:
        """Projects latent states to physical parameters enforcing randomized covariance."""
        T = z.shape[1]
        
        # ──────────────────────────────────────────────────────────────────────
        # Domain Randomization of Observation Matrix & Noise Channels
        # ──────────────────────────────────────────────────────────────────────
        # Randomizes sensor noise scales and mix coefficients while keeping core signs consistent
        obs_noise_scale = self.rng.uniform(0.01, 0.03)  # Sensor baseline variance (Default was 0.02)
        noise = obs_noise_scale * self.rng.standard_normal((4, T))
        
        # Randomize projection mixing weights within a +/- 20% plausible range
        w_temp_z0 = self.rng.uniform(0.8, 1.2)    # Default: 1.0
        w_temp_z1 = self.rng.uniform(0.3, 0.5)    # Default: 0.4
        
        w_salt_z0 = self.rng.uniform(-0.7, -0.5)  # Default: -0.6
        w_salt_z1 = self.rng.uniform(0.2, 0.4)    # Default: 0.3
        
        w_ssh_z0  = self.rng.uniform(0.7, 1.1)    # Default: 0.9
        w_ssh_z2  = self.rng.uniform(0.6, 1.0)    # Default: 0.8
        
        w_curr_z0 = self.rng.uniform(0.3, 0.5)    # Default: 0.4
        w_curr_z2 = self.rng.uniform(1.0, 1.4)    # Default: 1.2
        # ──────────────────────────────────────────────────────────────────────
        
        temp   = w_temp_z0 * z[0] + w_temp_z1 * z[1] + noise[0]
        salt   = w_salt_z0 * z[0] + w_salt_z1 * z[1] + noise[1]
        ssh    = w_ssh_z0 * z[0]  + w_ssh_z2 * z[2]  + noise[2]
        u_curr = w_curr_z0 * z[0] + w_curr_z2 * z[2] + noise[3]
        v_curr = w_curr_z0 * z[0] + w_curr_z2 * z[2] + noise[3]
        
        return self.engineer_ocean_features(temp, salt, ssh, u_curr,v_curr)

    def engineer_ocean_features(self,temp: np.ndarray, salt: np.ndarray, 
                                ssh: np.ndarray, u: np.ndarray, v: np.ndarray):
        """
        Transforms raw telemetry into physically informative features.
        Returns array of shape (5, T).
        """
        # 1. Potential Density (approximate Linearized Equation of State)
        # rho = rho0 * (1 - alpha*(T-T0) + beta*(S-S0))
        rho0 = 1025.0
        alpha = 2.5e-4  # Thermal expansion coeff
        beta = 7.5e-4   # Haline contraction coeff
        # Center around mean T/S for better scaling
        rho = rho0 * (1 - alpha * (temp - np.mean(temp)) + beta * (salt - np.mean(salt)))
        
        # 2. Kinetic Energy (KE = 0.5 * (u^2 + v^2))
        # Higher values indicate more energetic/unstable flow
        ke = 0.5 * (u**2 + v**2)
        
        # 3. SSH Anomaly (Deviation from long-term mean)
        # Bifurcations often show up as persistent anomalies rather than absolute values
        ssh_anom = ssh - np.mean(ssh)
        
        # 4. Velocity Magnitude (Speed)
        speed = np.sqrt(u**2 + v**2)
        
        # 5. Instability Proxy (Standard Deviation of SSH)
        # Calculating a rolling window variability as a proxy for potential regime shifts
        # (Using a simple 12-month rolling window)
        window = 12
        ssh_variability = pd.Series(ssh).rolling(window=12, center=True).std().bfill().ffill().values
        
        # Stack vertically -> shape (5, T)
        features = np.vstack((rho, ke, ssh_anom, speed, ssh_variability))
        
        return features

    def crop_window(self, data: np.ndarray, is_positive: bool, bif_center: int):
        """Randomly samples 64, 128, 256, or 512 windows to prevent positional bias."""
        T_full = data.shape[1]
        w_size = self.rng.choice(self.cfg.window_sizes)
        
        if is_positive:
            # Ensure the window captures the bifurcation event
            start_bound = max(0, bif_center - w_size + 20)
            end_bound = min(T_full - w_size, bif_center - 20)
            start_t = self.rng.integers(start_bound, max(start_bound + 1, end_bound))
            relative_bif_t = bif_center - start_t
        else:
            start_t = self.rng.integers(0, T_full - w_size)
            relative_bif_t = None
            
        return data[:, start_t : start_t + w_size], relative_bif_t

    def build_recording(self, rec_id: str, is_positive: bool) -> Recording:
        bif_type = self.rng.choice(list(BifurcationType)) if is_positive else None
        null_type = self.rng.choice(list(HardNegativeType)) if not is_positive else None
        
        # 1. Generate full baseline (T=2048)
        z, absolute_bif_center = self.generate_latent_system(
            self.cfg.base_time_len, is_positive, bif_type, null_type
        )
        
        # 2. Project to physical observations
        data_full = self.generate_observations(z)
        
        # 3. Crop window
        data_window, event_t = self.crop_window(data_full, is_positive, absolute_bif_center)
        
        return Recording(data=data_window, bifurcation_t=event_t, recording_id=rec_id)
        
# ──────────────────────────────────────────────────────────────────────────────
# Custom Channel-Wise Scaler Class for Time Series Arrays shaped (P, T)
# ──────────────────────────────────────────────────────────────────────────────
class ChannelWiseScaler:
    def __init__(self):
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None

    def fit(self, dataset: list[Recording]):
        """
        Extracts all windows, flattens along the time dimension, 
        and calculates global channel-wise mean and standard deviation.
        """
        # Concatenate all lists of shapes (5, W_i) along the time axis (axis 1)
        # Yields a massive matrix of shape (5, Total_Combined_Timesteps)
        all_data = np.concatenate([rec.data for rec in dataset], axis=1)
        
        # Calculate mean and std independently for each of the 5 channels
        self.means = np.mean(all_data, axis=1, keepdims=True) # Shape: (5, 1)
        self.stds = np.std(all_data, axis=1, keepdims=True)   # Shape: (5, 1)
        
        # Prevent division-by-zero errors on completely flat synthetic channels
        self.stds[self.stds == 0.0] = 1.0
        print("Scaler successfully fitted across training pool data.")

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Applies transformation to matrix shaped (5, T)"""
        if self.means is None or self.stds is None:
            raise RuntimeError("Scaler must be fitted or loaded before transforming data.")
        return (data - self.means) / self.stds

    def save(self, file_path: str):
        """Saves weights cleanly to disk."""
        with open(file_path, 'wb') as f:
            pickle.dump({'means': self.means, 'stds': self.stds}, f)
        print(f"Scaler statistics exported to '{file_path}'")

    @classmethod
    def load(cls, file_path: str):
        """Loads weights into memory for inference scripts."""
        scaler = cls()
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            scaler.means = data['means']
            scaler.stds = data['stds']
        return scaler
        
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualTCNBlock(nn.Module):
    """
    A standard Temporal Convolutional Network block with a residual connection.
    Uses dilated convolutions to expand the receptive field temporally.
    """
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        # Kernel size 3 with appropriate padding to maintain sequence length
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, 
                               padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, 
                               padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + res)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Create a matrix of [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate div_term for the sine/cosine frequency
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Fill the encoding matrix with sine and cosine values
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer (part of model state but not a trainable parameter)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        # Add the positional encoding to the input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return x

class BifurcationNet(nn.Module):
    def __init__(self, num_params: int = 5, d_model: int = 64,dropout:float = 0.2):
        super().__init__()
        
        # 1. Projection: (B, P, T) -> (B, 64, T)
        self.proj = nn.Conv1d(in_channels=num_params, 
                              out_channels=d_model, 
                              kernel_size=5, 
                              padding=2)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        # 2. Three residual TCN blocks (increasing dilation)
        self.tcn = nn.Sequential(
            ResidualTCNBlock(d_model, dilation=1),
            nn.Dropout(dropout),
            ResidualTCNBlock(d_model, dilation=2),
            nn.Dropout(dropout),
            ResidualTCNBlock(d_model, dilation=4),
        )
        
        # 3. 2-layer Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True,
            dropout=dropout  # TransformerEncoderLayer has its own dropout param
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 4. Attention Pooling sequence scores
        self.attn_weights = nn.Linear(d_model, 1)
        
        # 5. Dual Heads (Detection and Timing)
        self.detection_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, padding_mask:torch.Tensor = None):
        """
        Input x shape: (Batch, Params, Time)
        """
        # --- Local Temporal Feature Extraction ---
        x = self.proj(x)       # Shape: (B, 64, T)
        x = self.tcn(x)        # Shape: (B, 64, T)
        
        # --- Global Sequence Modeling ---
        # Transformer expects (Batch, Time, Channels) if batch_first=True
        x = x.transpose(1, 2)  # Shape: (B, T, 64)
        x = self.pos_encoder(x) # Inject temporal context
        x = self.transformer(x,src_key_padding_mask=padding_mask) # Shape: (B, T, 64)
        
        # Instead of attention pooling, use both the pooled context AND
        # the maximum-activated token (captures the transition peak)
        weights = F.softmax(self.attn_weights(x), dim=1)
        pooled_x = torch.sum(x * weights, dim=1)        
        # Timing: weighted average position of attention mass
        T = x.shape[1]
        positions = torch.linspace(0, 1, T, device=x.device)  # (T,)
        t_bifurcation = torch.sum(weights.squeeze(-1) * positions, dim=1)  # (B,)
        
        p_bifurcation = self.detection_head(pooled_x).squeeze(-1)
        # Remove the timing_head Linear layer — no longer needed
        
        return p_bifurcation, t_bifurcation
