import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch.nn.functional as F
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional,Tuple
import pickle
import pandas as pd
import math
from scipy.stats import linregress
import json
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
    NEAR_MISS = auto() #CSD tthat approaches but recovers

@dataclass
class DatasetConfig:
    n_recordings: int = 800
    num_params: int = 5
    base_time_len: int = 2048
    window_sizes: list[int] = field(default_factory=lambda: [64,128,192,256])
    dt: float = 0.1
    seed: int = 42
    sde_constants: dict = None

import pandas as pd
import numpy as np
from scipy.signal import detrend

def causal_smooth(raw_data, smooth_window=6):
    """
    Applies pure causal (trailing) smoothing to preserve variance shifts
    and mean step-changes while dampening high-frequency observational noise.
    
    Parameters:
    - raw_data: numpy array of shape (num_channels, total_timesteps)
    - smooth_window: Number of months for the trailing moving average
    """
    smoothed = np.empty_like(raw_data)
    for i in range(raw_data.shape[0]):
        smoothed[i] = (
            pd.Series(raw_data[i])
            .rolling(window=smooth_window, min_periods=1)
            .mean()
            .values
        )
    return smoothed
# ==============================================================================
# 1. CORE MATH SDE EXTRACTOR FUNCTIONS
# ==============================================================================
def _calculate_efolding_time(signal: np.ndarray, dt: float = 1.0) -> float:
    """
    Computes the fractional e-folding time of a signal using linear interpolation
    to prevent integer lag pinning at fast decorrelation scales.
    """
    n = len(signal)
    if n < 2:
        return dt
    
    # Center the signal
    signal_centered = signal - np.mean(signal)
    var = np.var(signal_centered)
    if var == 0:
        return dt
        
    # Target threshold (1/e)
    target = 1.0 / np.e
    
    # Calculate autocorrelation for the first few lags
    # (We only need to find where it drops below 1/e)
    max_lags = min(100, n - 1)
    acf = []
    
    for lag in range(max_lags):
        if lag == 0:
            acf.append(1.0)
        else:
            covariance = np.mean(signal_centered[:-lag] * signal_centered[lag:])
            acf.append(covariance / var)
            
        # Check if we crossed the 1/e threshold
        if acf[-1] < target:
            idx_after = lag
            idx_before = lag - 1
            val_after = acf[-1]
            val_before = acf[-2]
            
            # ──────────────────────────────────────────────────────────────────
            # LINEAR INTERPOLATION: Find the exact fractional crossing point
            # ──────────────────────────────────────────────────────────────────
            fraction = (val_before - target) / (val_before - val_after)
            fractional_lag = idx_before + fraction
            
            return max(fractional_lag * dt, 1e-3) # Prevent divide-by-zero downstream
            
    # Fallback if the signal decorrelates incredibly slowly
    return max_lags * dt

def calculate_sde_parameters(df: pd.DataFrame) -> dict:
    """
    Extracts high-fidelity physical constants for the coupled SDEs from dataframe columns.
    """
    # Create copy to prevent mutating raw data
    data = df.copy()
    data['month'] = pd.to_datetime(data['time']).dt.month
    
    # --- A. Meso Decay via EKE Anomalies ---
    # 1. Extract seasonal anomalies for velocity vectors
    u_climatology = data.groupby('month')['u_velocity'].transform('mean')
    v_climatology = data.groupby('month')['v_velocity'].transform('mean')
    u_anom = data['u_velocity'] - u_climatology
    v_anom = data['v_velocity'] - v_climatology
    
    # 2. Compute Eddy Kinetic Energy (EKE) profile
    eke = 0.5 * (u_anom**2 + v_anom**2).values
    tau_meso = _calculate_efolding_time(eke, dt=1.0)
    meso_decay = 1.0 / tau_meso
    
    # --- B. Stratification Decay via MLD Anomalies ---
    mld_climatology = data.groupby('month')['mld'].transform('mean')
    mld_anom = (data['mld'] - mld_climatology).values
    tau_strat = _calculate_efolding_time(mld_anom, dt=1.0)
    strat_decay = 1.0 / tau_strat
    
    # --- C. High-Frequency Noise Intensity (Sigma) via SST ---
    temp_climatology = data.groupby('month')['temperature'].transform('mean')
    temp_anom = detrend(data['temperature'] - temp_climatology)
    sigma = float(np.std(temp_anom))
    
    return {
        'sigma': sigma,
        'strat_decay': strat_decay,
        'meso_decay': meso_decay
    }

# ==============================================================================
# 2. OBSERVATION GENERATOR MAPPING (BASELINES & SCALES)
# ==============================================================================

def calculate_observation_mappings(df: pd.DataFrame, baseline_years: int = 10) -> tuple[dict, dict]:
    """
    Calculates regional baselines (mean of first 10 years) and 
    swing values (std of remaining array) for the observation space.
    """
    baseline_months = baseline_years * 12
    
    # Map CSV columns to observation key names expected by your generator
    mapping_keys = {
        'temp': 'temperature',
        'salt': 'salinity',
        'ssh': 'ssh',
        'u_curr': 'u_velocity',
        'v_curr': 'v_velocity'
    }
    
    baselines = {}
    scales = {}
    
    for gen_key, csv_col in mapping_keys.items():
        array = df[csv_col].values
        
        # 1. Baseline value = mean of first N years
        baselines[gen_key] = float(np.mean(array[:baseline_months]))
        
        # 2. Swing value = standard deviation of the remaining data
        remaining_data = array[baseline_months:]
        scales[gen_key] = float(np.std(remaining_data)) if len(remaining_data) > 0 else float(np.std(array))
        
        if scales[gen_key] == 0.0:
            scales[gen_key] = 1.0
            
    return baselines, scales

def engineer_ocean_features(temp: np.ndarray, salt: np.ndarray, 
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
        # ──────────────────────────────────────────────────────────────────────
        # NEW: Isolate velocity anomalies to neutralize absolute baseline shifts
        # ──────────────────────────────────────────────────────────────────────
        u_series = pd.Series(u)
        u_anom = (u_series - u_series.rolling(window=36, min_periods=1).mean()).values
        v_series = pd.Series(v)
        v_anom = (v_series - v_series.rolling(window=36, min_periods=1).mean()).values
        # ──────────────────────────────────────────────────────────────────────
        # Center around mean T/S for better scaling
        rho = rho0 * (1 - alpha * (temp - np.mean(temp)) + beta * (salt - np.mean(salt)))
        
        # 2. Kinetic Energy (KE = 0.5 * (u^2 + v^2))
        # Higher values indicate more energetic/unstable flow
        ke = 0.5 * (u_anom**2 + v_anom**2)
        
        # 3. SSH Anomaly (Deviation from long-term mean)
        # Bifurcations often show up as persistent anomalies rather than absolute values
        ssh_anom = ssh - np.mean(ssh)
        
        # 4. Velocity Magnitude (Speed)
        speed = np.sqrt(u_anom**2 + v_anom**2)
        
        # 5. Instability Proxy (Standard Deviation of SSH)
        # Calculating a rolling window variability as a proxy for potential regime shifts
        # (Using a simple 12-month rolling window)
        window = 12
        ssh_variability = pd.Series(ssh).rolling(window=window, min_periods=3).std().fillna(0).values
        
        # Stack vertically -> shape (5, T)
        features = np.vstack((rho, ke, ssh_anom, speed, ssh_variability))
        
        return features

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
        # OPTION A (UPDATED): Data-Anchored Domain Randomization
        # ──────────────────────────────────────────────────────────────────────
        # Pull the real empirical means extracted from your ocean model file
        empirical = self.cfg.sde_constants  # e.g., {'sigma': 0.042, 'strat_decay': 0.055, ...}
        
        # Sample using a Normal distribution centered on real physics
        # We use a small coefficient (e.g., 10% of the mean) as the standard deviation
        sigma = self.rng.normal(loc=empirical['sigma'], scale=empirical['sigma'] * 0.1)
        strat_decay = self.rng.normal(loc=empirical['strat_decay'], scale=empirical['strat_decay'] * 0.1)
        meso_decay = self.rng.normal(loc=empirical['meso_decay'], scale=empirical['meso_decay'] * 0.1)
        
        # Keep these original uniform balances if you don't have CSV equivalents yet
        theta0 = self.rng.uniform(0.07, 0.13)
        coupling_strength = self.rng.uniform(0.12, 0.18)
        
        # Physical Guardrails: Ensure random sampling never hits negative or zero values
        sigma = max(0.005, sigma)
        strat_decay = max(0.01, strat_decay)
        meso_decay = max(0.01, meso_decay)
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
        # --- Pre-compute near-miss ramp BEFORE the SDE loop ---
        near_miss_ramp = np.zeros(T)
        if not is_positive and null_type == HardNegativeType.NEAR_MISS:
            approach_c = int(T * self.rng.uniform(0.25, 0.45))
            recovery_c = int(T * self.rng.uniform(0.55, 0.75))
            near_miss_ramp = (
                1 / (1 + np.exp(-(t_axis - approach_c) / 20.0)) -
                1 / (1 + np.exp(-(t_axis - recovery_c) / 20.0))
            )
            near_miss_ramp = np.clip(near_miss_ramp, 0, 1)

        # Evolve coupled SDEs
        for t in range(1, T):
            # Dynamic restoring force for Critical Slowing Down (Variance)
            theta_t = theta0
            if is_positive and bif_type == BifurcationType.VARIANCE:
                theta_t = theta0 * (1 - ramp[t])
                theta_t = max(theta_t, 1e-3) # Cap to avoid strict division by zero
            elif not is_positive and null_type == HardNegativeType.NEAR_MISS:
                # Decays to 30% of theta0 maximum — never actually crosses
                decay_factor = near_miss_ramp[t] * 0.70
                theta_t = max(theta0 * (1 - decay_factor), theta0 * 0.30)
                
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

    def generate_observations(self, z: np.ndarray,baselines,scales) -> np.ndarray:
        """Projects latent states to physical parameters enforcing randomized covariance
           and maps them to realistic oceanographic baselines."""
        T = z.shape[1]
        
        # ──────────────────────────────────────────────────────────────────────
        # 1. Domain Randomization of Observation Matrix & Noise Channels
        # ──────────────────────────────────────────────────────────────────────
        obs_noise_scale = self.rng.uniform(0.01, 0.03)  
        noise = obs_noise_scale * self.rng.standard_normal((5, T))
        
        # Randomize projection mixing weights
        w_temp_z0 = self.rng.uniform(0.8, 1.2)    
        w_temp_z1 = self.rng.uniform(0.3, 0.5)    
        
        w_salt_z0 = self.rng.uniform(-0.7, -0.5)  
        w_salt_z1 = self.rng.uniform(0.2, 0.4)    
        
        w_ssh_z0  = self.rng.uniform(0.7, 1.1)    
        w_ssh_z2  = self.rng.uniform(0.6, 1.0)    
        
        w_curr_z0 = self.rng.uniform(0.3, 0.5)    
        w_curr_z2 = self.rng.uniform(1.0, 1.4)    

        # ──────────────────────────────────────────────────────────────────────
        # 2. Extract Raw Latent Anomalies
        # ──────────────────────────────────────────────────────────────────────
        raw_temp_anom  = w_temp_z0 * z[0] + w_temp_z1 * z[1] + noise[0]
        raw_salt_anom  = w_salt_z0 * z[0] + w_salt_z1 * z[1] + noise[1]
        raw_ssh_anom   = w_ssh_z0 * z[0]  + w_ssh_z2 * z[2]  + noise[2]
        raw_ucurr_anom = w_curr_z0 * z[0] + w_curr_z2 * z[2] + noise[3]
        raw_vcurr_anom = w_curr_z0 * z[0] + w_curr_z2 * z[2] + noise[4]

        # ──────────────────────────────────────────────────────────────────────
        # NEW: Simulate Secular Climate Forcing (SSP370 Background Ramp)
        # ──────────────────────────────────────────────────────────────────────
        # We create a long-term linear ramp across the timeline T
        secular_ramp = np.linspace(0, 1, T)

        # Randomize the severity of the background climate trend per generation
        temp_warming_trend = self.rng.uniform(0.5, 2.5)  # Up to 2.5 degrees warming
        ssh_rise_trend     = self.rng.uniform(0.05, 0.2) # Up to 0.2m sea level rise
        
        temp_forced = raw_temp_anom + (secular_ramp * temp_warming_trend / scales['temp'])
        ssh_forced  = raw_ssh_anom  + (secular_ramp * ssh_rise_trend / scales['ssh'])
        # ──────────────────────────────────────────────────────────────────────
    
        # Apply transformations with climate trends injected
        temp   = baselines['temp']   + (temp_forced * scales['temp'])
        salt   = baselines['salt']   + (raw_salt_anom * scales['salt'])
        ssh    = baselines['ssh']    + (ssh_forced * scales['ssh'])
        u_curr = baselines['u_curr'] + (raw_ucurr_anom * scales['u_curr'])
        v_curr = baselines['v_curr'] + (raw_vcurr_anom * scales['v_curr'])

        return engineer_ocean_features(temp, salt, ssh, u_curr,v_curr)


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

    def build_recording(self, rec_id: str, is_positive: bool,baselines,scales) -> Recording:
        bif_type = self.rng.choice(list(BifurcationType)) if is_positive else None
        null_weights = [0.15, 0.15, 0.15, 0.15, 0.40]  # NEAR_MISS gets 40%
        null_type = self.rng.choice(list(HardNegativeType), p=null_weights) if not is_positive else None
        
        # 1. Generate full baseline (T=2048)
        z, absolute_bif_center = self.generate_latent_system(
            self.cfg.base_time_len, is_positive, bif_type, null_type
        )
        
        # 2. Project to physical observations
        data_full = self.generate_observations(z,baselines,scales)

        # ──────────────────────────────────────────────────────────────────────
        # NEW: Apply Causal Detrending & Smoothing to Full Synthetic Stream
        # ──────────────────────────────────────────────────────────────────────
        # Transpose to (2048, 5) for the processing function, then back to (5, 2048)
        processed_data_full = causal_smooth(
            data_full, 
        )
        # ──────────────────────────────────────────────────────────────────────
        
        # 3. Crop window
        data_window, event_t = self.crop_window(processed_data_full, is_positive, absolute_bif_center)
        
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

class ResidualTCNBlock(nn.Module):
    """
    A strictly Causal Temporal Convolutional Network block.
    Uses asymmetric left-padding to ensure that features at timestep t
    never depend on information from timesteps > t.
    """
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        
        # For kernel_size=3, total causal padding needed on the left is 2 * dilation
        self.left_pad = nn.ConstantPad1d((2 * dilation, 0), 0.0)
        
        # Set padding=0 because we handle it manually via self.left_pad
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, 
                               padding=0, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, 
                               padding=0, dilation=dilation)
        
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        
        # First causal convolution layer
        x = self.left_pad(x)
        x = self.relu(self.conv1(x))
        
        # Second causal convolution layer
        x = self.left_pad(x)
        x = self.conv2(x)
        
        # Residual match (shapes will match perfectly because left-padding 
        # exactly compensates for the causal drop)
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
        
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        return ((1 - p_t) ** self.gamma * bce).mean()
        
class Chomp1d(nn.Module):
    """Trims the future timesteps introduced by asymmetric padding to ensure causality."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class BifurcationNet(nn.Module):
    def __init__(self, num_params: int = 5, d_model: int = 96, dropout: float = 0.2):
        super().__init__()
        
        # 1. Causal Projection: Pad on the left by (kernel_size - 1)
        self.proj = nn.Sequential(
            nn.ConstantPad1d((4, 0), 0.0), # Left-pad by 4, right-pad by 0
            nn.Conv1d(in_channels=num_params, out_channels=d_model, kernel_size=5, padding=0),
            nn.ReLU()
        )
        
        # 2. TCN Blocks (Assuming your ResidualTCNBlock handles its own internal causality)
        self.tcn_blocks = nn.ModuleList([
            ResidualTCNBlock(d_model, dilation=1),
            nn.Dropout(dropout),
            ResidualTCNBlock(d_model, dilation=2),
            nn.Dropout(dropout),
            ResidualTCNBlock(d_model, dilation=4),
            nn.Dropout(dropout),
            ResidualTCNBlock(d_model, dilation=8),   # new
        ])
        
        # 3. Normalization bridge before the Transformer
        self.norm_bridge = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 4. 2-layer Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True,
            dropout=dropout, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 5. Attention Pooling & Dual Heads
        self.attn_weights = nn.Linear(d_model, 1)
        # FIX 1: pre-head dropout gate — this is what MC sampling actually draws from.
        # Without this, the attention-pooled context is nearly identical across all
        # n_samples because the upstream dropout noise is averaged out over T timesteps.
        self.pre_head_dropout = nn.Dropout(dropout)

        self.detection_head = nn.Linear(d_model, 1)

        # FIX 2: give the timing head its own dedicated MLP so it has a gradient
        # path that doesn't fully compete with the detection head.
        # A single shared linear was collapsing to the dataset mean.
        self.timing_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),          # additional stochasticity for MC sampling
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()                  # constrain to [0, 1] if timing is normalised
        )


    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None):
        # x shape: (B, num_params, T)
        x = self.proj(x)
        
        for block in self.tcn_blocks:
            x = block(x)
            
        # Change shape from Conv1d (B, d_model, T) to Transformer (B, T, d_model)
        x = x.permute(0, 2, 1)
        
        # Stabilize and apply positional tags right before self-attention
        x = self.norm_bridge(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # 4. Transformer forward pass with the padding mask
        # (Tells self-attention layers to completely ignore padded ocean timesteps)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        
        # 5. Attention Pooling over the time dimension
        attn_scores = self.attn_weights(x)  # Shape: (B, T, 1)
        
        if pad_mask is not None:
            # Force the attention weights of padded tokens to -inf 
            # so their Softmax contribution drops to absolute zero
            attn_scores = attn_scores.masked_fill(pad_mask.unsqueeze(-1), float('-inf'))
            
        weights = F.softmax(attn_scores, dim=1)  # Shape: (B, T, 1)
        context = torch.sum(weights * x, dim=1)  # Shape: (B, d_model)
        
        context = self.pre_head_dropout(context)
        # 6. Extract predictions from your dual heads
        p_bif_pred = self.detection_head(context).squeeze(-1)  # 🔄 (B, 1) -> (B,)
        t_bif_pred = self.timing_head(context).squeeze(-1)     # 🔄 (B, 1) -> (B,)
        
        return p_bif_pred, t_bif_pred

    def mc_forward(self, x, pad_mask=None, n_samples=30):
        self.train()
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
        with torch.no_grad():
            samples = [self.forward(x, pad_mask) for _ in range(n_samples)]
            logit_samples  = torch.stack([s[0] for s in samples])  # (n_samples, B)
            timing_samples = torch.stack([s[1] for s in samples])  # (n_samples, B)
        self.eval()
        probs = torch.sigmoid(logit_samples)
        return (
            probs.mean(dim=0),          # mean P(bifurcation)   (B,)
            probs.std(dim=0),           # epistemic uncertainty (B,)
            timing_samples.mean(dim=0), # mean timing pred      (B,)
            timing_samples.std(dim=0),  # timing uncertainty    (B,)
        )

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # init slightly > 1

    def fit(self, model, val_loader, device) -> "TemperatureScaler":
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y_cls, _, pad_mask in val_loader:
                x, pad_mask = x.to(device), pad_mask.to(device)
                logits, _ = model(x, pad_mask)
                all_logits.append(logits.cpu())
                all_labels.append(y_cls)
        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)

        nll = nn.BCEWithLogitsLoss()
        opt = optim.LBFGS([self.temperature], lr=0.01, max_iter=200)

        def step():
            opt.zero_grad()
            loss = nll(logits_cat / self.temperature, labels_cat)
            loss.backward()
            return loss

        opt.step(step)
        print(f"Calibrated T = {self.temperature.item():.4f}")
        return self

    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.to(logits.device)

    
# ──────────────────────────────────────────────────────────────────────────────
# 1. PyTorch Dataset Wrapper
# ──────────────────────────────────────────────────────────────────────────────
class OceanBifurcationDataset(Dataset):
    def __init__(self, recordings):
        self.recordings = recordings

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        rec = self.recordings[idx]
        
        # Input tensor: (P, T)
        x = torch.tensor(rec.data, dtype=torch.float32)
        
        # Fix: Infer positive class if bifurcation_t is an integer index (not None)
        is_positive = rec.bifurcation_t is not None
        
        # Classification target: 1.0 if positive, 0.0 if negative
        y_cls = torch.tensor(1.0 if is_positive else 0.0, dtype=torch.float32)
        
        # Timing target: normalized to [0, 1] relative to the window size
        # If no bifurcation, we default to 0.0 (it gets masked out in compute_loss anyway)
        time_len = rec.data.shape[1]
        if is_positive:
            y_time = torch.tensor(rec.bifurcation_t / time_len, dtype=torch.float32)
        else:
            y_time = torch.tensor(0.0, dtype=torch.float32)
            
        return x, y_cls, y_time

# ──────────────────────────────────────────────────────────────────────────────
# 2. Custom Loss Function
# ──────────────────────────────────────────────────────────────────────────────
def compute_loss(p_bif_pred, t_bif_pred, y_cls, y_time, criterion_cls,
                 time_weight: float = 0.5):
    loss_cls = criterion_cls(p_bif_pred, y_cls)
    
    mask = (y_cls > 0.5)  # robust float comparison
    if mask.sum() > 0:
        loss_time = F.smooth_l1_loss(
            t_bif_pred[mask], y_time[mask], beta=0.1
        )
    else:
        # keep it in the computation graph so .item() calls don't break
        loss_time = p_bif_pred.sum() * 0.0

    total = loss_cls + time_weight * loss_time
    return total, loss_cls, loss_time

# ──────────────────────────────────────────────────────────────────────────────
# 3. Training and Evaluation Loops
# ──────────────────────────────────────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, device,criterion_cls):
    model.train()
    total_loss, total_bce, total_time = 0, 0, 0
    correct = 0
    
    for x, y_cls, y_time, pad_mask in dataloader:
        x, y_cls, y_time, pad_mask = x.to(device), y_cls.to(device), y_time.to(device),pad_mask.to(device)
        
        optimizer.zero_grad()
        p_bif_pred, t_bif_pred = model(x, pad_mask)
        
        loss, bce, time_loss = compute_loss(p_bif_pred, t_bif_pred, y_cls, y_time,criterion_cls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()       
        total_loss += loss.item()
        total_bce += bce.item()
        total_time += time_loss.item()
        
        # Calculate accuracy (threshold at 0.5)
        preds = (p_bif_pred >= 0.0).float()  # logit > 0 means p > 0.5
        correct += (preds == y_cls).sum().item()
        
    N = len(dataloader)
    acc = correct / len(dataloader.dataset)
    return total_loss/N, total_bce/N, total_time/N, acc

@torch.no_grad()
def evaluate(model, dataloader, device,criterion_cls):
    model.eval()
    total_loss, total_bce, total_time = 0, 0, 0
    
    # Lists to store aggregated results
    all_preds = []
    all_targets = []
    time_errors = []
    
    for x, y_cls, y_time, pad_mask in dataloader:
        x, y_cls, y_time, pad_mask = x.to(device), y_cls.to(device), y_time.to(device), pad_mask.to(device)
        p_bif_pred, t_bif_pred = model(x, pad_mask)
        
        # Calculate losses
        loss, bce, time_loss = compute_loss(p_bif_pred, t_bif_pred, y_cls, y_time,criterion_cls)
        
        total_loss += loss.item()
        total_bce += bce.item()
        total_time += time_loss.item()
        
        # Convert Logits -> Binary Predictions (Threshold 0.0 for BCEWithLogits)
        preds = (p_bif_pred >= 0.0).float()
        
        # Aggregate for metrics (move to CPU)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_cls.cpu().numpy())
        
        # Track timing errors
        mask = (y_cls == 1.0)
        if mask.sum() > 0:
            err = torch.abs(t_bif_pred[mask] - y_time[mask])
            time_errors.extend(err.cpu().tolist())
            
    # Compute Metrics using sklearn
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    N = len(dataloader)
    mean_time_err = np.mean(time_errors) if time_errors else 0.0
    
    # Return additional metrics
    return total_loss/N, total_bce/N, total_time/N, precision, recall, f1, mean_time_err


def variable_length_collate_fn(batch):
    """
    Dynamically pads sequences to the maximum length present within the current batch
    to allow stacking variable-length window signals.
    """
    # batch is a list of tuples returned from OceanBifurcationDataset: (x, y_cls, y_time)
    xs, y_clss, y_times = zip(*batch)
    batch_size = len(xs)
    # Find the maximum time length (T) present in this specific batch
    max_len = max(x.shape[1] for x in xs)

    # In collate_fn, also return a mask:
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    # Mark padded positions as True (ignored)
    for i, x in enumerate(xs):
        if x.shape[1] < max_len:
            padding_mask[i, x.shape[1]:] = True
    
    padded_xs = []
    for x in xs:
        pad_len = max_len - x.shape[1]
        if pad_len > 0:
            # x has shape (P, T). F.pad pads the last dimension (T) when given a 2-tuple (left, right)
            x_padded = F.pad(x, (0, pad_len), value=0.0)
        else:
            x_padded = x
        padded_xs.append(x_padded)
        
    # Stack individual samples into clean batch tensors
    x_batch = torch.stack(padded_xs, dim=0)
    y_cls_batch = torch.stack(y_clss, dim=0)
    y_time_batch = torch.stack(y_times, dim=0)
    
    return x_batch, y_cls_batch, y_time_batch, padding_mask

import matplotlib.pyplot as plt

def plot_training_curves(history,plot_path:str):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Find the epoch with the lowest validation loss
    best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Losses
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='orange', linestyle='--')
    ax1.axvline(best_epoch, color='red', linestyle=':', alpha=0.6, label=f'Best Epoch: {best_epoch}')
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Metrics
    ax2.plot(epochs, history['f1'], label='F1 Score', color='green')
    ax2.plot(epochs, history['prec'], label='Precision', color='purple', linestyle='-.')
    ax2.plot(epochs, history['rec'], label='Recall', color='red', linestyle='-.')
    ax2.axvline(best_epoch, color='red', linestyle=':', alpha=0.6)
    ax2.set_title('F1, Precision, and Recall')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, classification_report

def evaluate_and_plot_curves(model, data_loader, device, scaler:TemperatureScaler,plot_path:str):
    """
    Extracts true labels and model probabilities, plots ROC and PR curves,
    and calculates optimal decision thresholds.
    """
    model.eval()
    all_probs = []
    all_targets = []
    
    # 1. Extract raw predictions (probabilities) and ground truth
    with torch.no_grad():
        # FIX 1: Explicitly unpack the 4 elements returned by variable_length_collate_fn
        for batch_x, batch_y, _, pad_mask in data_loader:
            batch_x = batch_x.to(device)
            pad_mask = pad_mask.to(device) # FIX 2: Send mask to device
            
            # FIX 3: Unpack the tuple (p_bifurcation, t_bifurcation) returned by your model
            logits, _ = model(batch_x, pad_mask)
            logits = scaler.scale(logits) 
            probs = torch.sigmoid(logits) # Convert logits to probabilities [0, 1]
            
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # 2. Compute Metrics
    fpr, tpr, roc_thresholds = roc_curve(all_targets, all_probs)
    roc_auc = roc_auc_score(all_targets, all_probs)
    
    precision, recall, pr_thresholds = precision_recall_curve(all_targets, all_probs)
    
    # 3. Find Optimal Thresholds
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    best_threshold_j = roc_thresholds[best_j_idx]
    
    f1_scores = [2 * (p * r) / (p + r + 1e-8) for p, r in zip(precision[:-1], recall[:-1])]
    best_f1_idx = np.argmax(f1_scores)
    best_threshold_f1 = pr_thresholds[best_f1_idx]
    max_f1 = f1_scores[best_f1_idx]

    # Find threshold where FPR <= target_fpr (e.g. 5%)
    target_fpr = 0.05
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) > 0:
        best_precision_idx = valid_idx[np.argmax(tpr[valid_idx])]
        best_threshold_precision = roc_thresholds[best_precision_idx]
        print(f"Recommended Threshold (FPR ≤ {target_fpr*100:.0f}%): "
              f"{best_threshold_precision:.4f} "
              f"→ TPR: {tpr[best_precision_idx]:.3f}, "
              f"FPR: {fpr[best_precision_idx]:.3f}")

    # ---------------------------------------------------------
    # PLOTTING CODES
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 6))
    
    # Plot 1: ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[best_j_idx], tpr[best_j_idx], color='red', marker='X', s=150, 
                label=f"Youden's Optimal Thresh: {best_threshold_j:.3f}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall Curve')
    plt.scatter(recall[best_f1_idx], precision[best_f1_idx], color='purple', marker='o', s=150,
                label=f"Max F1 Thresh: {best_threshold_f1:.3f} (F1 = {max_f1:.3f})")
    plt.xlabel('Recall (True Positive Rate)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title('Precision-Recall Curve')
    # FIX 4: Complete the cut-off lines from the end of the script
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    
    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------
    print("=" * 60)
    print("THRESHOLD OPTIMIZATION REPORT")
    print("=" * 60)
    print(f"Area Under ROC Curve (ROC-AUC): {roc_auc:.4f}")
    print(f"Recommended Threshold (Youden's J): {best_threshold_j:.4f}")
    print(f"Recommended Threshold (Maximize F1):  {best_threshold_f1:.4f} -> Max F1: {max_f1:.4f}")
    print("-" * 60)
    
    # Demonstrate the impact of switching thresholds
    print(f"\nClassification Report using default threshold (0.500):")
    preds_50 = (all_probs >= 0.5).astype(int)
    print(classification_report(all_targets, preds_50, digits=3))
    
    print(f"\nClassification Report using optimized F1 threshold ({best_threshold_f1:.3f}):")
    preds_opt = (all_probs >= best_threshold_f1).astype(int)
    print(classification_report(all_targets, preds_opt, digits=3))

    print(f"\nClassification Report using minimised FPR threshold ({best_threshold_precision:.4f}):")
    preds_prec = (all_probs >= best_threshold_precision).astype(int)
    print(classification_report(all_targets, preds_prec, digits=3))
    print("=" * 60)
    
    return best_threshold_f1

def persistence_gate(
    prob_series: np.ndarray,
    threshold: float = 0.5,
    min_sustained: int = 12   # timesteps; ~12 months if monthly resolution
) -> np.ndarray:
    """
    Causal: only fires once probability has exceeded threshold for
    min_sustained consecutive steps. Resets immediately on any drop.
    """
    alerts = np.zeros(len(prob_series), dtype=bool)
    run = 0
    for t, p in enumerate(prob_series):
        run = run + 1 if p >= threshold else 0
        alerts[t] = run >= min_sustained
    return alerts

def uncertain_alert(model, x, pad_mask, device,
                    prob_threshold=0.5, uncertainty_ceiling=0.15):
    x, pad_mask = x.to(device), pad_mask.to(device)
    mean_p, std_p = model.mc_forward(x, pad_mask)
    alert = (mean_p >= prob_threshold) & (std_p <= uncertainty_ceiling)
    return alert, mean_p, std_p
    
def run_inference_on_series(
    model: BifurcationNet,
    ts: TemperatureScaler,
    scaler: ChannelWiseScaler,
    raw_series: np.ndarray,          # shape (5, T_full) — unscaled physical features
    window_size: int = 256,
    step: int = 1,                   # stride between windows (months)
    prob_threshold: float = 0.5,
    uncertainty_ceiling: float = 0.15,
    min_sustained: int = 12,
    device: torch.device = torch.device("cpu"),
    mc_samples: int = 30,
) -> dict:
    """
    Slides a window across a full time series and returns per-timestep
    calibrated probabilities, epistemic uncertainty, and gated alerts.
    """
    model.eval()
    T_full = raw_series.shape[1]
    n_windows = (T_full - window_size) // step + 1

    mean_probs    = np.full(T_full, np.nan)
    uncertainties = np.full(T_full, np.nan)

    for w in range(n_windows):
        start = w * step
        end   = start + window_size
        window = scaler.transform(raw_series[:, start:end])          # (5, W)

        x        = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)  # (1,5,W)
        pad_mask = torch.zeros(1, window_size, dtype=torch.bool).to(device)

        mean_p, std_p,_,_ = model.mc_forward(x, pad_mask, n_samples=mc_samples)

        # Apply temperature calibration
        # Note: mc_forward returns probs directly, so we calibrate the logit equivalent
        # Re-run a single calibrated forward for the point estimate
        with torch.no_grad():
            logit, _ = model(x, pad_mask)
            cal_prob  = torch.sigmoid(ts.scale(logit)).item()
            mc_std    = std_p.item()

        # Assign to the last timestep of the window (causal: decision made at t=end-1)
        t_out = end - 1
        mean_probs[t_out]    = cal_prob
        uncertainties[t_out] = mc_std

    # Forward-fill NaNs at the start (warm-up period)
    valid = ~np.isnan(mean_probs)
    mean_probs[~valid]    = 0.0
    uncertainties[~valid] = 1.0   # high uncertainty during warm-up

    # Gate on uncertainty then persistence
    credible   = uncertainties <= uncertainty_ceiling
    gated_prob = np.where(credible, mean_probs, 0.0)
    alerts     = persistence_gate(gated_prob, threshold=prob_threshold,
                                  min_sustained=min_sustained)

    return {
        "prob":          mean_probs,
        "uncertainty":   uncertainties,
        "credible_mask": credible,
        "alert":         alerts,
    }
# ──────────────────────────────────────────────────────────────────────────────
# 4. Main Execution
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_st = 1
    ensemble_end = 40
    best_thresholds = {}
    null_alerts = {}

    for i in range(ensemble_st-1,ensemble_end,1):
        # 1. Read Regional Data Frame
        input_file = f"ensembles/CANARI_HIST2_{i+1}_LON_-56.506_LAT_60.819.csv"
        try:
            df = pd.read_csv(input_file)
        except FileNotFoundError:
            print(f"input_file not found for ensemble {i+1}, skipping training")
            continue
        # 2. Run Calibrations
        sde_constants = calculate_sde_parameters(df)
        baselines, scales = calculate_observation_mappings(df, baseline_years=10)
        
        # 3. Print Configuration Profiles
        print("=" * 60)
        print("DYNAMIC CALIBRATION REPORT FOR TARGET REGION")
        print("=" * 60)
        print("A. LATENT SDE SYSTEM CONFIG:")
        print(f"   sigma (Noise Intensity)        : {sde_constants['sigma']:.5f}")
        print(f"   strat_decay (Damping factor)   : {sde_constants['strat_decay']:.5f}")
        print(f"   meso_decay (Dissipation rate)  : {sde_constants['meso_decay']:.5f}")
        print("-" * 60)
        print("B. OBSERVATION BASELINES & SWINGS:")
        for var in baselines.keys():
            print(f"   {var:<8} -> Baseline Mean: {baselines[var]:8.3f} | Physical Swing (±1σ): {scales[var]:.4f}")
        print("=" * 60)
    
        # --- Configuration ---
        cfg = DatasetConfig(n_recordings=5000, num_params=5,sde_constants=sde_constants) 
        rng = np.random.default_rng(cfg.seed)
        batch_size = 64
        epochs = 50
        
        # 1. Generate Dataset
        print("Generating synthetic dataset...")
        gen = OceanStateGenerator(cfg, rng)
        recordings = [gen.build_recording(f"rec_{i:04d}", is_positive=(i % 2 == 0),baselines=baselines,scales=scales) 
                      for i in range(cfg.n_recordings)]
    
        # 2. Split Data (using Subsets)
        full_dataset = OceanBifurcationDataset(recordings)
        train_set, val_set, test_set = random_split(
            full_dataset, 
            [0.7, 0.15, 0.15],
            generator=torch.Generator().manual_seed(cfg.seed)
        )
    
        # 3. Fit Scaler ONLY on Training Subset Indices
        print("Fitting scaler on training subset...")
        # Access the specific recordings that belong to the training split
        train_recs = [recordings[i] for i in train_set.indices]
        
        scaler = ChannelWiseScaler()
        scaler.fit(train_recs)
        scaler.save(f"scalers/synthetic_channel_scaler_{i+1}.pkl")
        # # ──────────────────────────────────────────────────────────────────────
        # # DIAGNOSTIC: EXPOSE SCALER TRAINING ORDER
        # # ──────────────────────────────────────────────────────────────────────
        # print("🔮 [SCALER] Internal Means Shape:", scaler.means.shape)
        # print("🔮 [SCALER] Trained Mean Values:\n", scaler.means)
        # # ──────────────────────────────────────────────────────────────
    
        # 4. Transform ALL data using the Train-fitted scaler
        # Note: We do this after splitting to keep the original split indices valid
        for rec in recordings:
            rec.data = scaler.transform(rec.data)
    
        # 5. Initialize Loaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=variable_length_collate_fn)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=variable_length_collate_fn)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=variable_length_collate_fn)
    
        # 6. Initialize Model & Optimizer
        model = BifurcationNet(num_params=cfg.num_params).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,eta_min=1e-6)
        criterion_cls = FocalLoss(gamma=2.0)
    
        # 7. Training Loop
        print("\nStarting Training...")
        history = {
            'train_loss': [],
            'val_time_loss': [],
            'val_loss': [],
            'f1': [],
            'prec': [],
            'rec': [],
            'v_bce': []
        }
        best_val_loss = float('inf')
        # Define your patience here:
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            tr_loss, tr_bce, tr_time, tr_acc = train_epoch(model, train_loader, optimizer, device,criterion_cls)
            scheduler.step()
            
            # Unpack the 7 return values from the new evaluate signature
            val_res = evaluate(model, val_loader, device,criterion_cls)
            v_loss, v_bce, t_loss, v_prec, v_rec, v_f1, _ = val_res
            # Store metrics
            history['train_loss'].append(tr_loss)
            history['val_time_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['f1'].append(v_f1)
            history['prec'].append(v_prec)
            history['rec'].append(v_rec)
            history['v_bce'].append(v_bce)
            print(f"Epoch {epoch+1:02d} | Val Loss: {v_loss:.4f} | Val Time Loss: {t_loss:.4f} | Val Bce: {v_bce:.4f} | F1: {v_f1:.3f} | Prec: {v_prec:.2f} | Rec: {v_rec:.2f}")
            
            # Save best model logic:
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(model.state_dict(), f"models/best_bifurcation_model_{i+1}.pth")
                patience_counter = 0  # reset
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
    
        # 8. Final Test
        print("\nTesting on hold-out set...")
        try:
            model.load_state_dict(torch.load(f"models/best_bifurcation_model_{i+1}.pth",weights_only=True))
        except FileNotFoundError as e: 
            print(f"[Errno 2] No such file or directory: '{e}'")
            print("ml model not found, skipping this hold out test")
            continue

        ts = TemperatureScaler().fit(model, val_loader, device)
        torch.save({'temperature': ts.temperature.item()}, f"scalers/temperature_scaler_{i+1}.pt")
        test_res = evaluate(model, test_loader, device,criterion_cls)
        _, _, _, t_prec, t_rec, t_f1, t_time_err = test_res
        
        print(f"Final Test -> F1: {t_f1:.3f} | Precision: {t_prec:.2f} | Recall: {t_rec:.2f}")
        plot_training_curves(history,plot_path=f"curves/training_curves_{i+1}.png")
    
        best_thresh = evaluate_and_plot_curves(model, test_loader, device,scaler=ts, plot_path=f"curves/roc_curve_{i+1}.png")
        print(f"Best Threshold -> {best_thresh}")
        best_thresholds[f"ensemble_{i+1}"] = float(best_thresh)
    
        # Verify on the historical input — should produce no sustained alerts
        hist_features = engineer_ocean_features(
            df['temperature'].values, df['salinity'].values,
            df['ssh'].values, df['u_velocity'].values, df['v_velocity'].values
        )  # shape (5, T_hist)
        
        inference_result = run_inference_on_series(
            model, ts, scaler, hist_features,
            window_size=128, step=1, device=device
        )
        n_alerts = inference_result["alert"].sum()
        print(f"Historical false-positive alert timesteps: {n_alerts} / {hist_features.shape[1]}")
        null_alerts[f"ensemble_{i+1}"] = int(n_alerts)

        with open('best_thresholds.json', 'w') as fp:
            json.dump(best_thresholds, fp)
    
        with open('null_alerts.json', 'w') as fp:
            json.dump(null_alerts, fp)