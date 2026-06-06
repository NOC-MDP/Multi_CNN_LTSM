import os
import torch
import numpy as np
import pandas as pd
# ─── 1. IMPORT YOUR MODEL ARCHITECTURE ───────────────────────────────────────
from pipeline import BifurcationNet, DatasetConfig, ChannelWiseScaler
# ─────────────────────────────────────────────────────────────────────────────
import ruptures as rpt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from scipy.stats import linregress

def causal_smooth(raw_data, smooth_window=6):
    """
    Applies pure causal (trailing) smoothing to preserve variance shifts
    and mean step-changes while dampening high-frequency observational noise.
    
    Parameters:
    - raw_data: numpy array of shape (num_channels, total_timesteps)
    - smooth_window: Number of months for the trailing moving average
    """
    num_channels, total_timesteps = raw_data.shape
    smoothed_data = np.zeros_like(raw_data)
    
    for i in range(num_channels):
        series = raw_data[i, :]
        smoothed_series = np.zeros_like(series)
        
        for t in range(total_timesteps):
            if t < smooth_window - 1:
                # Not enough history yet: average available window
                smoothed_series[t] = np.mean(series[:t+1])
            else:
                # Trailing average: looks only at current and past data
                smoothed_series[t] = np.mean(series[t - smooth_window + 1 : t + 1])
                
        smoothed_data[i, :] = smoothed_series
        
    return smoothed_data


def load_and_preprocess_csv(csv_path: str) -> tuple:

    """
    Loads telemetry CSV and reshapes it from (T, 5) to the model's (5, T) shape.
    """
    print(f"Loading telemetry data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. Verify required channels exist (adjust string names to match your CSV headers)
    required_columns = ['temperature', 'salinity', 'ssh', 'u_velocity','v_velocity']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required oceanographic column in CSV: '{col}'")
            
    # 2. Extract arrays in the exact sequence expected by your observation matrix:
    # Index 0: Temp, Index 1: Salt, Index 2: SSH, Index 3: U_curr
    temp = df['temperature'].values
    salt = df['salinity'].values
    ssh  = df['ssh'].values
    u_curr = df['u_velocity'].values
    v_curr = df['v_velocity'].values
    
    # Extract arrays and engineer features...
    data_stream = engineer_ocean_features(temp, salt, ssh, u_curr, v_curr)
    # data_stream = np.stack([temp, salt, ssh, u_curr, v_curr], axis=0)
    # 1. Capture the raw data BEFORE processing for visualization
    raw_stream = data_stream.copy()

    # 2. Fix NaNs
    for channel in range(data_stream.shape[0]):
        mask = np.isnan(data_stream[channel])
        if np.any(mask):
            xp = np.flatnonzero(~mask)
            fp = data_stream[channel][~mask]
            x = np.flatnonzero(mask)
            data_stream[channel][mask] = np.interp(x, xp, fp)
            raw_stream[channel][mask] = np.interp(x, xp, fp) # Also fix in raw

    # 3. Process (Detrend/Smooth)
    processed_stream = causal_smooth(data_stream,smooth_window=6)
    
    return raw_stream, processed_stream



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
    u_anom = u - np.mean(u)
    v_anom = v - np.mean(v)
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
    ssh_variability = pd.Series(ssh).rolling(window=window, center=True).std().bfill().ffill().values
    
    # Stack vertically -> shape (5, T)
    features = np.vstack((rho, ke, ssh_anom, speed, ssh_variability))
    
    return features



def run_changepoint_analysis(raw_stream, n_breakpoints=1, model_type="rbf"):
    """
    Runs offline changepoint detection on the full multivariate stream.
    
    Args:
        raw_stream:    np.ndarray of shape (P, T) — raw unscaled data
        n_breakpoints: how many breakpoints to search for (1 = single transition)
        model_type:    ruptures cost model: 'rbf', 'l2', or 'l1'
                       - 'rbf': captures variance AND mean shifts (recommended)
                       - 'l2': mean shifts only
                       - 'l1': robust to outliers
    Returns:
        dict with breakpoint location and confidence metrics, or None if unavailable
    """
    # ruptures expects shape (T, P) — transpose from (P, T)
    signal = raw_stream.T  # (T, P)

    # Pelt search with penalty (automatic n_breakpoints via penalty)
    # We use Binseg for exact n_breakpoints control
    algo = rpt.Binseg(model=model_type, min_size=20, jump=5).fit(signal)
    breakpoints = algo.predict(n_bkps=n_breakpoints)

    # breakpoints returns list of END indices of each segment; last is always T
    # so the transition is the first element
    transition_t = breakpoints[0]

    # Compute a simple confidence metric: ratio of between-segment variance
    # to total variance across all channels
    T = signal.shape[0]
    pre  = signal[:transition_t]
    post = signal[transition_t:]

    total_var = np.var(signal, axis=0).mean()
    between_var = (
        np.var(np.vstack([
            np.full((len(pre),  signal.shape[1]), pre.mean(axis=0)),
            np.full((len(post), signal.shape[1]), post.mean(axis=0))
        ]), axis=0).mean()
    )
    confidence = float(np.clip(between_var / (total_var + 1e-8), 0, 1))

    print(f"📊 Changepoint Analysis Complete")
    print(f"    Method          : Binary Segmentation ({model_type})")
    print(f"    Breakpoint at   : timestep {transition_t}")
    print(f"    Confidence score: {confidence:.3f}  (between/total variance ratio)")

    return {
        "breakpoint_timestep": transition_t,
        "confidence": confidence,
        "model_type": model_type,
        "pre_segment_mean":  pre.mean(axis=0).tolist(),
        "post_segment_mean": post.mean(axis=0).tolist(),
    }


def _plot_inference_result(raw_stream, processed_stream, event, cpa_result, channel_names, window_size):
    n_channels, T = processed_stream.shape
    t_axis = np.arange(T)

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2.5 * n_channels), sharex=True)
    
    for i, ax in enumerate(axes):
        # 1. Plot Raw Signal (Faded in background)
        ax.plot(t_axis, raw_stream[i], color='lightgray', linewidth=0.5, alpha=0.6, label='Raw (Original)')
        
        # 2. Plot Processed/Detrended Signal (Foreground)
        ax.plot(t_axis, processed_stream[i], color='steelblue', linewidth=1.0, alpha=0.9, label='Residual (Detrended)')
        
        ax.set_ylabel(channel_names[i], fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- ML prediction ---
        if event is not None:
            ax.axvspan(event["window_start"], event["window_end"],
                       alpha=0.12, color='orange')
            ax.axvline(event["estimated_bifurcation_timestep"],
                       color='crimson', linewidth=1.8, linestyle='--')

        # --- Changepoint result ---
        if cpa_result is not None:
            ax.axvline(cpa_result["breakpoint_timestep"],
                       color='forestgreen', linewidth=1.8, linestyle=':')

        # Legend on top panel only
        if i == 0:
            handles = []
            if event is not None:
                handles += [
                    mpatches.Patch(color='orange', alpha=0.4, label='ML detection window'),
                    plt.Line2D([0], [0], color='crimson', linewidth=1.8, linestyle='--',
                               label=f'ML bifurcation  p={event["probability"]*100:.1f}%  '
                                     f't={event["estimated_bifurcation_timestep"]}'),
                ]
            if cpa_result is not None:
                handles.append(
                    plt.Line2D([0], [0], color='forestgreen', linewidth=1.8, linestyle=':',
                               label=f'CPA breakpoint  conf={cpa_result["confidence"]:.2f}  '
                                     f't={cpa_result["breakpoint_timestep"]}')
                )
            if handles:
                ax.legend(handles=handles, loc='upper left', fontsize=8)

    # Agreement annotation below the top panel
    if event is not None and cpa_result is not None:
        ml_t   = event["estimated_bifurcation_timestep"]
        cpa_t  = cpa_result["breakpoint_timestep"]
        delta  = abs(ml_t - cpa_t)
        agree  = "✅ Methods agree" if delta < 0.05 * T else "⚠️  Methods diverge"
        axes[1].set_title(f"{agree}  |  ML: t={ml_t}  CPA: t={cpa_t}  |  Δ={delta} steps",
                          fontsize=8, color='dimgray', pad=3)

    axes[-1].set_xlabel("Timestep", fontsize=10)
    plt.tight_layout()
    plt.savefig("bifurcation_inference.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to bifurcation_inference.png")


def run_real_world_inference(model, raw_stream, processed_stream, window_size=128, stride=6, 
                             prob_threshold=0.85, channel_names=None):
    
    if channel_names is None:
        channel_names = ['Density (rho)', 'Kinetic Energy', 'SSH Anomaly', 'Speed', 'SSH Variability']

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_timesteps = processed_stream.shape[1]
    results = [] 
    
    last_trigger_t = -1 
    cooldown_period = window_size 
    
    print(f"Deploying model... (Threshold: {prob_threshold})")
    
    with torch.no_grad():
        for start_t in range(0, total_timesteps - window_size, stride):
            end_t = start_t + window_size
            window_slice = processed_stream[:, start_t:end_t]
            x_tensor = torch.tensor(window_slice, dtype=torch.float32).unsqueeze(0).to(device)
            
            logits, pred_time = model(x_tensor, pad_mask=None)
            prob = torch.sigmoid(logits).item()
            
            if prob > prob_threshold:
                if start_t > last_trigger_t + cooldown_period:
                    relative_frame = int(pred_time.item() * window_size)
                    absolute_frame = start_t + relative_frame
                    
                    print(f"⚠️ Bifurcation Found! Window: [{start_t}:{end_t}] | Prob: {prob:.4f}")
                    
                    results.append({
                        "window_start": start_t,
                        "window_end": end_t,
                        "probability": prob,
                        "estimated_bifurcation_timestep": absolute_frame
                    })
                    last_trigger_t = start_t 

    # 1. Run Changepoint Analysis (Independent of inference)
    cpa_result = run_changepoint_analysis(processed_stream)
    
    # 2. Get the first event for plotting
    first_event = results[0] if results else None
    
    # 3. Plotting
    _plot_inference_result(raw_stream, processed_stream, first_event, cpa_result, channel_names, window_size)
    
    return results # Return full list for logging
    
if __name__ == "__main__":
    # --- Paths Configurations ---
    CSV_INPUT_PATH = "CANARI_SSP370_LON_-49.101_LAT_55.939.csv"
    MODEL_WEIGHTS_PATH = "best_bifurcation_model3.pth"
    
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Could not locate model weights at '{MODEL_WEIGHTS_PATH}'.")
        
    # --- 2. INITIALIZE ARCHITECTURE & LOAD WEIGHTS ---
    print("Initializing model topology...")
    # Instantiate your exact BifurcationNet. (num_params=5 matching physical measurements)
    model = BifurcationNet(num_params=5) 
    
    # Loading the trained dictionary parameters
    device_map = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_WEIGHTS_PATH)
    model.load_state_dict(checkpoint)
    print("Trained model weights successfully loaded into memory.")
    
    # --- 3. LOAD DATA STREAM ---

    raw_ocean, processed_ocean = load_and_preprocess_csv(CSV_INPUT_PATH)

    # =====================================================================
    # 1. LOAD THE SCALER
    # =====================================================================
    scaler_path = "synthetic_channel_scaler3.pkl" 
    
    scaler = ChannelWiseScaler.load(scaler_path)

    # # ──────────────────────────────────────────────────────────────────────
    # # DIAGNOSTIC: EXPOSE SCALER TRAINING ORDER
    # # ──────────────────────────────────────────────────────────────────────
    # print("🔮 [SCALER] Internal Means Shape:", scaler.means.shape)
    # print("🔮 [SCALER] Trained Mean Values:\n", scaler.means)
    # # ──────────────────────────────────────────────────────────────────────

    scaled_raw_ocean = scaler.transform(raw_ocean)
    scaled_processed_ocean = scaler.transform(processed_ocean)
        
    # --- 4. EXECUTE WINDOW SLIDING INFERENCE ---
    # Adjust window_size to match one of your trained context windows (128, 256, or 512)
    # Lower stride value means higher temporal resolution, but higher compute cost
    # Pass your required arguments explicitly using keyword arguments
    results = run_real_world_inference(
        model=model, 
        raw_stream=scaled_raw_ocean,
        processed_stream=scaled_processed_ocean,
        window_size=64,
        prob_threshold=0.35
    )
    
    # --- 5. EXPORT INFERENCE REPORT ---
    print("\n----------------────────────────────────")
    print(f"INFERENCE COMPLETE. Total Triggers: {len(results)}")
    print("----------------────────────────────────")
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("detected_ocean_bifurcations3.csv", index=False)
        print("Detailed logs saved to 'detected_ocean_bifurcations3.csv'.")