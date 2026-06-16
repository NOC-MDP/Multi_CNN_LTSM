import os
import torch
import numpy as np
import pandas as pd
# ─── 1. IMPORT YOUR MODEL ARCHITECTURE ───────────────────────────────────────
from pipeline import BifurcationNet, DatasetConfig, ChannelWiseScaler, TemperatureScaler,causal_smooth,engineer_ocean_features
# ─────────────────────────────────────────────────────────────────────────────
import ruptures as rpt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from scipy.stats import linregress
import json


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
    processed_stream = causal_smooth(data_stream)
    
    return raw_stream, processed_stream

def plot_elbow(raw_stream, ensemble,model_type="rbf"):
    signal = raw_stream.T  # (T, P)
    penalties = np.linspace(1, 50, 20)
    costs = []
    n_bkps_list = []
    
    for p in penalties:
        algo = rpt.Pelt(model=model_type).fit(signal)
        bkps = algo.predict(pen=p)
        
        # Calculate the sum of costs (residual error) for this segmentation
        cost = algo.cost.sum_of_costs(bkps)
        costs.append(cost)
        
        # The last element in bkps is always the length of the signal, so we subtract 1
        n_bkps_list.append(len(bkps) - 1) 
    
    # Plotting the Elbow Curve
    plt.figure(figsize=(10, 4))
    
    # Plot 1: Penalty vs Cost
    plt.subplot(1, 2, 1)
    plt.plot(penalties, costs, marker='o')
    plt.xlabel('Penalty Value')
    plt.ylabel('Total Residual Cost')
    plt.title('Elbow Curve (Look for the flattening)')
    plt.grid(True)
    
    # Plot 2: Penalty vs Number of Breakpoints
    plt.subplot(1, 2, 2)
    plt.plot(penalties, n_bkps_list, marker='s', color='orange')
    plt.xlabel('Penalty Value')
    plt.ylabel('Number of Breakpoints Detected')
    plt.title('Breakpoints vs Penalty')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"results/changepoints/ensemble_{ensemble}_elbow.png")
    plt.show()

def run_changepoint_analysis(raw_stream, model_type="rbf",penalty_value=10):
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

    # Pelt search with penalty
    algo = rpt.Pelt(model=model_type, min_size=20, jump=5).fit(signal)
    # The critical part: Use 'pen' (penalty) instead of 'n_bkps'
    # A higher penalty = fewer change points detected.
    # A lower penalty = more change points detected.
    predicted_bkps = algo.predict(pen=penalty_value)
    # last breakpoint is always end of time series so remove.
    predicted_bkps = predicted_bkps[:-1]
    # Compute a simple confidence metric: ratio of between-segment variance
    # to total variance across all channels
    T = signal.shape[0]
    confidences = []
    pre_list = []
    post_list = []
    for bkps in predicted_bkps:
        pre  = signal[:bkps]
        post = signal[bkps:]
    
        total_var = np.var(signal, axis=0).mean()
        between_var = (
            np.var(np.vstack([
                np.full((len(pre),  signal.shape[1]), pre.mean(axis=0)),
                np.full((len(post), signal.shape[1]), post.mean(axis=0))
            ]), axis=0).mean()
        )
        confidences.append(float(np.clip(between_var / (total_var + 1e-8), 0, 1)))
        pre_list.append(pre.mean(axis=0).tolist())
        post_list.append(post.mean(axis=0).tolist())

    print(f"📊 Changepoint Analysis Complete")
    print(f"    Method PELT Algorithm       :  ({model_type})")
    print(f"    Predicted Breakpoints at   : timestep {predicted_bkps}")
    print(f"    Confidence scores: {confidences}  (between/total variance ratio)")

    return {
        "breakpoint_timesteps": predicted_bkps,
        "confidences": confidences,
        "model_type": model_type,
        "pre_segment_mean":  pre_list,
        "post_segment_mean": post_list,
    }


def _plot_inference_result(raw_stream, processed_stream, event, cpa_result, channel_names, window_size,ensemble:int):
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
            for bkps in cpa_result["breakpoint_timesteps"]:
                ax.axvline(bkps,color='forestgreen', linewidth=1.8, linestyle=':')

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
                               label=f'CPA breakpoint  conf={cpa_result["confidences"][0]:.2f}  '
                                     f't={cpa_result["breakpoint_timesteps"][0]}')
                )
            if handles:
                ax.legend(handles=handles, loc='upper left', fontsize=8)
    delta = np.inf
    # Agreement annotation below the top panel
    if event is not None and cpa_result is not None:
        ml_t   = event["estimated_bifurcation_timestep"]
        cpa_t  = min(cpa_result["breakpoint_timesteps"], key=lambda t: abs(ml_t - t))
        delta  = ml_t - cpa_t
        agree  = "✅ Methods agree" if delta < 0.05 * T else "⚠️  Methods diverge"
        axes[1].set_title(f"{agree}  |  ML: t={ml_t}  CPA: t={cpa_t}  |  Δ={delta} steps",
                          fontsize=8, color='dimgray', pad=3)

    axes[-1].set_xlabel("Timestep", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"results/bifurcation_inference_{ensemble}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to bifurcation_inference.png")
    return delta


def run_real_world_inference(model, raw_stream, processed_stream,ensemble, ts,window_size=128, stride=6, 
                             prob_threshold=0.50, channel_names=None,mc_samples=30,elbow=False):
    
    if channel_names is None:
        channel_names = ['Density (rho)', 'Kinetic Energy', 'SSH Anomaly', 'Speed', 'SSH Variability']

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_timesteps = processed_stream.shape[1]
    results = [] 
    
    last_trigger_t = -1 
    cooldown_period = window_size 
    all_probs = []
    print(f"Deploying model... (Threshold: {prob_threshold})")
    
    with torch.no_grad():
        for start_t in range(0, total_timesteps - window_size, stride):
            end_t = start_t + window_size
            window_slice = processed_stream[:, start_t:end_t] 
            x_tensor = torch.tensor(window_slice, dtype=torch.float32).unsqueeze(0).to(device)
            pad_mask = torch.zeros(1, window_size, dtype=torch.bool).to(device)

            mean_p, std_p, mean_t,std_t = model.mc_forward(x_tensor, pad_mask, n_samples=mc_samples)

            all_probs.append((start_t, float(mean_p.detach().cpu())))
            if mean_p > prob_threshold:
                if start_t > last_trigger_t + cooldown_period:
                    relative_frame = int(mean_t.item() * window_size)
                    absolute_frame = start_t + relative_frame
                    
                    print(
                        f"⚠️ Bifurcation Found! "
                        f"Window: [{start_t}:{end_t}] | "
                        f"Prob: {float(mean_p):.4f} ± {float(std_p):.4f} | "
                        f"Timing: {float(mean_t):.1f} ± {float(std_t):.1f}"
                    )
                    
                    results.append({
                        "window_start": start_t,
                        "window_end": end_t,
                        "probability": float(mean_p),
                        "estimated_bifurcation_timestep": absolute_frame
                    })
                    last_trigger_t = start_t 

    # 1. Run Changepoint Analysis (Independent of inference)
    cpa_result = run_changepoint_analysis(raw_stream)
    if elbow:
        plot_elbow(raw_stream,ensemble=ensemble)
    
    # 2. Get the first event for plotting
    first_event = results[0] if results else None
    

    delta =_plot_inference_result(raw_stream, processed_stream, first_event, cpa_result, channel_names, window_size,ensemble)

    if results:
        results[0]["delta"] = delta
    
    # After the loop, print the probability profile:
    all_probs = np.array(all_probs)
    print(f"Max probability seen  : {all_probs[:,1].max():.4f} at t={int(all_probs[all_probs[:,1].argmax(), 0])}")
    print(f"Mean probability      : {all_probs[:,1].mean():.4f}")
    print(f"% windows above 0.45  : {(all_probs[:,1] > 0.45).mean()*100:.1f}%")
    print(f"% windows above 0.50  : {(all_probs[:,1] > 0.50).mean()*100:.1f}%")
    print(f"% windows above thresh: {(all_probs[:,1] > prob_threshold).mean()*100:.1f}%")

    plt.figure(figsize=(14, 3))
    plt.plot(all_probs[:, 0], all_probs[:, 1], lw=0.8)
    plt.axhline(prob_threshold, color='red', linestyle='--', label=f'threshold={prob_threshold}')
    plt.axhline(0.5, color='orange', linestyle=':', label='p=0.5')
    plt.xlabel('Timestep'); plt.ylabel('P(bifurcation)')
    plt.title('Probability profile — SSP370 series')
    plt.legend(); plt.tight_layout()
    plt.savefig(f"results/prob_profile_{ensemble}.png", dpi=150)
    
    return results # Return full list for logging
    
if __name__ == "__main__":
    ensemble_st = 1
    ensemble_end = 40
    deltas = {}
    elbow = False
    for i in range(ensemble_st-1,ensemble_end,1):
        # --- Paths Configurations ---
        CSV_INPUT_PATH = f"ensembles/CANARI_SSP370_{i+1}_LON_-56.506_LAT_60.819.csv"
        MODEL_WEIGHTS_PATH = f"models/best_bifurcation_model_{i+1}.pth"
        THRESHOLDS_PATH = "best_thresholds.json"
        NULL_ALERTS_PATH = "null_alerts.json"
        with open(NULL_ALERTS_PATH, 'r') as fp:
            null_alerts = json.load(fp)
        try:
            if null_alerts[f"ensemble_{i+1}"] > 0:
                print("=" * 60)
                print(f"null alerts raised during training, skipping inference for ensemble {i+1}")
                print("=" * 60)
                continue
        except KeyError as e:
            print(e)
            
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            print(f"Could not locate model weights at '{MODEL_WEIGHTS_PATH}', skipping inference")
            continue
            
        # --- 2. INITIALIZE ARCHITECTURE & LOAD WEIGHTS ---
        print("Initializing model topology...")
        # Instantiate your exact BifurcationNet. (num_params=6 matching physical measurements)
        model = BifurcationNet(num_params=5) 
        
        # Loading the trained dictionary parameters
        device_map = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(MODEL_WEIGHTS_PATH)
        model.load_state_dict(checkpoint)
        print("Trained model weights successfully loaded into memory.")
    
        ts_data = torch.load(f"scalers/temperature_scaler_{i+1}.pt", weights_only=True)
        ts = TemperatureScaler()
        ts.temperature = torch.nn.Parameter(torch.tensor([ts_data['temperature']]))
        ts.eval()

        with open(THRESHOLDS_PATH, 'r') as fp:
            best_thresholds = json.load(fp)
        
        # --- 3. LOAD DATA STREAM ---
        try:
            raw_ocean, processed_ocean = load_and_preprocess_csv(CSV_INPUT_PATH)
        except FileNotFoundError as e:
            print(e)
            continue
    
        # =====================================================================
        # 1. LOAD THE SCALER
        # =====================================================================
        scaler_path = f"scalers/synthetic_channel_scaler_{i+1}.pkl" 
        
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
                ts=ts,
                window_size=128,
                prob_threshold=best_thresholds[f"ensemble_{i+1}"],
                ensemble=i+1,
                elbow = elbow
            )
        if results:
            deltas[f"e_{i+1}"] = results[0]["delta"]
        # --- 5. EXPORT INFERENCE REPORT ---
        print("\n----------------────────────────────────")
        print(f"INFERENCE COMPLETE. Total Triggers: {len(results)}")
        print("----------------────────────────────────")
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"results/detected_ocean_bifurcations_{i+1}.csv", index=False)
            print(f"Detailed logs saved to 'detected_ocean_bifurcations_{i+1}.csv'.")
    
    # 3. Initialize the plot using subplots
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # 4. Define contrasting colors for negative (before) and positive (after) deltas
    colors = ['#ff9999' if x < 0 else '#66b3ff' for x in list(deltas.values())]
    
    # 5. Plot the horizontal bars extending from the central axis
    bars = ax.barh(list(deltas.keys()), list(deltas.values()), color=colors, edgecolor='grey', height=0.7)
    
    # 6. Establish a prominent vertical central zero line
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--')

    # 10 year lines +/-
    ax.axvline(120, color='blue', linewidth=1.0, linestyle='--',alpha=0.5)
    ax.axvline(-120, color='blue', linewidth=1.0, linestyle='--',alpha=0.5)
    ax.axvline(60, color='red', linewidth=1.0, linestyle='-',alpha=0.25)
    ax.axvline(-60, color='red', linewidth=1.0, linestyle='-',alpha=0.25)
    # 7. Customize axis labels and titles using LaTeX formatting
    ax.set_xlabel('Timestep Offset')
    ax.set_ylabel('Ensembles')
    ax.set_title('Offsets from Change Point')
    
    # 8. Add a subtle grid along the x-axis for readability
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    
    # 9. Center the view properly around the zero line
    max_val = np.max(list(deltas.values())) + 10
    min_val = np.min(list(deltas.values())) - 10
    ax.set_xlim(min_val, max_val)
    
    # Adjust layout and save the visualization
    plt.tight_layout()
    plt.savefig('results/deltas_bar_chart.png')
    print("saved delta barchart plot")