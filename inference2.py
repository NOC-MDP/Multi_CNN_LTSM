import os
import torch
import numpy as np
import pandas as pd
# ─── 1. IMPORT YOUR MODEL ARCHITECTURE ───────────────────────────────────────
from pipeline import BifurcationNet, DatasetConfig, ChannelWiseScaler, TemperatureScaler, causal_smooth, engineer_ocean_features
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
    
    # 1. Verify required channels exist
    required_columns = ['temperature', 'salinity', 'ssh', 'u_velocity','v_velocity']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required oceanographic column in CSV: '{col}'")
            
    # 2. Extract arrays in the exact sequence expected by your observation matrix
    temp = df['temperature'].values
    salt = df['salinity'].values
    ssh  = df['ssh'].values
    u_curr = df['u_velocity'].values
    v_curr = df['v_velocity'].values
    
    data_stream = engineer_ocean_features(temp, salt, ssh, u_curr, v_curr)
    raw_stream = data_stream.copy()

    # Fix NaNs
    for channel in range(data_stream.shape[0]):
        mask = np.isnan(data_stream[channel])
        if np.any(mask):
            xp = np.flatnonzero(~mask)
            fp = data_stream[channel][~mask]
            x = np.flatnonzero(mask)
            data_stream[channel][mask] = np.interp(x, xp, fp)
            raw_stream[channel][mask] = np.interp(x, xp, fp)

    processed_stream = causal_smooth(data_stream)
    return raw_stream, processed_stream


def plot_elbow(raw_stream, ensemble, model_type="rbf"):
    signal = raw_stream.T  # (T, P)
    penalties = np.linspace(1, 50, 20)
    costs = []
    n_bkps_list = []
    
    for p in penalties:
        algo = rpt.Pelt(model=model_type).fit(signal)
        bkps = algo.predict(pen=p)
        cost = algo.cost.sum_of_costs(bkps)
        costs.append(cost)
        n_bkps_list.append(len(bkps) - 1) 
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(penalties, costs, marker='o')
    plt.xlabel('Penalty Value')
    plt.ylabel('Total Residual Cost')
    plt.title('Elbow Curve')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(penalties, n_bkps_list, marker='s', color='orange')
    plt.xlabel('Penalty Value')
    plt.ylabel('Number of Breakpoints Detected')
    plt.title('Breakpoints vs Penalty')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"results/ensemble_{ensemble}_elbow.png")
    plt.show()


def run_changepoint_analysis(raw_stream, model_type="rbf", penalty_value=10):
    signal = raw_stream.T  # (T, P)
    algo = rpt.Pelt(model=model_type, min_size=20, jump=5).fit(signal)
    predicted_bkps = algo.predict(pen=penalty_value)
    predicted_bkps = predicted_bkps[:-1]
    
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

    print(f"📊 Changepoint Analysis Complete (Timesteps: {predicted_bkps})")
    return {
        "breakpoint_timesteps": predicted_bkps,
        "confidences": confidences,
        "model_type": model_type,
        "pre_segment_mean":  pre_list,
        "post_segment_mean": post_list,
    }


def _plot_ensemble_inference_result(raw_stream, processed_stream, event, cpa_result, channel_names, window_size, ensemble: int):
    n_channels, T = processed_stream.shape
    t_axis = np.arange(T)

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2.5 * n_channels), sharex=True)
    
    for i, ax in enumerate(axes):
        # 1. Plot Raw and Processed Signal
        ax.plot(t_axis, raw_stream[i], color='lightgray', linewidth=0.5, alpha=0.6, label='Raw (Original)')
        ax.plot(t_axis, processed_stream[i], color='steelblue', linewidth=1.0, alpha=0.9, label='Residual (Detrended)')
        ax.set_ylabel(channel_names[i], fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- Collective Ensemble ML prediction with Uncertainty Bounds ---
        if event is not None:
            # Highlight sliding window span where threshold was crossed
            ax.axvspan(event["window_start"], event["window_end"], alpha=0.08, color='orange')
            
            # Draw mean calculated event trigger timestep
            est_t = event["estimated_bifurcation_timestep"]
            ax.axvline(est_t, color='crimson', linewidth=1.8, linestyle='--')
            
            # Shaded Area: Shows the timing spread uncertainty (±1 Standard Deviation across the folds)
            t_unc = event["timing_uncertainty"]
            ax.axvspan(max(0, est_t - t_unc), min(T, est_t + t_unc), alpha=0.22, color='crimson')

        # --- Offline Changepoint Reference lines ---
        if cpa_result is not None:
            for bkps in cpa_result["breakpoint_timesteps"]:
                ax.axvline(bkps, color='forestgreen', linewidth=1.8, linestyle=':')

        if i == 0:
            handles = []
            if event is not None:
                handles += [
                    mpatches.Patch(color='orange', alpha=0.3, label='ML detection window'),
                    plt.Line2D([0], [0], color='crimson', linewidth=1.8, linestyle='--',
                               label=f'Ensemble Avg Bifurcation t={est_t} (p={event["probability"]*100:.1f}%)'),
                    mpatches.Patch(color='crimson', alpha=0.2, label=f'Fold Timing Uncertainty (±{t_unc:.1f} steps)')
                ]
            if cpa_result is not None and len(cpa_result["breakpoint_timesteps"]) > 0:
                handles.append(
                    plt.Line2D([0], [0], color='forestgreen', linewidth=1.8, linestyle=':',
                               label=f'CPA Breakpoint t={cpa_result["breakpoint_timesteps"][0]}')
                )
            if handles:
                ax.legend(handles=handles, loc='upper left', fontsize=8)

    delta = np.inf
    if event is not None and cpa_result is not None and len(cpa_result["breakpoint_timesteps"]) > 0:
        ml_t   = event["estimated_bifurcation_timestep"]
        cpa_t  = min(cpa_result["breakpoint_timesteps"], key=lambda t: abs(ml_t - t))
        delta  = ml_t - cpa_t
        agree  = "✅ Methods agree" if abs(delta) < 0.05 * T else "⚠️  Methods diverge"
        axes[1].set_title(f"{agree}  |  Ensemble ML Avg: t={ml_t}  CPA: t={cpa_t}  |  Δ={delta} steps",
                          fontsize=8, color='dimgray', pad=3)

    axes[-1].set_xlabel("Timestep", fontsize=10)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/bifurcation_inference_{ensemble}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to results/bifurcation_inference_{ensemble}.png")
    return delta


def run_real_world_ensemble_inference(raw_ocean, processed_ocean, ensemble_idx, n_folds=5, window_size=64, stride=6, 
                                      threshold_type="Youden", fallback_threshold=0.50, channel_names=None, mc_samples=30, elbow=False):
    if channel_names is None:
        channel_names = ['Density (rho)', 'Kinetic Energy', 'SSH Anomaly', 'Speed', 'SSH Variability']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_timesteps = processed_ocean.shape[1]
    
    # Pre-calculate sliding windows to align multi-fold steps perfectly
    window_starts = list(range(0, total_timesteps - window_size, stride))
    n_windows = len(window_starts)
    
    # Lists to dynamically accumulate outputs from valid cross-validation paths
    all_fold_probs = []
    all_fold_times = []
    
    # 1. Look up threshold targets
    resolved_threshold = fallback_threshold
    THRESHOLDS_PATH = f"best_thresholds.json"
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH, 'r') as fp:
            best_thresholds = json.load(fp)
        
        # Check for ensemble-level or fallback to average fold thresholds
        if f"ensemble_{ensemble_idx}" in best_thresholds:
            resolved_threshold = best_thresholds[f"ensemble_{ensemble_idx}"][threshold_type]
        else:
            fold_thresh_vals = [best_thresholds[k][threshold_type] for k in best_thresholds if f"ensemble_{ensemble_idx}_fold_" in k]
            if fold_thresh_vals:
                resolved_threshold = np.mean(fold_thresh_vals)

    print(f"Deploying Model Committee... Resolved Consensus Threshold: {resolved_threshold:.4f}")

    # 2. Iterate and evaluate every single fold model over the timeline
    for fold in range(n_folds):
        fold_suffix = f"{ensemble_idx}_fold_{fold+1}"
        MODEL_WEIGHTS_PATH = f"models/best_bifurcation_model_{fold_suffix}.pth"
        scaler_path = f"scalers/synthetic_channel_scaler_{fold_suffix}.pkl"
        
        if not os.path.exists(MODEL_WEIGHTS_PATH) or not os.path.exists(scaler_path):
            continue
            
        model = BifurcationNet(num_params=5)
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device).eval()
        
        # Use fold-specific scaling metrics for feature extraction transformation alignment
        scaler = ChannelWiseScaler.load(scaler_path)
        scaled_processed_ocean = scaler.transform(processed_ocean)
        
        current_fold_probs = []
        current_fold_times = []
        
        with torch.no_grad():
            for start_t in window_starts:
                end_t = start_t + window_size
                window_slice = scaled_processed_ocean[:, start_t:end_t]
                x_tensor = torch.tensor(window_slice, dtype=torch.float32).unsqueeze(0).to(device)
                pad_mask = torch.zeros(1, window_size, dtype=torch.bool).to(device)
                
                mean_p, _, mean_t, _ = model.mc_forward(x_tensor, pad_mask, n_samples=mc_samples)
                
                current_fold_probs.append(float(mean_p.cpu()))
                current_fold_times.append(float(mean_t.cpu()))
                
        all_fold_probs.append(current_fold_probs)
        all_fold_times.append(current_fold_times)

    # Convert to matrices for array calculations: shape (Active Folds, Windows)
    prob_matrix = np.array(all_fold_probs)
    time_matrix = np.array(all_fold_times)
    
    # 3. Compute Ensemble Mean and Fold Variance Shading Distributions
    avg_probs = np.mean(prob_matrix, axis=0)
    std_probs = np.std(prob_matrix, axis=0)
    
    avg_times = np.mean(time_matrix, axis=0)
    std_times = np.std(time_matrix, axis=0)

    # 4. Extract triggers based entirely on the aggregate cross-fold average probability profile
    results = []
    last_trigger_t = -1
    cooldown_period = window_size
    
    for idx, start_t in enumerate(window_starts):
        end_t = start_t + window_size
        mean_p = avg_probs[idx]
        std_p  = std_probs[idx]
        
        if mean_p > resolved_threshold:
            if start_t > last_trigger_t + cooldown_period:
                relative_frame = int(avg_times[idx] * window_size)
                absolute_frame = start_t + relative_frame
                
                # Dynamic translation of cross-validation timing variance into absolute steps bounds
                timing_uncertainty_steps = std_times[idx] * window_size
                
                print(
                    f"⚠️ Ensemble Consolidated Alert Triggered! "
                    f"Window: [{start_t}:{end_t}] | "
                    f"Consensus Prob: {mean_p:.4f} ± {std_p:.4f} | "
                    f"Target Frame: {absolute_frame} (Uncertainty Spread: ±{timing_uncertainty_steps:.1f} steps)"
                )
                
                results.append({
                    "window_start": start_t,
                    "window_end": end_t,
                    "probability": float(mean_p),
                    "prob_std": float(std_p),
                    "estimated_bifurcation_timestep": absolute_frame,
                    "timing_uncertainty": float(timing_uncertainty_steps)
                })
                last_trigger_t = start_t

    # Initialize a baseline scaler transform for CPA and multi-panel processing
    scaler_f1 = ChannelWiseScaler.load(f"scalers/synthetic_channel_scaler_{ensemble_idx}_fold_1.pkl")
    scaled_raw_ocean_f1 = scaler_f1.transform(raw_ocean)
    scaled_processed_ocean_f1 = scaler_f1.transform(processed_ocean)
    
    cpa_result = run_changepoint_analysis(scaled_raw_ocean_f1)
    if elbow:
        plot_elbow(scaled_raw_ocean_f1, ensemble=ensemble_idx)
        
    first_event = results[0] if results else None
    delta = _plot_ensemble_inference_result(scaled_raw_ocean_f1, scaled_processed_ocean_f1, first_event, cpa_result, channel_names, window_size, ensemble_idx)
    
    if results:
        results[0]["delta"] = delta

    # 5. Generate Average Probability Plot with Shaded Fold Uncertainty Bounds
    plt.figure(figsize=(14, 3))
    plt.plot(window_starts, avg_probs, lw=1.2, color='darkblue', label='Committee Avg P(bifurcation)')
    
    # Fill standard deviation uncertainty spread across all folds
    plt.fill_between(window_starts, np.clip(avg_probs - std_probs, 0, 1), np.clip(avg_probs + std_probs, 0, 1), 
                     color='blue', alpha=0.18, label='Fold Prediction Variance (±1 SD)')
    
    plt.axhline(resolved_threshold, color='red', linestyle='--', label=f'Threshold={resolved_threshold:.3f}')
    plt.axhline(0.5, color='orange', linestyle=':', label='p=0.5')
    plt.xlabel('Timestep')
    plt.ylabel('P(bifurcation)')
    plt.title(f'Ensemble {ensemble_idx} Average Probability Profile with Cross-Fold Variance Shading')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"results/prob_profile_{ensemble_idx}.png", dpi=150)
    plt.close()
    
    return results


if __name__ == "__main__":
    ensemble_st = 1
    ensemble_end = 1
    n_folds = 5
    threshold_type = "Youden" # "f1" | "Youden" | "min_fpr"
    deltas = {}
    elbow = False
    
    for i in range(ensemble_st-1, ensemble_end, 1):
        ensemble_num = i + 1
        CSV_INPUT_PATH = f"ensembles/CANARI_SSP370_{ensemble_num}_LON_-56.506_LAT_60.819.csv"
        NULL_ALERTS_PATH = f"null_alerts.json"
        
        # Guard against checking metrics if filters triggered a warning during validation checks
        if os.path.exists(NULL_ALERTS_PATH):
            with open(NULL_ALERTS_PATH, 'r') as fp:
                null_alerts = json.load(fp)
            
            skip_ensemble = False
            if null_alerts.get(f"ensemble_{ensemble_num}", 0) > 0:
                skip_ensemble = True
            else:
                for f in range(n_folds):
                    if null_alerts.get(f"ensemble_{ensemble_num}_fold_{f+1}", 0) > 0:
                        skip_ensemble = True
                        break
            if skip_ensemble:
                print("=" * 60)
                print(f"Null alerts raised during validation training, skipping inference for Ensemble {ensemble_num}")
                print("=" * 60)
                continue

        try:
            raw_ocean, processed_ocean = load_and_preprocess_csv(CSV_INPUT_PATH)
        except FileNotFoundError as e:
            print(e)
            continue
            
        # Run consolidated K-Fold ensemble stream processor
        results = run_real_world_ensemble_inference(
            raw_ocean=raw_ocean,
            processed_ocean=processed_ocean,
            ensemble_idx=ensemble_num,
            n_folds=n_folds,
            window_size=64,
            stride=6,
            threshold_type=threshold_type,
            elbow=elbow,
            mc_samples=30
        )
        
        if results:
            deltas[f"e_{ensemble_num}"] = results[0]["delta"]
            results_df = pd.DataFrame(results)
            os.makedirs("results", exist_ok=True)
            results_df.to_csv(f"results/detected_ocean_bifurcations_{ensemble_num}.csv", index=False)
            print(f"Consensus logs exported to 'results/detected_ocean_bifurcations_{ensemble_num}.csv'.")
        
    # Generate summary delta chart across active ensembles
    if deltas:
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ['#ff9999' if x < 0 else '#66b3ff' for x in list(deltas.values())]
        bars = ax.barh(list(deltas.keys()), list(deltas.values()), color=colors, edgecolor='grey', height=0.7)
        ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
        ax.axvline(120, color='blue', linewidth=1.0, linestyle='--', alpha=0.5)
        ax.axvline(-120, color='blue', linewidth=1.0, linestyle='--', alpha=0.5)
        ax.axvline(60, color='red', linewidth=1.0, linestyle='-', alpha=0.25)
        ax.axvline(-60, color='red', linewidth=1.0, linestyle='-', alpha=0.25)
        ax.set_xlabel('Timestep Offset')
        ax.set_ylabel('Ensembles')
        ax.set_title('Offsets from Change Point (Ensemble Averaged Predictions)')
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        
        max_val = np.max(list(deltas.values())) + 10
        min_val = np.min(list(deltas.values())) - 10
        if min_val > 0: min_val = -65
        if max_val < 60: max_val = 65
        ax.set_xlim(min_val, max_val)
        
        plt.tight_layout()
        plt.savefig(f'results/deltas_bar_chart.png')
        print("Saved final delta barchart plot.")