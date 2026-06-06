#!/usr/bin/env python
# coding: utf-8

"""
Bifurcation Detection — Regression Training Pipeline & Streaming Deployment
============================================================================

Model outputs TWO heads per window:
  • p_bifurc  : P(a bifurcation exists in this window)          — binary [0,1]
  • t_bifurc  : normalised timestep of bifurcation ∈ [0,1]      — regression
                (only meaningful / penalised when p_bifurc > 0)

Why two heads?
  Null windows have no meaningful regression target.  Penalising the timestep
  head on null samples would force the network to predict an arbitrary number,
  corrupting the gradient signal.  Instead the regression loss is gated by
  whether the window actually contains a bifurcation.

Normalised timestep convention
  t_norm = (bifurcation_absolute_timestep − window_start) / window_length
  So t_norm=0.0 → bifurcation at the very start of the window
     t_norm=1.0 → bifurcation at the very end

Streaming deployment
  A rolling window strides across incoming data. Each stride produces:
    • updated p_bifurc estimate
    • predicted absolute timestep of the bifurcation
  An alert fires when p_bifurc passes a configurable threshold, with
  hysteresis to avoid duplicate alerts.
"""

import numpy as np
import torch
from bi_data_sim import make_synthetic_recordings
from bi_data_struct import BifurcationWindowDataset, recording_level_split
from bi_model2 import (
    BifurcationRegressor,
    GatedBifurcationLoss,
    collate_nan_safe,
    evaluate,
    train,
    RecordingDataset
)
from bi_plot import plot_recordings
from bi_stream import StreamingBifurcationDetector
from torch.utils.data import DataLoader
from syn_data_gen import DatasetConfig, make_synthetic_recordings, dataset_summary, split_dataset

# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ── 1. Generate recordings and split by recording ─────────────────────
print("── Generating synthetic recordings ──")
# all_recordings = make_synthetic_recordings(
#     n_null=1000,
#     n_positive=1000,
#     num_params=4,
#     depth=75,
#     time_len=1048,
#     seed=np.random.randint(1e6, size=1),
# )
# train_recs, val_recs, test_recs = recording_level_split(all_recordings)
# print(
#     f"  Train recordings: {len(train_recs)}  Val: {len(val_recs)}  Test: {len(test_recs)}\n"
# )
cfg = DatasetConfig(
    n_null=3000, n_positive=3000,
    num_params=4, depth=75, time_len=512,
    seed=np.random.randint(1e6, size=1),
)
recordings = make_synthetic_recordings(cfg)
dataset_summary(recordings)

splits = split_dataset(recordings)
print(f"\n  Train recordings : {len(splits['train'])}")
print(f"  Val   recordings : {len(splits['val'])}")
print(f"  Test  recordings : {len(splits['test'])}")

train_recs = splits["train"]
val_recs = splits["val"]
test_recs = splits["test"]

# ── 2. Build window datasets ──────────────────────────────────────────
print("── Building window datasets ──")
WINDOW = 64
STRIDE = 16

train_ds = RecordingDataset(splits["train"], normalise=True, augment=True)
val_ds   = RecordingDataset(splits["val"],   normalise=True, augment=False)
test_ds  = RecordingDataset(splits["test"],  normalise=True, augment=False)
# ── 3. Build and train model ──────────────────────────────────────────
print("── Training ──")
model = BifurcationRegressor(num_params=4).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters: {total_params:,}\n")

model = train(
    model,
    train_ds,
    val_ds,
    device,
    epochs=50,  # increase for real training
    batch_size=64,
    patience=10,
)

# ── 4. Test set evaluation ────────────────────────────────────────────
print("\n── Test set evaluation ──")
test_loader = DataLoader(test_ds, batch_size=64, collate_fn=collate_nan_safe)
criterion = GatedBifurcationLoss()
test_metrics = evaluate(model, test_loader, criterion, device)
print(
    f"  test loss={test_metrics['loss']:.4f}  "
    f"det_acc={test_metrics['det_acc']:.3f}  "
    f"timing_mae={test_metrics['timing_mae']:.4f} (normalised)"
)
print(
    f"  timing_mae in timesteps ≈ {test_metrics['timing_mae'] * WINDOW:.1f} / {WINDOW}"
)

# ── 5. Streaming demo ─────────────────────────────────────────────────
print("\n── Streaming demo ──")
detector = StreamingBifurcationDetector(
    model,
    window_len=WINDOW,
    stride=STRIDE,
    threshold=0.55,
    cooldown_steps=32,
    ema_alpha=0.4,
    device=device,
)

# Pick a positive test recording and stream it in stride-sized chunks
pos_recs = [r for r in test_recs if r.is_positive]
if pos_recs:
    rec = pos_recs[0]
    print(
        f"  Streaming recording '{rec.recording_id}'  "
        f"(true bifurcation at t={rec.bifurcation_t}, "
        f"total T={rec.time_len})"
    )

    for t in range(0, rec.time_len, STRIDE):
        chunk = rec.data[:, :, t : t + STRIDE]
        if chunk.shape[2] == 0:
            print("warning chunk wrong shape!")
            break
        alerts = detector.push(chunk)
        for alert in alerts:
            lead = alert.steps_until_bifurc
            lead_str = f"{lead:+d} steps" if lead >= 0 else f"{abs(lead)} steps AFTER"
            print(
                f"  🚨 ALERT  t={alert.absolute_timestep:4d}  "
                f"p={alert.p_bifurcation:.3f}  "
                f"predicted_t={alert.predicted_bifurc_t}  "
                f"lead={lead_str}"
            )
else:
    print(
        "  (No positive recordings in test split — increase n_positive or use a fixed seed)"
    )

print(f"Highlight: {rec.recording_id}")
print(f"  Actual bifurcation    : t={rec.bifurcation_t}")
print(f"  Prediction bifurcation  : t={alert.predicted_bifurc_t}")

# ── 6. Plot recordings ─────────────────────────────────────────────────
plot_recordings(
    recordings=test_recs,
    highlight=rec,
    predicted_t=alert.predicted_bifurc_t,
    depth_summary="mean",
    param_names=["Parameter 1", "Parameter 2", "Parameter 3", "Parameter 4"],
    save_path="bifurcation_plot.png",
)

# might be needed to adjust for prior assumption of 50/50 bifurcation/null

# def adjust_for_prior(p_pred, train_prevalence=0.5, deploy_prevalence=0.1):
#     """
#     Shift model output probabilities to account for a different deployment
#     prior without retraining.  Uses Bayes' theorem on the likelihood ratio.
#     """
#     # Convert to likelihood ratio (removes the training prior)
#     lr = (p_pred / (1 - p_pred + 1e-8)) * \
#          ((1 - train_prevalence) / train_prevalence)
#     # Apply deployment prior
#     p_adjusted = (lr * deploy_prevalence) / \
#                  (lr * deploy_prevalence + (1 - deploy_prevalence))
#     return p_adjusted