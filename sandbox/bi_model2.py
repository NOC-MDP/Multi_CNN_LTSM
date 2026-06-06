"""
bi_model.py
───────────
Variable-asynchronous CNN-LSTM bifurcation detector.

Changes from previous version are marked  # ← FIX: <reason>
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR

# ──────────────────────────────────────────────────────────────────────────────
# Dataset  (NEW — was missing entirely; without this the model never sees data)
# ──────────────────────────────────────────────────────────────────────────────

class RecordingDataset(Dataset):
    """
    Wraps a list[Recording] for use with DataLoader.

    FIX: The original file had no Dataset class, so there was no way to feed
    recordings into the training loop.  Without per-sample normalisation the
    CNN receives raw OU + bifurcation signals that differ wildly in scale
    across recordings (σ in [0.15, 0.50], strength in [1.5, 4.0], spike
    magnitude = 5.0).  BatchNorm inside the CNN helps but a per-recording
    z-score first removes inter-sample scale variation so the network can
    focus on shape rather than amplitude.

    Labels
    ──────
    p_true : (1,) float  — 1.0 if positive, 0.0 if null
    t_true : (1,) float  — bifurcation_t / T, normalised to [0, 1]
                           NaN for null recordings (gated out by the loss)
    """
    def __init__(self, recordings, normalise: bool = True, augment: bool = False):
        self.augment = augment
        self.samples = []
        for rec in recordings:
            x = torch.tensor(rec.data, dtype=torch.float32)   # (P, D, T)
            if normalise:
                x = (x - x.mean()) / x.std().clamp(min=1e-6)
            self.samples.append((x, rec.bifurcation_t, x.shape[-1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, bf_t, T = self.samples[idx]
        x = x.clone()   # never mutate the cached tensor in-place

        # ── Augmentation (training only) ──────────────────────────────────
        if self.augment:
            # 1. Time reversal
            if torch.rand(1) < 0.3:
                x = x.flip(-1)
                if bf_t is not None:
                    bf_t = T - bf_t

            # 2. Gaussian noise jitter
            if torch.rand(1) < 0.5:
                x = x + torch.randn_like(x) * 0.02

            # 3. Parameter dropout
            if torch.rand(1) < 0.2:
                p_idx = torch.randint(0, x.shape[0], (1,))
                x[p_idx] = 0.0

        # ── Labels (built after augmentation so bf_t is already updated) ──
        is_pos = bf_t is not None
        p_true = torch.tensor([1.0 if is_pos else 0.0], dtype=torch.float32)
        t_true = torch.tensor(
            [bf_t / T if is_pos else float("nan")],
            dtype=torch.float32,
        )
        return x, p_true, t_true


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class PerVariableDepthTimeEncoder(nn.Module):
    """
    Shared-weight CNN applied independently to each variable's (D, T) slice.

    Input  : (B, P, D, T)
    Output : (B, P, C, T')

    FIX 1 – kernel sizes were (3,5)/(3,5)/(3,3) with padding (1,2)/(1,2)/(1,1).
    With D=32 and two (2,2) + (2,1) MaxPool layers the depth dimension becomes
    32→16→8, so the final (3,3) conv still has room.  BUT with smaller D (e.g.
    D=8 after future pooling) this breaks.  Using AdaptiveAvgPool for depth
    collapse at the end is fine but we should be explicit about it.  No change
    needed here — left as-is with a comment.

    FIX 2 – Dropout2d AFTER BatchNorm zeros entire feature maps, which
    interacts badly with BN statistics on small batches.  Moved Dropout2d to
    AFTER activation, BEFORE the next conv, which is the standard order.
    """

    def __init__(self, out_channels: int, dropout: float = 0.2):   # ← FIX: dropout 0.3→0.2 (see note below)
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64,  kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout),           # ← FIX: was before pool, now after activation
            nn.Conv2d(64, 128, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            # No Dropout2d here — next op is AdaptiveAvgPool which already
            # averages over spatial dims, making channel dropout redundant.
        )
        self.depth_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, D, T = x.shape
        x    = x.reshape(B * P, 1, D, T)
        feat = self.encoder(x)                       # (B*P, C, D', T')
        feat = self.depth_pool(feat).squeeze(2)      # (B*P, C, T')
        _, C, Tp = feat.shape
        return feat.reshape(B, P, C, Tp)             # (B, P, C, T')


class BifurcationRegressor(nn.Module):
    """
    Variable-asynchronous two-head model.

    FIX summary (see inline comments for details):
      A. cnn_channels 256 → 128  — 256 is over-parameterised for D=32, T=512
         with only 800 training samples; causes slow / no convergence.
      B. lstm_hidden  256 → 128  — paired reduction to keep feat_dim=256.
      C. lstm_layers  3   → 2    — 3 layers of bidir LSTM = 6 LSTMs stacked;
         gradient flow is poor and it vastly over-fits small datasets.
      D. Pre-LSTM attention projection added — LSTM input_size must equal
         cnn_channels (C), but after the residual add feat_bp is still (B*P,T',C)
         which is correct.  However the LayerNorm was applied before the residual
         add, so the residual path was un-normalised.  Fixed to post-add norm
         (Pre-LN style is fine too but must be consistent).
      E. Temporal score MLP: Tanh → GELU.  Tanh saturates and kills gradients
         for the timing head early in training when scores are large.
      F. Duplicate assignment bug:  best_val_loss = va["loss"] appeared twice
         in the training loop.
    """

    def __init__(
        self,
        num_params:   int   = 4,
        cnn_channels: int   = 192,   
        lstm_hidden:  int   = 192,    
        lstm_layers:  int   = 2, 
        attn_heads:   int   = 8,     
        dropout:      float = 0.3,   
    ):
        super().__init__()
        self.num_params = num_params
        feat_dim = lstm_hidden * 2   # bidirectional → 256

        # Step 1: per-variable CNN
        self.encoder = PerVariableDepthTimeEncoder(cnn_channels, dropout)

        # Step 2: per-variable temporal self-attention
        # FIX D: LayerNorm applied AFTER residual add (post-LN), not before.
        # Pre-LN (norm → attn → add) also works but was inconsistently applied.
        self.pre_attn = nn.MultiheadAttention(
            embed_dim=cnn_channels,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pre_attn_norm = nn.LayerNorm(cnn_channels)   # applied post-add

        # Step 3: per-variable LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Step 4: temporal attention scoring
        # FIX E: Tanh → GELU in the MLP to avoid saturation early in training
        self.temporal_score = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),                # ← FIX E: was Tanh
            nn.Linear(64, 1),
        )

        # Step 5: cross-variable attention
        self.cross_var_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_var_norm = nn.LayerNorm(feat_dim)      # post-add

        # Step 6: shared trunk + heads
        self.shared = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.detection_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.timing_head = nn.Sequential(
            nn.Linear(256 + num_params, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        B, P, D, T = x.shape

        # 1. Per-variable CNN → (B, P, C, T')
        feat = self.encoder(x)
        _, _, C, Tp = feat.shape

        feat_bp = feat.reshape(B * P, Tp, C)   # (B*P, T', C)

        # 2. Per-variable self-attention  (post-LN)
        # FIX D: residual first, then norm
        attn_out, _ = self.pre_attn(feat_bp, feat_bp, feat_bp)
        feat_bp     = self.pre_attn_norm(feat_bp + attn_out)   # (B*P, T', C)

        # 3. Per-variable LSTM
        lstm_out, _ = self.lstm(feat_bp)        # (B*P, T', H*2)
        H2 = lstm_out.shape[-1]

        # 4. Temporal attention scoring
        time_scores   = self.temporal_score(lstm_out)               # (B*P, T', 1)
        time_weights  = torch.softmax(time_scores, dim=1)           # (B*P, T', 1)
        t_positions   = torch.linspace(0, 1, Tp, device=x.device)  # (T',)
        per_var_timing_flat = (time_weights.squeeze(-1) * t_positions).sum(dim=-1)  # (B*P,)
        per_var_timing      = per_var_timing_flat.reshape(B, P)     # (B, P)

        var_feat = (lstm_out * time_weights).sum(dim=1)             # (B*P, H*2)
        var_feat = var_feat.reshape(B, P, H2)                       # (B, P, H*2)

        # 5. Cross-variable attention  (post-LN)
        cv_out,  _ = self.cross_var_attn(var_feat, var_feat, var_feat)
        var_feat   = self.cross_var_norm(var_feat + cv_out)         # (B, P, H*2)

        global_feat = var_feat.mean(dim=1)      # (B, H*2)

        # 6. Heads
        shared   = self.shared(global_feat)     # (B, 256)
        p_bifurc = self.detection_head(shared)  # (B, 1)
        t_norm   = self.timing_head(torch.cat([shared, per_var_timing], dim=-1))  # (B, 1)

        return p_bifurc, t_norm, per_var_timing


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

class GatedBifurcationLoss(nn.Module):
    """
    L = λ_det · FocalLoss(p_pred, p_true)
      + λ_reg · mask · HuberLoss(t_pred, t_true)

    FIX: focal_alpha 0.75 → 0.25.
    focal_alpha is the weight applied to the POSITIVE class.  With a balanced
    dataset (50 / 50 null / positive) alpha=0.75 heavily up-weights positives,
    creating an artificial class imbalance signal and biasing the detection
    head toward always predicting 1.  For a balanced dataset use 0.25–0.5.
    If your real data is imbalanced, set alpha = 1 - (n_pos / n_total).
    """

    def __init__(
        self,
        lambda_det:  float = 1.0,
        lambda_reg:  float = 2.0,
        focal_alpha: float = 0.25,   # ← FIX: 0.75 → 0.25 for balanced dataset
        focal_gamma: float = 2.0,
        huber_delta: float = 0.1,
    ):
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_reg = lambda_reg
        self.alpha      = focal_alpha
        self.gamma      = focal_gamma
        self.huber      = nn.HuberLoss(reduction="none", delta=huber_delta)

    def forward(self, p_pred, t_pred, p_true, t_true):
        # Focal detection loss
        bce     = F.binary_cross_entropy(p_pred, p_true, reduction="none")
        p_t     = p_pred * p_true + (1 - p_pred) * (1 - p_true)
        focal_w = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * p_true + (1 - self.alpha) * (1 - p_true)
        det_loss = (alpha_t * focal_w * bce).mean()

        # Gated Huber regression loss
        mask     = (~torch.isnan(t_true)) & (p_true > 0)
        reg_loss = torch.tensor(0.0, device=p_pred.device)
        if mask.any():
            reg_loss = self.huber(t_pred[mask], t_true[mask]).mean()

        total = self.lambda_det * det_loss + self.lambda_reg * reg_loss
        return total, {"det_loss": det_loss.item(),
                       "reg_loss": reg_loss.item() if mask.any() else 0.0}


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────

def collate_nan_safe(batch):
    xs, ps, ts = zip(*batch)
    return torch.stack(xs), torch.stack(ps), torch.stack(ts)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    totals = {"loss": 0.0, "det_loss": 0.0, "reg_loss": 0.0, "det_acc": 0.0, "timing_mae": 0.0}
    n_timing, n_total = 0, 0

    for x, p_true, t_true in loader:
        x, p_true, t_true = x.to(device), p_true.to(device), t_true.to(device)
        optimizer.zero_grad()
        p_pred, t_pred, _ = model(x)
        loss, breakdown   = criterion(p_pred, t_pred, p_true, t_true)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        B = x.size(0)
        totals["loss"]     += loss.item() * B
        totals["det_loss"] += breakdown["det_loss"] * B
        totals["reg_loss"] += breakdown["reg_loss"] * B
        totals["det_acc"]  += ((p_pred > 0.5).float() == p_true).float().sum().item()

        mask = (~torch.isnan(t_true)) & (p_true > 0)
        if mask.any():
            mae = (t_pred[mask] - t_true[mask]).abs().mean().item()
            totals["timing_mae"] += mae * mask.sum().item()
            n_timing += mask.sum().item()
        n_total += B

    return {k: v / (n_timing if "timing" in k and n_timing > 0 else n_total)
            for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    totals = {"loss": 0.0, "det_loss": 0.0, "reg_loss": 0.0, "det_acc": 0.0, "timing_mae": 0.0}
    n_timing, n_total = 0, 0

    for x, p_true, t_true in loader:
        x, p_true, t_true = x.to(device), p_true.to(device), t_true.to(device)
        p_pred, t_pred, _ = model(x)
        loss, breakdown   = criterion(p_pred, t_pred, p_true, t_true)

        B = x.size(0)
        totals["loss"]     += loss.item() * B
        totals["det_loss"] += breakdown["det_loss"] * B
        totals["reg_loss"] += breakdown["reg_loss"] * B
        totals["det_acc"]  += ((p_pred > 0.5).float() == p_true).float().sum().item()

        mask = (~torch.isnan(t_true)) & (p_true > 0)
        if mask.any():
            mae = (t_pred[mask] - t_true[mask]).abs().mean().item()
            totals["timing_mae"] += mae * mask.sum().item()
            n_timing += mask.sum().item()
        n_total += B

    return {k: v / (n_timing if "timing" in k and n_timing > 0 else n_total)
            for k, v in totals.items()}


def train(
    model,
    train_ds,
    val_ds,
    device,
    epochs:       int   = 50,      # ← FIX: 30 → 50; cosine LR needs more epochs to be useful
    batch_size:   int   = 64,
    lr:           float = 3e-4,    # ← FIX: 1e-3 → 3e-4; standard AdamW starting LR for transformers
    weight_decay: float = 1e-4,
    patience:     int   = 10,      # ← FIX: 7 → 10; gives cosine schedule more room
    save_path: str = "best_model.pt",
):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_nan_safe, num_workers=4,pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_nan_safe, num_workers=4,pin_memory=True
    )

    criterion = GatedBifurcationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # FIX: add linear warmup for the first 5 epochs before cosine decay.
    # Without warmup, large initial gradients from random init interact badly
    # with the attention layers and can put the model in a bad basin.
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # cosine decay from 1.0 → ~0 over remaining epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,        # 10% warmup
        anneal_strategy='cos',
    )

    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"| train loss={tr['loss']:.4f}  det={tr['det_loss']:.4f}  "
            f"reg={tr['reg_loss']:.4f}  acc={tr['det_acc']:.3f}  "
            f"t_mae={tr['timing_mae']:.4f}"
            f"  || val loss={va['loss']:.4f}  det={va['det_loss']:.4f}  "
            f"acc={va['det_acc']:.3f}  t_mae={va['timing_mae']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if va["loss"] < best_val_loss:
            best_val_loss    = va["loss"]    # ← FIX F: removed duplicate assignment
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ Saved checkpoint (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch}")
                break
        # Save every 10 epochs regardless of val loss
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, f"checkpoint_epoch{epoch:03d}.pt")


    if save_path and Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("  ✓ Loaded best checkpoint")

    return model