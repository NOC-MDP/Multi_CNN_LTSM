from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────────────────────────────────────
# Model — two-head CNN-LSTM
# ──────────────────────────────────────────────────────────────────────────────

class DepthTimeEncoder(nn.Module):
    def __init__(self, num_params: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_params, 64,  kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 128, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(dropout),

            nn.Conv2d(128, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.depth_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)          # (B, C, D', T')
        feat = self.depth_pool(feat)    # (B, C, 1, T')
        return feat.squeeze(2)          # (B, C, T')


class BifurcationRegressor(nn.Module):
    """
    Two-head model:
      • detection_head  → P(bifurcation in window)   scalar in [0, 1]
      • timing_head     → normalised timestep t_norm  scalar in [0, 1+]
                          (sigmoid output, so naturally bounded near [0,1])

    The timing head output is only trusted when detection_head > threshold.
    """

    def __init__(
        self,
        num_params: int  = 4,
        cnn_channels: int = 256,
        lstm_hidden: int  = 256,
        lstm_layers: int  = 3,
        attn_heads: int   = 8,
        dropout: float    = 0.3,
    ):
        super().__init__()

        self.encoder = DepthTimeEncoder(num_params, cnn_channels, dropout)

        self.attn_norm = nn.LayerNorm(cnn_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=cnn_channels, num_heads=attn_heads,
            dropout=dropout, batch_first=True,
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        feat_dim = lstm_hidden * 2

        # Shared trunk
        self.shared = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Head 1: binary detection
        self.detection_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Head 2: timing regression
        # Sigmoid keeps output in [0, 1]; for precursor windows slightly > 1
        # the model will be close enough for deployment purposes.
        self.timing_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, P, D, T)
        feat = self.encoder(x)              # (B, C, T')
        feat = feat.permute(0, 2, 1)        # (B, T', C)

        residual = feat
        feat = self.attn_norm(feat)
        feat, _ = self.attn(feat, feat, feat)
        feat = feat + residual

        out, _ = self.lstm(feat)            # (B, T', H*2)
        out = out[:, -1, :]                 # (B, H*2)

        shared = self.shared(out)           # (B, 256)
        p_bifurc = self.detection_head(shared)   # (B, 1)
        t_norm   = self.timing_head(shared)       # (B, 1)

        return p_bifurc, t_norm


# ──────────────────────────────────────────────────────────────────────────────
# Loss — gated regression
# ──────────────────────────────────────────────────────────────────────────────

class GatedBifurcationLoss(nn.Module):
    """
    Combined loss:

      L = λ_det · FocalLoss(p_pred, p_true)
        + λ_reg · mask · HuberLoss(t_pred, t_true)

    where mask = 1 only for windows that actually contain a bifurcation.

    Gating the regression loss is critical:
      • Null windows have no ground-truth timestep — punishing the network
        for any output on those samples corrupts training.
      • It also means the model is free to output any timing on null windows
        without penalty; only the detection head matters there.
    """

    def __init__(
        self,
        lambda_det: float = 1.0,
        lambda_reg: float = 2.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        huber_delta: float = 0.1,
    ):
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_reg = lambda_reg
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.huber = nn.HuberLoss(reduction="none", delta=huber_delta)

    def forward(
        self,
        p_pred: torch.Tensor,   # (B, 1)
        t_pred: torch.Tensor,   # (B, 1)
        p_true: torch.Tensor,   # (B, 1)
        t_true: torch.Tensor,   # (B, 1)  — NaN for null windows
    ) -> tuple[torch.Tensor, dict]:

        # ── Focal detection loss ──────────────────────────────────────────
        bce = F.binary_cross_entropy(p_pred, p_true, reduction="none")
        p_t = p_pred * p_true + (1 - p_pred) * (1 - p_true)
        focal_w = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * p_true + (1 - self.alpha) * (1 - p_true)
        det_loss = (alpha_t * focal_w * bce).mean()

        # ── Gated Huber regression loss ───────────────────────────────────
        mask = (~torch.isnan(t_true)) & (p_true > 0)   # (B, 1) bool
        reg_loss = torch.tensor(0.0, device=p_pred.device)

        if mask.any():
            t_pred_masked = t_pred[mask]
            t_true_masked = t_true[mask]
            reg_loss = self.huber(t_pred_masked, t_true_masked).mean()

        total = self.lambda_det * det_loss + self.lambda_reg * reg_loss

        return total, {
            "det_loss": det_loss.item(),
            "reg_loss": reg_loss.item() if mask.any() else 0.0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────

def collate_nan_safe(batch):
    """Custom collate that stacks tensors containing NaN safely."""
    xs, ps, ts = zip(*batch)
    return torch.stack(xs), torch.stack(ps), torch.stack(ts)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    totals = {"loss": 0, "det_loss": 0, "reg_loss": 0, "det_acc": 0, "timing_mae": 0}
    n_timing, n_total = 0, 0

    for x, p_true, t_true in loader:
        x, p_true, t_true = x.to(device), p_true.to(device), t_true.to(device)
        optimizer.zero_grad()

        p_pred, t_pred = model(x)
        loss, breakdown = criterion(p_pred, t_pred, p_true, t_true)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        B = x.size(0)
        totals["loss"]     += loss.item() * B
        totals["det_loss"] += breakdown["det_loss"] * B
        totals["reg_loss"] += breakdown["reg_loss"] * B
        totals["det_acc"]  += ((p_pred > 0.5).float() == p_true).float().sum().item()

        # Timing MAE on positive windows only
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
    totals = {"loss": 0, "det_loss": 0, "reg_loss": 0, "det_acc": 0, "timing_mae": 0}
    n_timing, n_total = 0, 0

    for x, p_true, t_true in loader:
        x, p_true, t_true = x.to(device), p_true.to(device), t_true.to(device)
        p_pred, t_pred = model(x)
        loss, breakdown = criterion(p_pred, t_pred, p_true, t_true)

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
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    save_path: Optional[str] = "best_model.pt",
):
    """
    Full training loop with:
      • AdamW + cosine LR schedule
      • Early stopping on val loss
      • Best checkpoint saving
    """
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_nan_safe, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_nan_safe, num_workers=0,
    )

    criterion = GatedBifurcationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"| train loss={tr['loss']:.4f}  det={tr['det_loss']:.4f}  reg={tr['reg_loss']:.4f}  "
            f"acc={tr['det_acc']:.3f}  t_mae={tr['timing_mae']:.4f}"
            f"  || val loss={va['loss']:.4f}  det={va['det_loss']:.4f}  "
            f"acc={va['det_acc']:.3f}  t_mae={va['timing_mae']:.4f}"
        )

        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ Saved checkpoint (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    if save_path and Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("  ✓ Loaded best checkpoint")

    return model