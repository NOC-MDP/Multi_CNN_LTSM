import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import Optional
# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Recording:
    """
    A single (possibly multi-parameter) depth×time recording.

    data            : np.ndarray  shape (num_params, depth, time)
    bifurcation_t   : int | None  — absolute timestep of bifurcation,
                                    or None if this is a null recording
    recording_id    : str         — unique identifier (for record-level splits)
    """
    data: np.ndarray
    bifurcation_t: Optional[int]
    recording_id: str

    @property
    def is_positive(self) -> bool:
        return self.bifurcation_t is not None

    @property
    def num_params(self) -> int:
        return self.data.shape[0]

    @property
    def depth(self) -> int:
        return self.data.shape[1]

    @property
    def time_len(self) -> int:
        return self.data.shape[2]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset  — recording-level split, correct window labelling
# ──────────────────────────────────────────────────────────────────────────────

class BifurcationWindowDataset(Dataset):
    """
    Slices recordings into fixed-length windows with stride.

    Label assignment (critical to get right)
    ─────────────────────────────────────────
    For each window [t_start, t_start + window_len):
      • NULL recording              → p=0, t_norm=NaN  (regression ignored)
      • Positive, bifurc before window start → p=0, t_norm=NaN  (precursor only)
      • Positive, bifurc inside window       → p=1, t_norm=(bifurc-t_start)/window_len
      • Positive, bifurc after window end    → p=0, t_norm=NaN  (no bifurc yet)

    The third case is the only positive training example.  This ensures the
    model learns to detect the bifurcation event itself (and its precursors
    within the window), not just the dynamics of a positive recording.

    Precursor window option
    ───────────────────────
    If precursor_steps > 0, windows that *end* exactly precursor_steps before
    the bifurcation are also labelled positive, with t_norm slightly > 1.0.
    This teaches the model to anticipate approaching bifurcations.
    Leave at 0 to disable.
    """

    def __init__(
        self,
        recordings: list[Recording],
        window_len: int = 128,
        stride: int = 16,
        precursor_steps: int = 0,
        normalise: bool = True,
    ):
        self.window_len = window_len
        self.stride = stride
        self.normalise = normalise

        # Each sample: (recording_idx, t_start, p_label, t_norm_label)
        self.samples: list[tuple[int, int, float, float]] = []
        self.recordings = recordings

        for rec_idx, rec in enumerate(recordings):
            T = rec.time_len
            for t_start in range(0, T - window_len + 1, stride):
                t_end = t_start + window_len

                if not rec.is_positive:
                    self.samples.append((rec_idx, t_start, 0.0, float("nan")))
                    continue

                bf = rec.bifurcation_t

                # Bifurcation strictly inside this window
                if t_start <= bf < t_end:
                    t_norm = (bf - t_start) / window_len
                    self.samples.append((rec_idx, t_start, 1.0, t_norm))

                # Precursor window: bifurcation is just ahead
                elif precursor_steps > 0 and (t_end <= bf < t_end + precursor_steps):
                    t_norm = (bf - t_start) / window_len   # > 1.0, intentionally
                    self.samples.append((rec_idx, t_start, 1.0, t_norm))

                else:
                    self.samples.append((rec_idx, t_start, 0.0, float("nan")))

        pos = sum(1 for *_, p, _ in self.samples if p > 0)
        print(
            f"  Dataset: {len(self.samples)} windows  "
            f"({pos} positive = {100*pos/len(self.samples):.1f}%)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec_idx, t_start, p_label, t_norm = self.samples[idx]
        rec = self.recordings[rec_idx]

        # Slice window: (P, D, T_window)
        x = rec.data[:, :, t_start : t_start + self.window_len].copy()
        x = torch.from_numpy(x).float()

        if self.normalise:
            # Per-window z-score per parameter
            mean = x.mean(dim=(1, 2), keepdim=True)
            std  = x.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            x = (x - mean) / std

        p_tensor = torch.tensor([p_label], dtype=torch.float32)
        t_tensor = torch.tensor([t_norm],  dtype=torch.float32)

        return x, p_tensor, t_tensor


def recording_level_split(
    recordings: list[Recording],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Recording], list[Recording], list[Recording]]:
    """
    Split recordings (NOT windows) into train / val / test.

    Splitting by window would allow windows from the same recording to
    appear in both train and val — the model would memorise the recording.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(recordings)).tolist()

    n_val  = max(1, int(len(recordings) * val_frac))
    n_test = max(1, int(len(recordings) * test_frac))

    test_idx  = indices[:n_test]
    val_idx   = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    return (
        [recordings[i] for i in train_idx],
        [recordings[i] for i in val_idx],
        [recordings[i] for i in test_idx],
    )

