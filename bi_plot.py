"""
Bifurcation Dataset Visualiser
===============================
Shows all recordings overlaid per parameter, with:
  • Null recordings         — thin dark lines, low opacity
  • Positive recordings     — thin lines + grey shaded region after bifurcation
  • Highlighted recording   — bold red line + red dashed actual bifurcation line
  • Predicted bifurcation   — blue dashed vertical line (optional)

Usage
─────
    # After training / streaming inference:
    plot_recordings(
        recordings       = all_recordings,
        highlight        = my_recording,
        predicted_t      = 312,           # from streaming detector (optional)
        depth_summary    = "mean",        # how to collapse depth axis
        save_path        = "bifurc.png",  # or None to show interactively
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Optional

# Import your data structures — adjust the import path to match your project
# from bifurcation_training import Recording, make_synthetic_recordings


# ──────────────────────────────────────────────────────────────────────────────
# Core plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_recordings(
    recordings: list,                       # list[Recording]
    highlight: object,                      # Recording to draw in red
    predicted_t: Optional[int] = None,      # predicted bifurcation timestep
    depth_summary: str = "mean",            # "mean" | "max" | "std" | int (depth index)
    param_names: Optional[list[str]] = None,
    figsize_per_param: tuple = (14, 2.8),
    null_color: str = "#2c2c2c",
    null_alpha: float = 0.12,
    null_lw: float = 0.6,
    pos_color: str = "#5a5a5a",
    pos_alpha: float = 0.18,
    pos_lw: float = 0.6,
    shade_color: str = "#c8c8c8",
    shade_alpha: float = 0.25,
    highlight_color: str = "#d62728",
    highlight_lw: float = 2.0,
    actual_line_color: str = "#d62728",
    predicted_line_color: str = "#1f77b4",
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """
    Parameters
    ──────────
    recordings      All Recording objects (nulls + positives).
    highlight       The single Recording to draw prominently in red.
    predicted_t     Absolute timestep predicted by the streaming detector.
                    Pass None to omit the predicted line.
    depth_summary   How to collapse the depth dimension before plotting:
                      "mean"  — average across depth (default)
                      "max"   — max across depth
                      "std"   — std across depth (useful for instability signals)
                      int     — use a single depth slice at that index
    param_names     Optional list of axis labels, one per parameter.
    """
    num_params = highlight.data.shape[0]
    T_highlight = highlight.data.shape[2]

    if param_names is None:
        param_names = [f"Parameter {i+1}" for i in range(num_params)]

    fig, axes = plt.subplots(
        num_params, 1,
        figsize=(figsize_per_param[0], figsize_per_param[1] * num_params),
        sharex=True,
    )
    if num_params == 1:
        axes = [axes]

    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#161616")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.yaxis.label.set_color("#cccccc")
        ax.xaxis.label.set_color("#cccccc")

    def summarise(data, param_idx):
        """Collapse (depth, time) → (time,) for one parameter."""
        d = data[param_idx]                     # (depth, time)
        if isinstance(depth_summary, int):
            return d[depth_summary]
        elif depth_summary == "mean":
            return d.mean(axis=0)
        elif depth_summary == "max":
            return d.max(axis=0)
        elif depth_summary == "std":
            return d.std(axis=0)
        else:
            raise ValueError(f"Unknown depth_summary='{depth_summary}'")

    # ── Pass 1: nulls ─────────────────────────────────────────────────────
    for rec in recordings:
        if rec.is_positive or rec is highlight:
            continue
        T = rec.data.shape[2]
        t = np.arange(T)
        for p_idx, ax in enumerate(axes):
            signal = summarise(rec.data, p_idx)
            ax.plot(t, signal, color=null_color, alpha=null_alpha,
                    lw=null_lw, rasterized=True)

    # ── Pass 2: positives (not highlight) — line + shaded post-bifurc ────
    for rec in recordings:
        if not rec.is_positive or rec is highlight:
            continue
        T = rec.data.shape[2]
        t = np.arange(T)
        bf = rec.bifurcation_t
        for p_idx, ax in enumerate(axes):
            signal = summarise(rec.data, p_idx)
            ax.plot(t, signal, color=pos_color, alpha=pos_alpha,
                    lw=pos_lw, rasterized=True)
            # Grey shading after bifurcation
            ax.axvspan(bf, T - 1, color=shade_color, alpha=shade_alpha,
                       zorder=0, lw=0)

    # ── Pass 3: highlighted recording ─────────────────────────────────────
    T_h = highlight.data.shape[2]
    t_h = np.arange(T_h)

    for p_idx, ax in enumerate(axes):
        signal = summarise(highlight.data, p_idx)

        # Split into pre / post bifurcation for subtle shading
        if highlight.is_positive:
            bf = highlight.bifurcation_t
            # Post-bifurcation shading specific to this recording
            ax.axvspan(bf, T_h - 1,
                       color=highlight_color, alpha=0.07, zorder=1, lw=0)

        # Main highlighted signal
        ax.plot(t_h, signal,
                color=highlight_color, lw=highlight_lw,
                alpha=0.95, zorder=4, rasterized=True)

        # ── Vertical lines ────────────────────────────────────────────────
        if highlight.is_positive:
            ax.axvline(
                highlight.bifurcation_t,
                color=actual_line_color, lw=1.5, ls="--",
                alpha=0.9, zorder=5,
                label="Actual bifurcation" if p_idx == 0 else None,
            )

        if predicted_t is not None:
            ax.axvline(
                predicted_t,
                color=predicted_line_color, lw=1.5, ls="--",
                alpha=0.9, zorder=5,
                label="Predicted bifurcation" if p_idx == 0 else None,
            )

        # ── Y-label ───────────────────────────────────────────────────────
        label = param_names[p_idx]
        if depth_summary != "mean":
            summary_label = (
                f"depth[{depth_summary}]" if isinstance(depth_summary, int)
                else depth_summary
            )
            label += f"\n({summary_label})"
        ax.set_ylabel(label, fontsize=9, color="#cccccc")
        ax.grid(axis="x", color="#2a2a2a", lw=0.5)
        ax.grid(axis="y", color="#1e1e1e", lw=0.4)

    axes[-1].set_xlabel("Timestep", fontsize=10, color="#aaaaaa")

    # ── Title ─────────────────────────────────────────────────────────────
    rec_id = getattr(highlight, "recording_id", "highlighted")
    title_parts = [f"Recording: {rec_id}"]
    if highlight.is_positive:
        title_parts.append(f"actual bifurcation t={highlight.bifurcation_t}")
    if predicted_t is not None:
        err = predicted_t - highlight.bifurcation_t if highlight.is_positive else None
        err_str = f"  (error: {err:+d} steps)" if err is not None else ""
        title_parts.append(f"predicted t={predicted_t}{err_str}")

    fig.suptitle("  |  ".join(title_parts),
                 fontsize=11, color="#eeeeee", y=1.002,
                 fontfamily="monospace")

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=null_color,        alpha=0.6, lw=1.2, label="Null recordings"),
        Line2D([0], [0], color=pos_color,          alpha=0.6, lw=1.2, label="Positive recordings"),
        mpatches.Patch(color=shade_color,          alpha=0.5,         label="Post-bifurcation region"),
        Line2D([0], [0], color=highlight_color,    alpha=0.9, lw=2.0, label=f"Highlighted: {rec_id}"),
        Line2D([0], [0], color=actual_line_color,  alpha=0.9, lw=1.5,
               ls="--", label="Actual bifurcation"),
    ]
    if predicted_t is not None:
        legend_elements.append(
            Line2D([0], [0], color=predicted_line_color, alpha=0.9, lw=1.5,
                   ls="--", label="Predicted bifurcation")
        )

    axes[0].legend(
        handles=legend_elements,
        loc="upper left",
        framealpha=0.2,
        facecolor="#1a1a1a",
        edgecolor="#444444",
        fontsize=8,
        labelcolor="#cccccc",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {save_path}")
    else:
        plt.show()

    return fig, axes


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper: pick the highlight recording automatically
# ──────────────────────────────────────────────────────────────────────────────

def plot_from_detector_result(
    recordings: list,
    streaming_alerts: list,             # list[BifurcationAlert] from the detector
    highlight_recording_id: str,
    window_len: int = 128,
    **plot_kwargs,
):
    """
    Convenience wrapper when you have streaming alerts and want to plot
    a specific recording with the detector's best prediction overlaid.

    The predicted_t is taken from the alert with the highest confidence.
    """
    rec_map = {r.recording_id: r for r in recordings}
    highlight = rec_map[highlight_recording_id]

    predicted_t = None
    if streaming_alerts:
        best = max(streaming_alerts, key=lambda a: a.p_bifurcation)
        predicted_t = best.predicted_bifurc_t

    return plot_recordings(
        recordings=recordings,
        highlight=highlight,
        predicted_t=predicted_t,
        **plot_kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
    # # ── Inline minimal Recording class so this file runs standalone ───────
    # from dataclasses import dataclass

    # @dataclass
    # class Recording:
    #     data: np.ndarray
    #     bifurcation_t: int | None
    #     recording_id: str

    #     @property
    #     def is_positive(self):
    #         return self.bifurcation_t is not None

    # # ── Generate synthetic data ───────────────────────────────────────────
    # rng = np.random.default_rng(42)
    # NUM_PARAMS, DEPTH, TIME = 4, 32, 512

    # def ou_process(T, D, P, rng, theta=0.1, sigma=0.3):
    #     x = np.zeros((P, D, T))
    #     for t in range(1, T):
    #         x[:, :, t] = (
    #             x[:, :, t-1]
    #             - theta * x[:, :, t-1]
    #             + sigma * rng.standard_normal((P, D))
    #         )
    #     return x

    # recordings = []

    # # Nulls
    # for i in range(30):
    #     data = ou_process(TIME, DEPTH, NUM_PARAMS, rng)
    #     recordings.append(Recording(data, None, f"null_{i:03d}"))

    # # Positives
    # for i in range(15):
    #     bf_t = int(rng.uniform(0.25 * TIME, 0.75 * TIME))
    #     data = ou_process(TIME, DEPTH, NUM_PARAMS, rng)
    #     for t in range(bf_t, TIME):
    #         scale = 1.0 + 3.0 * (t - bf_t) / (TIME - bf_t)
    #         data[:, :, t] *= scale
    #         data[:, :, t] += 0.4 * rng.standard_normal((NUM_PARAMS, DEPTH))
    #     recordings.append(Recording(data, bf_t, f"pos_{i:03d}"))

    # Pick highlight and simulate a predicted_t with a small error
    # highlight = next(r for r in recordings if r.is_positive)
    # simulated_predicted_t = highlight.bifurcation_t + int(rng.integers(-18, 18))

    # print(f"Highlight: {highlight.recording_id}")
    # print(f"  Actual bifurcation    : t={highlight.bifurcation_t}")
    # print(f"  Simulated prediction  : t={simulated_predicted_t}")
    # rec = pos_recs[0]
    # print(f"Highlight: {rec.recording_id}")
    # print(f"  Actual bifurcation    : t={rec.bifurcation_t}")
    # print(f"  Simulated prediction  : t={alert.predicted_bifurc_t}")

    # plot_recordings(
    #     recordings=test_recs,#recordings,
    #     highlight=rec,#highlight,
    #     predicted_t=alert.predicted_bifurc_t,
    #     depth_summary="mean",
    #     param_names=["Temperature", "Pressure", "Salinity", "Oxygen"],
    #     save_path="bifurcation_plot.png",
    # )
