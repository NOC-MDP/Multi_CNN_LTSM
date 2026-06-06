"""
Bifurcation Dataset Visualiser
===============================
Shows all recordings overlaid per parameter, with:
  • Null recordings         — thin dark lines, low opacity
  • Positive recordings     — thin lines + grey shaded region after bifurcation
  • Highlighted recording   — bold red line + per-parameter red dashed actual
                              bifurcation line (one per subplot)
  • Predicted bifurcation   — blue dashed vertical line, either a single shared
                              value (int) or one per parameter (list[int | float])

Usage
─────
    # Scalar predicted_t — single blue line across all subplots (legacy)
    plot_recordings(
        recordings    = all_recordings,
        highlight     = my_recording,
        predicted_t   = 312,
    )

    # Per-parameter predicted_t — one blue line per subplot
    # e.g. from per_var_timing returned by BifurcationRegressor.forward()
    plot_recordings(
        recordings    = all_recordings,
        highlight     = my_recording,
        predicted_t   = [289, 312, 301, 328],   # one per parameter
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Optional, Union

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_predicted_t(
    predicted_t: Optional[Union[int, float, list]],
    num_params: int,
) -> Optional[list]:
    """
    Normalise predicted_t to either None or a list of length num_params.

    Accepts:
      None                 → None (no predicted line drawn)
      int / float          → [value] * num_params  (same line on every subplot)
      list[int|float|None] → used as-is; must have length num_params
    """
    if predicted_t is None:
        return None
    if isinstance(predicted_t, (int, float)):
        return [predicted_t] * num_params
    if len(predicted_t) != num_params:
        raise ValueError(
            f"predicted_t has {len(predicted_t)} entries but the recording "
            f"has {num_params} parameters."
        )
    return list(predicted_t)


def _get_param_bifurcation_ts(rec, num_params: int) -> Optional[list]:
    """
    Return per-parameter bifurcation timesteps from a Recording, or None.

    Looks for `rec.param_bifurcation_ts` (list[int], length P).
    Falls back to replicating `rec.bifurcation_t` for all parameters so the
    code stays compatible with recordings that only carry the central time.
    """
    if hasattr(rec, "param_bifurcation_ts") and rec.param_bifurcation_ts is not None:
        return list(rec.param_bifurcation_ts)
    if rec.is_positive:
        return [rec.bifurcation_t] * num_params
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Core plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_recordings(
    recordings: list,                       # list[Recording]
    highlight: object,                      # Recording to draw in red
    predicted_t: Optional[Union[int, float, list]] = None,
    depth_summary: str = "mean",            # "mean" | "max" | "std" | int
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
    predicted_t     Predicted bifurcation timestep(s).
                      int / float  → same vertical line on every subplot
                      list         → one value per parameter, drawn on its own
                                     subplot only.  Pass None values in the
                                     list to skip individual subplots.
                      None         → no predicted line drawn anywhere.
    depth_summary   How to collapse the depth dimension before plotting.
    param_names     Optional list of axis labels, one per parameter.

    Recording contract
    ──────────────────
    The function reads two attributes from `highlight`:
      .bifurcation_t          int | None   — central / mean bifurcation time
      .param_bifurcation_ts   list[int] | None  — per-parameter times (length P)

    If `param_bifurcation_ts` is absent or None the code falls back to
    replicating `bifurcation_t` across all parameters, preserving backward
    compatibility with older Recording objects.
    """
    num_params  = highlight.data.shape[0]
    T_highlight = highlight.data.shape[2]

    if param_names is None:
        param_names = [f"Parameter {i+1}" for i in range(num_params)]

    # ── Resolve per-parameter timing lists ───────────────────────────────────
    pred_ts   = _resolve_predicted_t(predicted_t, num_params)   # list|None
    actual_ts = _get_param_bifurcation_ts(highlight, num_params) # list|None

    # ── Figure / axes setup ──────────────────────────────────────────────────
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
        d = data[param_idx]     # (depth, time)
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

    # ── Pass 1: nulls ─────────────────────────────────────────────────────────
    for rec in recordings:
        if rec.is_positive or rec is highlight:
            continue
        T = rec.data.shape[2]
        t = np.arange(T)
        for p_idx, ax in enumerate(axes):
            ax.plot(t, summarise(rec.data, p_idx),
                    color=null_color, alpha=null_alpha,
                    lw=null_lw, rasterized=True)

    # ── Pass 2: positives (not highlight) — line + shaded post-bifurc ─────────
    for rec in recordings:
        if not rec.is_positive or rec is highlight:
            continue
        T   = rec.data.shape[2]
        t   = np.arange(T)
        # Use per-parameter times if available, else fall back to central time
        rec_param_ts = _get_param_bifurcation_ts(rec, num_params)
        for p_idx, ax in enumerate(axes):
            ax.plot(t, summarise(rec.data, p_idx),
                    color=pos_color, alpha=pos_alpha,
                    lw=pos_lw, rasterized=True)
            if rec_param_ts is not None:
                bf = rec_param_ts[p_idx]
                ax.axvspan(bf, T - 1,
                           color=shade_color, alpha=shade_alpha,
                           zorder=0, lw=0)

    # ── Pass 3: highlighted recording ─────────────────────────────────────────
    t_h = np.arange(T_highlight)

    for p_idx, ax in enumerate(axes):
        signal = summarise(highlight.data, p_idx)

        # Per-parameter post-bifurcation shading
        if highlight.is_positive and actual_ts is not None:
            bf_p = actual_ts[p_idx]
            ax.axvspan(bf_p, T_highlight - 1,
                       color=highlight_color, alpha=0.07, zorder=1, lw=0)

        # Main highlighted signal
        ax.plot(t_h, signal,
                color=highlight_color, lw=highlight_lw,
                alpha=0.95, zorder=4, rasterized=True)

        # ── Per-parameter actual bifurcation line ─────────────────────────
        if highlight.is_positive and actual_ts is not None:
            bf_p = actual_ts[p_idx]
            ax.axvline(
                bf_p,
                color=actual_line_color, lw=1.5, ls="--",
                alpha=0.9, zorder=5,
                # Only add the label on the first subplot to keep the legend clean
                label="Actual bifurcation" if p_idx == 0 else None,
            )
            # Small annotation showing the exact timestep
            ax.text(
                bf_p + T_highlight * 0.005,
                ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 0.95,
                f"t={bf_p}",
                color=actual_line_color, fontsize=7, alpha=0.8,
                va="top", transform=ax.get_xaxis_transform(),
                zorder=6,
            )

        # ── Per-parameter predicted bifurcation line ──────────────────────
        if pred_ts is not None and pred_ts[p_idx] is not None:
            pt = pred_ts[p_idx]
            ax.axvline(
                pt,
                color=predicted_line_color, lw=1.5, ls="--",
                alpha=0.9, zorder=5,
                label="Predicted bifurcation" if p_idx == 0 else None,
            )
            ax.text(
                pt + T_highlight * 0.005,
                ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 0.85,
                f"t={int(pt)}",
                color=predicted_line_color, fontsize=7, alpha=0.8,
                va="top", transform=ax.get_xaxis_transform(),
                zorder=6,
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

    # ── Title ─────────────────────────────────────────────────────────────────
    rec_id = getattr(highlight, "recording_id", "highlighted")
    title_parts = [f"Recording: {rec_id}"]

    if highlight.is_positive and actual_ts is not None:
        ts_str = ", ".join(str(t) for t in actual_ts)
        title_parts.append(f"actual bf_ts=[{ts_str}]")

    if pred_ts is not None:
        pt_str = ", ".join(
            str(int(v)) if v is not None else "—" for v in pred_ts
        )
        title_parts.append(f"predicted=[{pt_str}]")

        # Per-parameter errors when both are available
        if highlight.is_positive and actual_ts is not None:
            errors = [
                int(p - a) if (p is not None) else None
                for p, a in zip(pred_ts, actual_ts)
            ]
            err_str = ", ".join(
                f"{e:+d}" if e is not None else "—" for e in errors
            )
            title_parts.append(f"errors=[{err_str}]")

    fig.suptitle(
        "  |  ".join(title_parts),
        fontsize=10, color="#eeeeee", y=1.002,
        fontfamily="monospace",
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=null_color,     alpha=0.6, lw=1.2, label="Null recordings"),
        Line2D([0], [0], color=pos_color,       alpha=0.6, lw=1.2, label="Positive recordings"),
        mpatches.Patch(color=shade_color,       alpha=0.5,         label="Post-bifurcation region"),
        Line2D([0], [0], color=highlight_color, alpha=0.9, lw=2.0, label=f"Highlighted: {rec_id}"),
        Line2D([0], [0], color=actual_line_color, alpha=0.9, lw=1.5,
               ls="--", label="Actual bifurcation (per parameter)"),
    ]
    if pred_ts is not None:
        pred_label = (
            "Predicted bifurcation (per parameter)"
            if len(set(v for v in pred_ts if v is not None)) > 1
            else "Predicted bifurcation"
        )
        legend_elements.append(
            Line2D([0], [0], color=predicted_line_color, alpha=0.9, lw=1.5,
                   ls="--", label=pred_label)
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
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def plot_from_detector_result(
    recordings: list,
    streaming_alerts: list,             # list[BifurcationAlert]
    highlight_recording_id: str,
    window_len: int = 128,
    use_per_var_timing: bool = True,    # use alert.per_var_timing if present
    **plot_kwargs,
):
    """
    Convenience wrapper when you have streaming alerts and want to plot
    a specific recording with the detector's best prediction overlaid.

    If `use_per_var_timing` is True and the best alert carries a
    `per_var_timing` attribute (from the updated BifurcationRegressor),
    those are passed as a per-parameter list to predicted_t so each
    subplot gets its own predicted line.

    Otherwise falls back to a single scalar predicted_bifurc_t.
    """
    rec_map   = {r.recording_id: r for r in recordings}
    highlight = rec_map[highlight_recording_id]

    predicted_t = None
    if streaming_alerts:
        best = max(streaming_alerts, key=lambda a: a.p_bifurcation)
        if (
            use_per_var_timing
            and hasattr(best, "per_var_timing")
            and best.per_var_timing is not None
        ):
            # per_var_timing is normalised [0, 1]; scale to absolute timestep
            T  = highlight.data.shape[2]
            predicted_t = [v * T for v in best.per_var_timing]
        else:
            predicted_t = best.predicted_bifurc_t

    return plot_recordings(
        recordings=recordings,
        highlight=highlight,
        predicted_t=predicted_t,
        **plot_kwargs,
    )