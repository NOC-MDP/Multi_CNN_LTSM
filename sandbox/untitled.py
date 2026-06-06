"""
synthetic_data_gen.py
─────────────────────
Production-quality synthetic dataset generator for the multi-parameter CNN-LSTM
bifurcation detector.

Key improvements over the demo version
───────────────────────────────────────
1.  Signal diversity        – OU parameters (θ, σ) are sampled per recording so
                              the model cannot memorise a single noise regime.
2.  Multiple bifurcation    – four distinct dynamical signatures are included:
    types                     • VARIANCE   – critical-slowing-down variance inflation
                              • MEAN_SHIFT – step-change in process mean
                              • HOPF       – growing sinusoidal oscillation onset
                              • CASCADE    – two sequential bifurcation events
3.  Realistic noise floor   – per-channel i.i.d. sensor noise + rare spike artefacts.
4.  Stratified splits       – every Recording carries a 'split' field
                              ('train' / 'val' / 'test') drawn deterministically
                              from the seed so experiments are reproducible.
5.  Severity score          – positive recordings carry a continuous 'severity'
                              in [0, 1] alongside the binary label, useful for
                              soft-target or auxiliary regression training.
6.  Data integrity          – NaN / Inf values are detected and cause a hard error
                              rather than silently polluting the dataset.
7.  Convenience helpers     – split_dataset() and dataset_summary() utilities.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
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
    param_bifurcation_ts: list[int] | None = field(default=None)

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




# ─────────────────────────────────────────────────────────────────────────────
# Bifurcation-type taxonomy
# ─────────────────────────────────────────────────────────────────────────────

class BifurcationType(Enum):
    VARIANCE   = auto()   # critical-slowing-down variance inflation (original)
    MEAN_SHIFT = auto()   # abrupt mean displacement
    HOPF       = auto()   # onset of growing oscillation (Hopf-like)
    CASCADE    = auto()   # two sequential events at different time points


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    # ── Dataset composition ──────────────────────────────────────────────────
    n_null:       int = 400     # total null recordings
    n_positive:   int = 400     # total positive recordings (split evenly across types)

    # ── Signal dimensions ────────────────────────────────────────────────────
    num_params:   int = 4       # P – number of physiological / sensor parameters
    depth:        int = 32      # D – spatial / channel depth
    time_len:     int = 512     # T – time steps per recording

    # ── OU baseline diversity ────────────────────────────────────────────────
    theta_range:  tuple[float, float] = (0.05, 0.25)   # mean-reversion speed
    sigma_range:  tuple[float, float] = (0.25, 0.60)   # baseline volatility

    # ── Bifurcation geometry ─────────────────────────────────────────────────
    bf_center_range: tuple[float, float] = (0.45, 0.80)  # fraction of T
    bf_jitter_frac:  float = 0.05          # ±jitter as fraction of T

    # ── Noise & artefacts ────────────────────────────────────────────────────
    sensor_noise_std:  float = 0.10        # per-channel additive Gaussian noise
    spike_prob:        float = 0.004       # probability of a spike at any (p,d,t)
    spike_magnitude:   float = 5.0        # spike height in σ units

    # ── Splits ───────────────────────────────────────────────────────────────
    train_frac: float = 0.70
    val_frac:   float = 0.15
    # test_frac is inferred as 1 - train - val

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# def _ou_process(
#     T: int,
#     D: int,
#     P: int,
#     rng: np.random.Generator,
#     theta: float,
#     sigma: float,
# ) -> np.ndarray:
#     """Ornstein-Uhlenbeck baseline: shape (P, D, T)."""
#     x = np.zeros((P, D, T))
#     for t in range(1, T):
#         x[:, :, t] = (
#             x[:, :, t - 1]
#             - theta * x[:, :, t - 1]
#             + sigma * rng.standard_normal((P, D))
#         )
#     return x


# def _add_sensor_noise(
#     data: np.ndarray,
#     rng: np.random.Generator,
#     noise_std: float,
#     spike_prob: float,
#     spike_magnitude: float,
# ) -> np.ndarray:
#     """Additive Gaussian noise + rare spike artefacts, in-place."""
#     data += noise_std * rng.standard_normal(data.shape)

#     if spike_prob > 0:
#         spike_mask = rng.random(data.shape) < spike_prob
#         spike_signs = rng.choice([-1, 1], size=data.shape)
#         data += spike_mask * spike_signs * spike_magnitude

#     return data

"""
ocn_signal_gen.py
─────────────────
Oceanographic baseline signal generator — drop-in replacement for the
generic _ou_process / _add_sensor_noise layer in synthetic_data_gen.py.

Design philosophy
─────────────────
Physical model output (NEMO, ROMS, MOM6, etc.) is the current data source.
The generator is structured in three tiers so it can grow toward biogeochemistry
and observations without a rewrite:

  Tier 1 – Physical variables (implemented)
      Temperature, salinity, pressure/depth, horizontal velocity (u, v),
      vertical velocity (w), sea surface height (SSH), mixed layer depth (MLD).

  Tier 2 – Biogeochemical variables (stubs — structure in place, params TBD)
      Dissolved oxygen, nitrate, chlorophyll-a, pCO2, pH.

  Tier 3 – Observational degradation (implemented as optional layer)
      Instrument noise, drift, gaps, biofouling drift, mooring motion.

Each variable has a VariableProfile that encodes:
  • Baseline dynamics  (OU mean-reversion + background variance)
  • Dominant periodicities  (diurnal, tidal, seasonal)
  • Distribution shape  (Gaussian, log-normal, bounded)
  • Cross-variable correlation structure
  • Observational noise characteristics

Usage
─────
Replace the two calls in make_synthetic_recordings:

    # OLD
    data = _ou_process(T, D, P, rng, theta, sigma)
    _add_sensor_noise(data, rng, cfg.sensor_noise_std, cfg.spike_prob, cfg.spike_magnitude)

    # NEW
    generator = OceanographicSignalGenerator(cfg, rng)
    data = generator.generate_baseline(T, D)   # returns (P, D, T)
    generator.add_observational_noise(data)    # in-place
"""



# ─────────────────────────────────────────────────────────────────────────────
# Variable taxonomy
# ─────────────────────────────────────────────────────────────────────────────

class OceanVar(Enum):
    # Tier 1 — physical (fully implemented)
    TEMPERATURE     = auto()   # potential temperature anomaly (°C)
    SALINITY        = auto()   # practical salinity anomaly (PSU)
    PRESSURE        = auto()   # pressure / depth proxy (dbar)
    VELOCITY_U      = auto()   # eastward current anomaly (m/s)
    VELOCITY_V      = auto()   # northward current anomaly (m/s)
    VELOCITY_W      = auto()   # vertical velocity (m/day)
    SSH             = auto()   # sea surface height anomaly (m)
    MLD             = auto()   # mixed layer depth anomaly (m)

    # Tier 2 — biogeochemical (stubs — parameters TBD from real data)
    OXYGEN          = auto()   # dissolved oxygen anomaly (mmol/m³)
    NITRATE         = auto()   # nitrate anomaly (mmol/m³)
    CHLOROPHYLL     = auto()   # chlorophyll-a (mg/m³) — log-normal
    PCO2            = auto()   # partial pressure CO₂ anomaly (µatm)
    PH              = auto()   # pH anomaly


# ─────────────────────────────────────────────────────────────────────────────
# Per-variable physical parameter profiles
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VariableProfile:
    """
    All parameters that govern the baseline signal for one oceanographic
    variable.  Values are for *anomalies* (deviations from the mean state)
    so the generator is mean-free by default — consistent with how physical
    model output is typically pre-processed before ML ingestion.

    Periodicity amplitudes are given as fractions of sigma so they scale
    automatically when sigma is varied across recordings.
    """
    # ── Baseline OU dynamics ──────────────────────────────────────────────────
    theta_range:   tuple[float, float]  # mean-reversion timescale (1/timestep)
    sigma_range:   tuple[float, float]  # baseline innovation std

    # ── Dominant periodicities ────────────────────────────────────────────────
    # Each entry: (period_in_timesteps, amplitude_as_fraction_of_sigma, phase_randomise)
    periodicities: list[tuple[float, float]] = field(default_factory=list)

    # ── Distribution shape ────────────────────────────────────────────────────
    log_normal:    bool  = False   # if True, exponentiate after OU (chlorophyll)
    lower_bound:   Optional[float] = None   # hard lower bound (e.g. O2 >= 0)
    upper_bound:   Optional[float] = None   # hard upper bound (e.g. pH <= 9)

    # ── Slow non-stationarity ─────────────────────────────────────────────────
    # Random-walk drift superimposed on the OU — simulates model drift /
    # slowly-evolving background state
    drift_sigma:   float = 0.0     # std of the drift innovation per timestep

    # ── Observational noise (Tier 3) ─────────────────────────────────────────
    sensor_noise_std:  float = 0.01   # white Gaussian noise
    drift_rate:        float = 0.0    # sensor drift per timestep (fraction of sigma)
    burst_prob:        float = 0.001  # probability of a correlated burst onset
    burst_decay:       float = 0.90   # AR(1) decay of burst amplitude
    burst_magnitude:   float = 3.0    # burst peak in units of sigma
    gap_prob:          float = 0.0    # probability of a missing-data gap start
    gap_length_range:  tuple[int, int] = (2, 10)  # gap duration in timesteps

    # ── Tier 2 flag ──────────────────────────────────────────────────────────
    is_stub:       bool  = False   # True = biogeochem stub, params not validated


# ─────────────────────────────────────────────────────────────────────────────
# Variable profile registry
# ─────────────────────────────────────────────────────────────────────────────
# Timestep convention: 1 timestep = 1 day (physical model daily output).
# Adjust periodicity periods if your model output is at a different frequency.
# Diurnal cycle (period=1) is absent from daily output by construction.

_PROFILES: dict[OceanVar, VariableProfile] = {

    # ── Temperature anomaly ───────────────────────────────────────────────────
    # Physical model SST/subsurface T: decorrelation ~10-30 days, strong
    # seasonal cycle, weak tidal signal at depth.
    OceanVar.TEMPERATURE: VariableProfile(
        theta_range   = (0.03, 0.10),   # ~10–33 day decorrelation
        sigma_range   = (0.20, 0.60),   # °C anomaly std
        periodicities = [
            (365.0, 0.8),   # seasonal — dominant
            (14.0,  0.05),  # spring-neap tidal beating
        ],
        drift_sigma        = 0.002,
        sensor_noise_std   = 0.01,
        burst_prob         = 0.002,
        burst_decay        = 0.88,
        burst_magnitude    = 2.5,
    ),

    # ── Salinity anomaly ──────────────────────────────────────────────────────
    # Slower mean reversion than T (haline restoring is weaker than thermal),
    # seasonal cycle from precipitation/evaporation and river runoff.
    OceanVar.SALINITY: VariableProfile(
        theta_range   = (0.02, 0.07),   # ~14–50 day decorrelation
        sigma_range   = (0.05, 0.20),   # PSU anomaly std
        periodicities = [
            (365.0, 0.6),
            (30.0,  0.08),  # monthly precipitation cycle
        ],
        drift_sigma        = 0.001,
        sensor_noise_std   = 0.005,
        burst_prob         = 0.001,
        burst_decay        = 0.92,
        burst_magnitude    = 2.0,
    ),

    # ── Pressure / depth proxy ────────────────────────────────────────────────
    # At a mooring depth is nearly constant; anomalies come from internal waves
    # and isopycnal heaving.  Short decorrelation, no strong seasonal cycle.
    OceanVar.PRESSURE: VariableProfile(
        theta_range   = (0.15, 0.40),   # ~2.5–7 day decorrelation
        sigma_range   = (0.10, 0.30),
        periodicities = [
            (1.0,  0.20),   # diurnal internal wave (if sub-daily output)
            (0.5,  0.10),   # semi-diurnal tide
        ],
        sensor_noise_std   = 0.002,
        burst_prob         = 0.003,
        burst_decay        = 0.80,
        burst_magnitude    = 3.0,
    ),

    # ── Eastward velocity anomaly (u) ─────────────────────────────────────────
    # Energetic on synoptic scales, strong tidal and near-inertial signals.
    OceanVar.VELOCITY_U: VariableProfile(
        theta_range   = (0.05, 0.20),
        sigma_range   = (0.02, 0.15),   # m/s
        periodicities = [
            (1.0,  0.30),   # diurnal
            (0.5,  0.40),   # semi-diurnal M2 tide — dominant for velocity
            (3.5,  0.15),   # near-inertial (~3.5 days at mid-lat)
        ],
        sensor_noise_std   = 0.005,
        burst_prob         = 0.003,
        burst_decay        = 0.85,
        burst_magnitude    = 3.5,
    ),

    # ── Northward velocity anomaly (v) ────────────────────────────────────────
    # Similar to u but typically weaker at mid-latitudes.
    OceanVar.VELOCITY_V: VariableProfile(
        theta_range   = (0.05, 0.20),
        sigma_range   = (0.02, 0.12),
        periodicities = [
            (1.0,  0.25),
            (0.5,  0.35),
            (3.5,  0.15),
        ],
        sensor_noise_std   = 0.005,
        burst_prob         = 0.003,
        burst_decay        = 0.85,
        burst_magnitude    = 3.5,
    ),

    # ── Vertical velocity (w) ─────────────────────────────────────────────────
    # Very small magnitude, strongly skewed (upwelling events are intermittent).
    # Heavier-tailed than Gaussian — modelled as OU + occasional bursts.
    OceanVar.VELOCITY_W: VariableProfile(
        theta_range   = (0.20, 0.50),   # fast decorrelation
        sigma_range   = (0.002, 0.01),  # m/day
        periodicities = [],
        burst_prob         = 0.005,     # upwelling/downwelling events
        burst_decay        = 0.75,
        burst_magnitude    = 5.0,       # heavy tail
        sensor_noise_std   = 0.001,
    ),

    # ── Sea surface height anomaly ────────────────────────────────────────────
    # Low-frequency, red spectrum. Seasonal + mesoscale eddy signal.
    OceanVar.SSH: VariableProfile(
        theta_range   = (0.01, 0.04),   # ~25–100 day decorrelation (eddies)
        sigma_range   = (0.05, 0.25),   # m
        periodicities = [
            (365.0, 0.50),
            (90.0,  0.20),  # inter-seasonal
        ],
        drift_sigma        = 0.003,
        sensor_noise_std   = 0.02,      # altimeter noise
        burst_prob         = 0.001,
        burst_decay        = 0.95,
        burst_magnitude    = 2.0,
    ),

    # ── Mixed layer depth anomaly ─────────────────────────────────────────────
    # Strongly seasonal, highly asymmetric (deep winter, shallow summer).
    # Log-normal after seasonal cycle to keep MLD positive.
    OceanVar.MLD: VariableProfile(
        theta_range   = (0.02, 0.06),
        sigma_range   = (0.15, 0.40),
        periodicities = [
            (365.0, 1.20),  # very strong seasonal — MLD defined by it
            (30.0,  0.10),
        ],
        log_normal         = True,      # MLD is positive-definite
        drift_sigma        = 0.002,
        sensor_noise_std   = 0.03,
    ),

    # ── Dissolved oxygen (Tier 2 stub) ────────────────────────────────────────
    OceanVar.OXYGEN: VariableProfile(
        theta_range   = (0.03, 0.10),
        sigma_range   = (0.10, 0.40),
        periodicities = [(365.0, 0.60)],
        lower_bound        = -5.0,      # anomaly bounded below by climatological mean
        sensor_noise_std   = 0.02,
        is_stub            = True,
    ),

    # ── Nitrate (Tier 2 stub) ─────────────────────────────────────────────────
    OceanVar.NITRATE: VariableProfile(
        theta_range   = (0.02, 0.08),
        sigma_range   = (0.10, 0.50),
        periodicities = [(365.0, 0.80), (90.0, 0.20)],
        lower_bound        = -2.0,
        sensor_noise_std   = 0.05,
        is_stub            = True,
    ),

    # ── Chlorophyll-a (Tier 2 stub) ───────────────────────────────────────────
    # Log-normal: bloom dynamics produce heavy positive tails.
    OceanVar.CHLOROPHYLL: VariableProfile(
        theta_range   = (0.05, 0.20),
        sigma_range   = (0.20, 0.60),
        periodicities = [(365.0, 0.90), (30.0, 0.15)],
        log_normal         = True,
        sensor_noise_std   = 0.05,
        burst_prob         = 0.003,     # bloom onset
        burst_decay        = 0.90,
        burst_magnitude    = 4.0,
        is_stub            = True,
    ),

    # ── pCO2 (Tier 2 stub) ────────────────────────────────────────────────────
    OceanVar.PCO2: VariableProfile(
        theta_range   = (0.03, 0.10),
        sigma_range   = (0.10, 0.40),
        periodicities = [(365.0, 0.70)],
        sensor_noise_std   = 0.03,
        is_stub            = True,
    ),

    # ── pH (Tier 2 stub) ──────────────────────────────────────────────────────
    OceanVar.PH: VariableProfile(
        theta_range   = (0.03, 0.10),
        sigma_range   = (0.05, 0.20),
        periodicities = [(365.0, 0.60)],
        upper_bound        = 2.0,       # anomaly bounded
        lower_bound        = -2.0,
        sensor_noise_std   = 0.01,
        is_stub            = True,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-variable correlation structure
# ─────────────────────────────────────────────────────────────────────────────
# Physical couplings encoded as a correlation matrix over the Tier 1 variables
# in OceanVar enum order.  Off-diagonal values reflect well-established
# physical relationships; sign and magnitude are approximate.
#
# Key relationships:
#   T–S:  negative in many regions (warm/salty Mediterranean water vs
#         cold/fresh Arctic water); set mildly negative as a default.
#   T–MLD: negative — deep mixing cools the surface layer.
#   u–v:  weak positive (both respond to the same wind forcing).
#   SSH–T: positive — warm eddies are associated with high SSH.
#   SSH–MLD: negative — high SSH (anticyclonic eddies) → deep MLD.

_PHYSICAL_CORR = np.array([
    #  T     S     P     u     v     w    SSH   MLD
    [1.00, -0.30, 0.05, 0.05, 0.05,-0.10, 0.40,-0.50],  # T
    [-0.30, 1.00, 0.00, 0.00, 0.00, 0.05,-0.20, 0.15],  # S
    [0.05,  0.00, 1.00, 0.10, 0.10, 0.20, 0.05,-0.10],  # P
    [0.05,  0.00, 0.10, 1.00, 0.30,-0.05, 0.10, 0.00],  # u
    [0.05,  0.00, 0.10, 0.30, 1.00,-0.05, 0.10, 0.00],  # v
    [-0.10, 0.05, 0.20,-0.05,-0.05, 1.00,-0.10,-0.15],  # w
    [0.40, -0.20, 0.05, 0.10, 0.10,-0.10, 1.00,-0.40],  # SSH
    [-0.50, 0.15,-0.10, 0.00, 0.00,-0.15,-0.40, 1.00],  # MLD
], dtype=np.float64)


def _nearest_spd(A: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest symmetric positive-definite."""
    A = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-6)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ─────────────────────────────────────────────────────────────────────────────
# Core signal generator
# ─────────────────────────────────────────────────────────────────────────────

class OceanographicSignalGenerator:
    """
    Generates physically-motivated baseline signals for P oceanographic
    variables, each represented by D depth/channel realisations over T timesteps.

    Parameters
    ----------
    cfg          : DatasetConfig — uses num_params, depth, time_len, seed,
                   and the new variable_set field.
    rng          : np.random.Generator — shared RNG from the outer generation loop.
    variable_set : list[OceanVar] of length cfg.num_params.  If None, defaults
                   to the first cfg.num_params Tier 1 physical variables.

    The generator exposes two public methods:
        generate_baseline(T, D)     → np.ndarray (P, D, T)
        add_observational_noise(x)  → in-place
    """

    # Default variable ordering for physical-only runs
    _DEFAULT_PHYSICAL = [
        OceanVar.TEMPERATURE,
        OceanVar.SALINITY,
        OceanVar.VELOCITY_U,
        OceanVar.VELOCITY_V,
        OceanVar.SSH,
        OceanVar.MLD,
        OceanVar.PRESSURE,
        OceanVar.VELOCITY_W,
    ]

    def __init__(
        self,
        cfg,                              # DatasetConfig
        rng: np.random.Generator,
        variable_set: list[OceanVar] | None = None,
    ):
        self.rng = rng
        self.P   = cfg.num_params
        self.T   = cfg.time_len
        self.D   = cfg.depth

        if variable_set is not None:
            if len(variable_set) != self.P:
                raise ValueError(
                    f"variable_set has {len(variable_set)} entries but "
                    f"cfg.num_params={self.P}"
                )
            self.variables = variable_set
        else:
            self.variables = self._DEFAULT_PHYSICAL[: self.P]

        self.profiles = [_PROFILES[v] for v in self.variables]

        # Build the cross-variable Cholesky factor for the physical variables
        # (biogeochem stubs use identity — no validated cross-correlations yet)
        phys_vars   = [v for v in OceanVar
                       if v.value <= OceanVar.MLD.value]   # Tier 1 only
        var_indices = []
        for v in self.variables:
            if v in phys_vars:
                var_indices.append(phys_vars.index(v))
            else:
                var_indices.append(None)   # stub — no cross-correlation

        # Sub-matrix of _PHYSICAL_CORR for the selected physical variables
        phys_idx = [i for i, idx in enumerate(var_indices) if idx is not None]
        corr_idx = [var_indices[i] for i in phys_idx]

        self._corr_L      = None   # Cholesky factor (P, P) or None if all stubs
        self._phys_idx    = phys_idx
        self._full_P      = self.P

        if len(phys_idx) >= 2:
            sub_corr = _PHYSICAL_CORR[np.ix_(corr_idx, corr_idx)]
            sub_corr = _nearest_spd(sub_corr)
            self._corr_L    = np.linalg.cholesky(sub_corr)   # (n_phys, n_phys)

        # Per-recording parameters sampled fresh each call to generate_baseline
        self._sampled_theta  : list[float] = []
        self._sampled_sigma  : list[float] = []

    # ─────────────────────────────────────────────────────────────────────────

    def generate_baseline(self, T: int | None = None, D: int | None = None) -> np.ndarray:
        """
        Generate a (P, D, T) baseline array with:
          • Per-variable OU dynamics with appropriate timescales
          • Dominant periodicities (seasonal, tidal, near-inertial)
          • Slow background drift
          • Cross-variable physical correlations (Tier 1 variables)
          • Log-normal transform where appropriate (MLD, chlorophyll)
          • Hard bounds applied last (O2 >= 0, etc.)
        """
        T = T or self.T
        D = D or self.D
        rng = self.rng

        self._sampled_theta = []
        self._sampled_sigma = []

        t_axis = np.arange(T, dtype=np.float64)

        # ── Step 1: independent OU per variable, single channel ──────────────
        ou_signals = np.zeros((self.P, T))   # (P, T) — correlated later

        for p, prof in enumerate(self.profiles):
            theta = float(rng.uniform(*prof.theta_range))
            sigma = float(rng.uniform(*prof.sigma_range))
            self._sampled_theta.append(theta)
            self._sampled_sigma.append(sigma)

            x = np.zeros(T)
            for t in range(1, T):
                x[t] = x[t-1] - theta * x[t-1] + sigma * rng.standard_normal()

            # Slow drift superimposed on the OU
            if prof.drift_sigma > 0:
                drift = np.cumsum(
                    prof.drift_sigma * rng.standard_normal(T)
                )
                # High-pass the drift so it doesn't dominate — keep only the
                # very low-frequency component
                from numpy.fft import rfft, irfft
                F = rfft(drift)
                cutoff = max(1, T // 50)   # keep only the lowest 2% of freqs
                F[cutoff:] = 0
                drift = irfft(F, n=T)
                x += drift

            ou_signals[p] = x

        # ── Step 2: apply cross-variable correlations (Tier 1 only) ─────────
        if self._corr_L is not None and len(self._phys_idx) >= 2:
            # Standardise the physical sub-signals, mix, then re-scale
            pi = self._phys_idx
            sub = ou_signals[pi]                            # (n_phys, T)
            stds = sub.std(axis=1, keepdims=True).clip(1e-8)
            sub_norm = sub / stds                           # unit variance
            sub_corr = self._corr_L @ sub_norm              # (n_phys, T)
            ou_signals[pi] = sub_corr * stds               # restore scale

        # ── Step 3: add periodicities ────────────────────────────────────────
        for p, prof in enumerate(self.profiles):
            sigma = self._sampled_sigma[p]
            for period, amp_frac in prof.periodicities:
                phase = rng.uniform(0, 2 * np.pi)
                amplitude = amp_frac * sigma
                ou_signals[p] += amplitude * np.sin(
                    2 * np.pi * t_axis / period + phase
                )

        # ── Step 4: log-normal transform where appropriate ───────────────────
        for p, prof in enumerate(self.profiles):
            if prof.log_normal:
                # exp(x) - 1 centres the transform near zero for small x
                ou_signals[p] = np.expm1(
                    np.clip(ou_signals[p], -4, 4)
                )

        # ── Step 5: apply hard bounds ────────────────────────────────────────
        for p, prof in enumerate(self.profiles):
            if prof.lower_bound is not None:
                ou_signals[p] = np.maximum(ou_signals[p], prof.lower_bound)
            if prof.upper_bound is not None:
                ou_signals[p] = np.minimum(ou_signals[p], prof.upper_bound)

        # ── Step 6: broadcast to D depth channels with small spatial noise ───
        # Depth channels represent spatial realisations (e.g. ensemble members,
        # neighbouring grid cells, or depth levels).  Each channel gets a tiny
        # independent OU perturbation to avoid being purely identical.
        data = np.zeros((self.P, D, T))
        for p in range(self.P):
            sigma_p = self._sampled_sigma[p]
            for d in range(D):
                # Small spatial perturbation — 10% of signal variance
                spatial_noise = 0.10 * sigma_p * rng.standard_normal(T)
                data[p, d] = ou_signals[p] + spatial_noise

        return data

    # ─────────────────────────────────────────────────────────────────────────

    def add_observational_noise(self, data: np.ndarray) -> np.ndarray:
        """
        Add instrument-realistic noise to a (P, D, T) array in-place.

        Tier 3 effects applied per variable per channel:
          1. White Gaussian sensor noise
          2. Slow sensor drift (random-walk, very low amplitude)
          3. Correlated burst events (storms, eddies passing the mooring)
          4. Data gaps filled with NaN (optional — off by default)

        Returns the modified array (same object).
        """
        rng  = self.rng
        P, D, T = data.shape

        for p, prof in enumerate(self.profiles):
            sigma_p = self._sampled_sigma[p] if self._sampled_sigma else 0.3

            for d in range(D):

                # 1. White sensor noise
                data[p, d] += prof.sensor_noise_std * rng.standard_normal(T)

                # 2. Slow sensor drift — independent per channel (biofouling,
                #    calibration drift)
                if prof.drift_rate > 0:
                    drift = np.cumsum(
                        prof.drift_rate * sigma_p * rng.standard_normal(T)
                    )
                    data[p, d] += drift

                # 3. Correlated burst noise — AR(1) process that occasionally
                #    activates and decays.  Represents storm passages, eddy
                #    encounters, internal wave packets.
                if prof.burst_prob > 0:
                    burst = 0.0
                    for t in range(T):
                        if rng.random() < prof.burst_prob:
                            burst = (
                                rng.uniform(1.0, prof.burst_magnitude)
                                * sigma_p
                                * rng.choice([-1, 1])
                            )
                        burst  *= prof.burst_decay
                        data[p, d, t] += burst

                # 4. Data gaps (NaN) — disabled by default; enable for
                #    observational realism in Tier 3 runs
                if prof.gap_prob > 0:
                    t = 0
                    while t < T:
                        if rng.random() < prof.gap_prob:
                            gap_len = int(rng.integers(*prof.gap_length_range))
                            data[p, d, t: t + gap_len] = np.nan
                            t += gap_len
                        else:
                            t += 1

        return data

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def sampled_params(self) -> dict:
        """Returns the per-variable OU params from the last generate_baseline call."""
        return {
            str(v): {"theta": self._sampled_theta[i], "sigma": self._sampled_sigma[i]}
            for i, v in enumerate(self.variables)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: default physical variable set for common num_params values
# ─────────────────────────────────────────────────────────────────────────────

def default_variable_set(num_params: int) -> list[OceanVar]:
    """
    Returns the recommended default variable set for a given P.

    P=2 : T, S              (minimal thermohaline)
    P=3 : T, S, SSH
    P=4 : T, S, u, SSH      (adds horizontal dynamics)
    P=5 : T, S, u, v, SSH
    P=6 : T, S, u, v, SSH, MLD
    P=7 : T, S, u, v, w, SSH, MLD
    P=8 : all Tier 1
    """
    defaults = {
        2: [OceanVar.TEMPERATURE, OceanVar.SALINITY],
        3: [OceanVar.TEMPERATURE, OceanVar.SALINITY, OceanVar.SSH],
        4: [OceanVar.TEMPERATURE, OceanVar.SALINITY,
            OceanVar.VELOCITY_U, OceanVar.SSH],
        5: [OceanVar.TEMPERATURE, OceanVar.SALINITY,
            OceanVar.VELOCITY_U, OceanVar.VELOCITY_V, OceanVar.SSH],
        6: [OceanVar.TEMPERATURE, OceanVar.SALINITY,
            OceanVar.VELOCITY_U, OceanVar.VELOCITY_V,
            OceanVar.SSH, OceanVar.MLD],
        7: [OceanVar.TEMPERATURE, OceanVar.SALINITY,
            OceanVar.VELOCITY_U, OceanVar.VELOCITY_V, OceanVar.VELOCITY_W,
            OceanVar.SSH, OceanVar.MLD],
        8: list(OceanVar)[:8],
    }
    if num_params not in defaults:
        raise ValueError(
            f"No default variable set for num_params={num_params}. "
            f"Supported: {list(defaults)}. Pass variable_set explicitly."
        )
    return defaults[num_params]



def _sample_bf_times(
    rng: np.random.Generator,
    cfg: DatasetConfig,
    num_events: int = 1,
) -> tuple[int, np.ndarray]:
    """
    Returns (center_bf_t, per_param_bf_ts).
    center_bf_t  – integer, the 'canonical' bifurcation time stored in Recording
    per_param_ts – shape (num_events, P), one row per cascade event
    """
    T   = cfg.time_len
    P   = cfg.num_params
    lo, hi = cfg.bf_center_range
    hw  = int(cfg.bf_jitter_frac * T)

    centers = np.sort(
        rng.integers(int(lo * T), int(hi * T), size=num_events)
    )
    # ensure cascade events are at least 10% apart
    if num_events == 2:
        gap = int(0.10 * T)
        if centers[1] - centers[0] < gap:
            centers[1] = min(centers[0] + gap, int(hi * T))

    per_param = np.clip(
        centers[:, np.newaxis]
        + rng.integers(-hw, hw + 1, size=(num_events, P)),
        1,
        T - 2,
    )
    return int(centers[0]), per_param


def _apply_variance_inflation(
    data: np.ndarray,
    bf_ts: np.ndarray,   # shape (P,)
    T: int,
    D: int,
    rng: np.random.Generator,
    strength: float,
) -> None:
    """Critical-slowing-down: post-bifurcation variance grows linearly."""
    for p, bf_t in enumerate(bf_ts):
        t_range = np.arange(bf_t, T)
        scales  = 1.0 + strength * (t_range - bf_t) / (T - bf_t)
        data[p, :, bf_t:] *= scales[np.newaxis, :]
        data[p, :, bf_t:] += 0.3 * rng.standard_normal((D, len(t_range)))


def _apply_mean_shift(
    data: np.ndarray,
    bf_ts: np.ndarray,   # shape (P,)
    T: int,
    rng: np.random.Generator,
    strength: float,
) -> None:
    """Abrupt shift in process mean; direction randomised per parameter."""
    for p, bf_t in enumerate(bf_ts):
        direction = rng.choice([-1, 1])
        shift     = direction * strength * (1.0 + rng.uniform(0, 0.5))
        data[p, :, bf_t:] += shift


def _apply_hopf(
    data: np.ndarray,
    bf_ts: np.ndarray,   # shape (P,)
    T: int,
    rng: np.random.Generator,
    strength: float,
) -> None:
    """Growing sinusoidal oscillation onset (Hopf bifurcation signature)."""
    for p, bf_t in enumerate(bf_ts):
        t_range  = np.arange(bf_t, T)
        freq     = rng.uniform(0.02, 0.10)          # cycles per timestep
        phase    = rng.uniform(0, 2 * np.pi)
        envelope = strength * (t_range - bf_t) / (T - bf_t)
        osc      = envelope * np.sin(2 * np.pi * freq * t_range + phase)
        data[p, :, bf_t:] += osc[np.newaxis, :]


def _apply_cascade(
    data: np.ndarray,
    event_bf_ts: np.ndarray,   # shape (2, P)
    T: int,
    D: int,
    rng: np.random.Generator,
    strength: float,
) -> None:
    """Two sequential events: first variance, then mean-shift."""
    # Event 1 – variance inflation
    _apply_variance_inflation(data, event_bf_ts[0], T, D, rng, strength * 0.6)
    # Event 2 – mean shift on top
    _apply_mean_shift(data, event_bf_ts[1], T, rng, strength * 0.8)


def _severity_score(bf_center: int, T: int, strength: float) -> float:
    """
    Continuous severity in [0, 1].
    Earlier bifurcations with higher strength → closer to 1.
    """
    earliness = 1.0 - (bf_center / T)               # earlier = higher
    s = np.clip(0.5 * earliness + 0.5 * (strength / 4.0), 0.0, 1.0)
    return float(s)


def _assign_split(
    idx: int,
    total: int,
    rng: np.random.Generator,
    cfg: DatasetConfig,
    _cache: dict,
) -> str:
    """
    Deterministic stratified split assignment.  We generate the full permutation
    once (cached by 'total') so every index maps to a stable split label.
    """
    key = total
    if key not in _cache:
        perm = rng.permutation(total)
        n_train = int(cfg.train_frac * total)
        n_val   = int(cfg.val_frac   * total)
        mapping = {}
        for rank, orig_idx in enumerate(perm):
            if rank < n_train:
                mapping[orig_idx] = "train"
            elif rank < n_train + n_val:
                mapping[orig_idx] = "val"
            else:
                mapping[orig_idx] = "test"
        _cache[key] = mapping
    return _cache[key][idx]


def _validate(data: np.ndarray, label: str) -> None:
    if not np.isfinite(data).all():
        n_bad = (~np.isfinite(data)).sum()
        raise ValueError(
            f"Recording '{label}' contains {n_bad} NaN/Inf values. "
            "Check OU parameters or bifurcation strength."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_recordings(
    cfg: DatasetConfig | None = None,
    **kwargs,
) -> list[Recording]:
    """
    Generate a list of Recording objects suitable for CNN-LSTM training.

    Parameters
    ----------
    cfg     : DatasetConfig instance.  If None, a default config is used and any
              **kwargs are forwarded to DatasetConfig().
    **kwargs: Convenience overrides forwarded to DatasetConfig when cfg is None.

    Returns
    -------
    list[Recording]  – shuffled so null and positive recordings are interleaved.

    Each Recording carries the following extra attributes (stored in metadata):
        split     : 'train' | 'val' | 'test'
        bif_type  : BifurcationType | None
        severity  : float in [0, 1] | None   (positive recordings only)
    """
    if cfg is None:
        cfg = DatasetConfig(**kwargs)

    rng = np.random.default_rng(cfg.seed)

    # Separate RNG for split assignment so split labels are stable even if
    # generation logic changes.
    split_rng = np.random.default_rng(cfg.seed + 9999)
    split_cache: dict = {}

    T, D, P = cfg.time_len, cfg.depth, cfg.num_params
    recordings: list[Recording] = []

    # ── Null recordings ───────────────────────────────────────────────────────
    for i in range(cfg.n_null):
        theta = float(rng.uniform(*cfg.theta_range))
        sigma = float(rng.uniform(*cfg.sigma_range))


    
        # NEW
        generator = OceanographicSignalGenerator(cfg, rng)
        data = generator.generate_baseline(T, D)   # returns (P, D, T)
        generator.add_observational_noise(data)    # in-place
        
        _validate(data, f"null_{i:04d}")

        rec = Recording(data=data, bifurcation_t=None, recording_id=f"null_{i:04d}")
        rec.metadata = {
            "split":    _assign_split(i, cfg.n_null, split_rng, cfg, split_cache),
            "bif_type": None,
            "severity": None,
            "theta":    theta,
            "sigma":    sigma,
        }
        recordings.append(rec)

    # ── Positive recordings ───────────────────────────────────────────────────
    # Distribute evenly across bifurcation types
    bif_types   = list(BifurcationType)
    n_per_type  = cfg.n_positive // len(bif_types)
    remainder   = cfg.n_positive % len(bif_types)
    type_counts = {bt: n_per_type + (1 if j < remainder else 0)
                   for j, bt in enumerate(bif_types)}

    pos_idx = 0
    for bif_type, count in type_counts.items():
        for _ in range(count):
            theta    = float(rng.uniform(*cfg.theta_range))
            sigma    = float(rng.uniform(*cfg.sigma_range))
            strength = float(rng.uniform(0.3, 1.2))   # bifurcation strength

            # data = _ou_process(T, D, P, rng, theta, sigma)
            # NEW
            generator = OceanographicSignalGenerator(cfg, rng)
            data = generator.generate_baseline(T, D)   # returns (P, D, T)
            generator.add_observational_noise(data)    # in-place

            if bif_type == BifurcationType.CASCADE:
                center_bf_t, event_bf_ts = _sample_bf_times(rng, cfg, num_events=2)
                _apply_cascade(data, event_bf_ts, T, D, rng, strength)
                per_param_bf_ts = event_bf_ts[0]   # primary event for label
            else:
                center_bf_t, event_bf_ts = _sample_bf_times(rng, cfg, num_events=1)
                per_param_bf_ts = event_bf_ts[0]

                if bif_type == BifurcationType.VARIANCE:
                    _apply_variance_inflation(data, per_param_bf_ts, T, D, rng, strength)
                elif bif_type == BifurcationType.MEAN_SHIFT:
                    _apply_mean_shift(data, per_param_bf_ts, T, rng, strength)
                elif bif_type == BifurcationType.HOPF:
                    _apply_hopf(data, per_param_bf_ts, T, rng, strength)

            # _add_sensor_noise(data, rng, cfg.sensor_noise_std, cfg.spike_prob, cfg.spike_magnitude)
            _validate(data, f"pos_{pos_idx:04d}")

            severity = _severity_score(center_bf_t, T, strength)

            rec = Recording(
                data=data,
                bifurcation_t=center_bf_t,
                recording_id=f"pos_{pos_idx:04d}",
                param_bifurcation_ts=per_param_bf_ts,
            )
            rec.metadata = {
                "split":    _assign_split(pos_idx, cfg.n_positive, split_rng, cfg, split_cache),
                "bif_type": bif_type,
                "severity": severity,
                "theta":    theta,
                "sigma":    sigma,
                "strength": strength,
            }
            recordings.append(rec)
            pos_idx += 1

    # Shuffle so batches see a mix of classes (preserves split labels)
    final_rng = np.random.default_rng(cfg.seed + 1)
    order = final_rng.permutation(len(recordings))
    recordings = [recordings[i] for i in order]

    return recordings


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    recordings: list[Recording],
) -> dict[str, list[Recording]]:
    """
    Partition a list of Recordings into {'train': [...], 'val': [...], 'test': [...]}.
    Relies on rec.metadata['split'] set by make_synthetic_recordings().
    """
    splits: dict[str, list[Recording]] = {"train": [], "val": [], "test": []}
    for rec in recordings:
        s = getattr(rec, "metadata", {}).get("split", "train")
        splits[s].append(rec)
    return splits


def dataset_summary(recordings: list[Recording]) -> None:
    """Print a human-readable summary of a dataset."""
    from collections import Counter

    total   = len(recordings)
    labels  = ["positive" if r.bifurcation_t is not None else "null" for r in recordings]
    splits  = [getattr(r, "metadata", {}).get("split", "?") for r in recordings]
    btypes  = [getattr(r, "metadata", {}).get("bif_type", None) for r in recordings]

    label_counts = Counter(labels)
    split_counts = Counter(splits)
    btype_counts = Counter(str(b) for b in btypes if b is not None)

    severities = [
        getattr(r, "metadata", {}).get("severity")
        for r in recordings
        if getattr(r, "metadata", {}).get("severity") is not None
    ]

    print(f"{'─'*55}")
    print(f"  Dataset summary  ({total} recordings)")
    print(f"{'─'*55}")
    print(f"  Classes   : null={label_counts['null']}  positive={label_counts['positive']}")
    print(f"  Splits    : train={split_counts['train']}  val={split_counts['val']}  test={split_counts['test']}")
    print(f"  Bif types : {dict(btype_counts)}")
    if severities:
        sev = np.array(severities)
        print(f"  Severity  : min={sev.min():.3f}  mean={sev.mean():.3f}  max={sev.max():.3f}")
    shape = recordings[0].data.shape if recordings else "n/a"
    print(f"  Data shape: {shape}  (P×D×T)")
    print(f"{'─'*55}")

def plot_recording(
    recordings: "list[Recording]",
    recording_id: "str | None" = None,
    index: "int | None" = None,
    filter_type: "str | None" = None,
    params: "list[int] | None" = None,
    depth_agg: str = "mean",
    figsize: "tuple[int, int] | None" = None,
    seed: int = 0,
) -> "plt.Figure":
    """
    Plot the timeseries of a single Recording.

    Parameters
    ----------
    recordings   : list of Recording objects, as returned by
                   make_synthetic_recordings().
    recording_id : recording_id string to look up (e.g. 'pos_0012').
                   Mutually exclusive with `index`.
    index        : integer index into `recordings`.
                   Mutually exclusive with `recording_id`.
                   If neither is given, a random recording is chosen.
    filter_type  : restrict random / index selection to a recording subtype.
                   Accepted values (case-insensitive):
                     'null'       – stable null recordings only
                     'positive'   – any positive recording
                     'variance'   – BifurcationType.VARIANCE
                     'mean_shift' – BifurcationType.MEAN_SHIFT
                     'hopf'       – BifurcationType.HOPF
                     'cascade'    – BifurcationType.CASCADE
                   Ignored when `recording_id` is supplied.
    params       : list of parameter indices (0-based) to plot.
                   Defaults to all parameters.
    depth_agg    : how to collapse the depth (D) dimension before plotting.
                   'mean'   – plot the mean across channels (default).
                   'median' – plot the median across channels.
                   'all'    – overlay every channel as faint lines + bold mean.
    figsize      : (width, height) in inches.  Auto-sized if None.
    seed         : RNG seed used only when choosing a random recording.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object (caller can call plt.show(), fig.savefig(), etc.).

    Examples
    --------
    # Completely random:
    fig = plot_recording(recordings)
    plt.show()

    # Random null recording:
    fig = plot_recording(recordings, filter_type='null')

    # Random Hopf recording:
    fig = plot_recording(recordings, filter_type='hopf')

    # Specific index within cascade recordings:
    fig = plot_recording(recordings, index=2, filter_type='cascade')

    # By exact id:
    fig = plot_recording(recordings, recording_id='pos_0007', depth_agg='all')
    plt.show()
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # ── Validate mutual exclusion ─────────────────────────────────────────────
    if recording_id is not None and index is not None:
        raise ValueError("Specify at most one of `recording_id` or `index`.")

    # ── Apply filter_type to build the candidate pool ─────────────────────────
    _FILTER_MAP = {
        "null":       lambda r: r.bifurcation_t is None,
        "positive":   lambda r: r.bifurcation_t is not None,
        "variance":   lambda r: _bif_name(r) == "VARIANCE",
        "mean_shift": lambda r: _bif_name(r) == "MEAN_SHIFT",
        "hopf":       lambda r: _bif_name(r) == "HOPF",
        "cascade":    lambda r: _bif_name(r) == "CASCADE",
    }

    def _bif_name(r):
        bt = getattr(r, "metadata", {}).get("bif_type", None)
        return bt.name if bt is not None else None

    if filter_type is not None and recording_id is None:
        key = filter_type.lower()
        if key not in _FILTER_MAP:
            raise ValueError(
                f"Unknown filter_type '{filter_type}'. "
                f"Choose from: {list(_FILTER_MAP)}"
            )
        pool = [r for r in recordings if _FILTER_MAP[key](r)]
        if not pool:
            raise ValueError(
                f"No recordings match filter_type='{filter_type}'."
            )
    else:
        pool = recordings

    # ── Select the recording ──────────────────────────────────────────────────
    if recording_id is not None:
        matches = [r for r in recordings if r.recording_id == recording_id]
        if not matches:
            raise KeyError(f"No recording with id '{recording_id}'.")
        rec = matches[0]
    elif index is not None:
        if index >= len(pool):
            raise IndexError(
                f"index {index} out of range for pool of {len(pool)} "
                f"recordings (filter_type='{filter_type}')."
            )
        rec = pool[index]
    else:
        rng = np.random.default_rng(seed)
        rec = pool[int(rng.integers(0, len(pool)))]

    # ── Unpack ────────────────────────────────────────────────────────────────
    data     = rec.data                          # (P, D, T)
    P, D, T  = data.shape
    t_axis   = np.arange(T)
    is_pos   = rec.bifurcation_t is not None
    meta     = getattr(rec, "metadata", {})
    bif_type = meta.get("bif_type", None)
    severity = meta.get("severity", None)
    split    = meta.get("split", "?")
    theta    = meta.get("theta", None)
    sigma    = meta.get("sigma", None)

    params_to_plot = list(range(P)) if params is None else params

    # ── Layout ────────────────────────────────────────────────────────────────
    n_rows = len(params_to_plot)
    if figsize is None:
        figsize = (13, 2.6 * n_rows + 1.5)   # +1.5 reserves space for title

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )

    # Leave a fixed top margin for the two-line suptitle before tight_layout
    fig.subplots_adjust(top=0.88)

    # ── Colour palette ────────────────────────────────────────────────────────
    SIGNAL_COLOUR  = "#378add"
    CHANNEL_ALPHA  = 0.12
    ENV_COLOUR     = "#ef9f27"
    BF1_COLOUR     = "#e24b4a"
    BF2_COLOUR     = "#993c1d"

    # ── Per-parameter subplots ────────────────────────────────────────────────
    for row, p in enumerate(params_to_plot):
        ax  = axes[row][0]
        sig = data[p]                            # (D, T)

        # Aggregate or overlay depth channels
        if depth_agg == "all":
            for d in range(D):
                ax.plot(t_axis, sig[d], color=SIGNAL_COLOUR,
                        alpha=CHANNEL_ALPHA, linewidth=0.6, zorder=1)
            mean_sig = sig.mean(axis=0)
            ax.plot(t_axis, mean_sig, color=SIGNAL_COLOUR,
                    linewidth=1.4, zorder=3, label="mean")
        elif depth_agg == "median":
            mean_sig = np.median(sig, axis=0)
            ax.plot(t_axis, mean_sig, color=SIGNAL_COLOUR,
                    linewidth=1.4, zorder=3)
        else:   # 'mean'
            mean_sig = sig.mean(axis=0)
            ax.plot(t_axis, mean_sig, color=SIGNAL_COLOUR,
                    linewidth=1.4, zorder=3)

        # Rolling ±1σ envelope (window = 5 % of T)
        if is_pos:
            w = max(2, T // 20)
            roll_std = np.array([
                sig[:, max(0, t - w): t + 1].std()
                for t in range(T)
            ])
            ax.fill_between(
                t_axis,
                mean_sig - roll_std,
                mean_sig + roll_std,
                color=ENV_COLOUR, alpha=0.18, zorder=2, label="±1σ envelope",
            )

        # Per-parameter bifurcation time
        bf_t_p = None
        if is_pos and hasattr(rec, "param_bifurcation_ts"):
            pbt = rec.param_bifurcation_ts
            if pbt is not None and p < len(pbt):
                bf_t_p = int(pbt[p])

        # Canonical bifurcation line
        if is_pos:
            ax.axvline(rec.bifurcation_t, color=BF1_COLOUR,
                       linewidth=1.5, zorder=4, label="bifurcation t")

        # Per-parameter line when it differs from canonical
        if bf_t_p is not None and bf_t_p != rec.bifurcation_t:
            ax.axvline(bf_t_p, color=BF1_COLOUR, linewidth=1.0,
                       linestyle="--", zorder=4, alpha=0.7,
                       label=f"bf_t P{p}")

        # Cascade second event
        bf2 = meta.get("cascade_bf2_t", None)
        if bf2 is not None:
            ax.axvline(bf2, color=BF2_COLOUR, linewidth=1.2,
                       linestyle=":", zorder=4, label="cascade bf₂")

        ax.set_ylabel(f"P{p}", fontsize=10, labelpad=4)
        ax.tick_params(labelsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        if row == 0 and is_pos:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.7, ncol=3)

    axes[-1][0].set_xlabel("Timestep", fontsize=10)

    # ── Suptitle (inside the figure, two lines) ───────────────────────────────
    label_str = "POSITIVE" if is_pos else "NULL"
    type_str  = f"  ·  {bif_type.name}" if bif_type is not None else ""
    sev_str   = f"  ·  severity {severity:.2f}" if severity is not None else ""
    split_str = f"  [{split}]"
    noise_str = (
        f"θ={theta:.3f}  σ={sigma:.3f}  ·  " if theta is not None else ""
    )
    agg_label = "all channels" if depth_agg == "all" else depth_agg
    agg_str   = f"depth agg: {agg_label}"

    line1 = f"{rec.recording_id}  –  {label_str}{type_str}{sev_str}{split_str}"
    line2 = f"{noise_str}{agg_str}"

    fig.suptitle(f"{line1}\n{line2}", fontsize=11, y=0.97,
                 ha="left", x=0.01, va="top")

    fig.tight_layout(rect=[0, 0, 1, 0.88])   # leaves top 12 % for suptitle

    return fig

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