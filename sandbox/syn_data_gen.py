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
from bi_data_struct import Recording



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
# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = DatasetConfig(
        n_null=400, n_positive=400,
        num_params=4, depth=32, time_len=512,
        seed=42,
    )
    recordings = make_synthetic_recordings(cfg)
    dataset_summary(recordings)

    splits = split_dataset(recordings)
    print(f"\n  Train recordings : {len(splits['train'])}")
    print(f"  Val   recordings : {len(splits['val'])}")
    print(f"  Test  recordings : {len(splits['test'])}")
    fig = plot_recording(recordings, filter_type='null')
    plt.savefig("example_null.png")
    fig = plot_recording(recordings, filter_type='hopf')
    plt.savefig("example_hopf.png")
    fig = plot_recording(recordings, filter_type='cascade')
    plt.savefig("example_cascade.png")
    fig = plot_recording(recordings, filter_type='variance')
    plt.savefig("example_variance.png")
    fig = plot_recording(recordings, filter_type='mean_shift')
    plt.savefig("example_mean_shift.png")
    fig = plot_recording(recordings, filter_type='positive')  # any positive
    plt.savefig("example_any_positive.png")
    plt.show()