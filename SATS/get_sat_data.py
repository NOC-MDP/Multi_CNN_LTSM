#!/usr/bin/env python3
"""
extract_timeseries.py
---------------------
Extract a monthly-mean timeseries at a single lat/lon point from multiple
NetCDF files, each potentially having different temporal resolutions, spatial
resolutions, and time spans. All series are clipped to the shortest common
time period and written to a single CSV.

Output format
-------------
time,temperature,salinity,ssh,u_velocity,v_velocity,mld
1950-01-16 00:00:00,3.6882887,35.013508,-1.445205,-0.0022555264,-0.0052452167,584.9641

Usage
-----
    python extract_timeseries.py

Edit the CONFIGURATION block below to point at your files, choose your
target point, and map output column names to (file, variable) pairs.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

TARGET_LAT = 60.819
TARGET_LON = -56.506
WORK_DIR = "/work/scratch-pw5/thopri/"
OUTPUT_CSV = Path(f"ensembles/sat_obs_1_LON_{TARGET_LON}_LAT_{TARGET_LAT}.csv")

# Mapping:  output_column_name → (netcdf_file_path, variable_name_in_file)
# Files may be on any grid (regular or curvilinear) and any time resolution.
# Variables that carry a depth dimension will have their surface level selected
# automatically (see DEPTH_NAMES below).
VARIABLE_FILES = {
    "temperature": (f"{WORK_DIR}/C3S-GLO-SST-L4-REP-OBS-SST_analysed_sst_66.98W-43.03W_58.03N-71.98N_1982-01-01-2024-12-31.nc",  "analysed_sst"),
    "salinity":    (f"{WORK_DIR}/cmems_obs-mob_glo_phy-sss_my_multi_P1M_dos-sos_179.94W-179.94E_89.94S-89.94N_0.00m_1993-01-01-2024-12-01.nc",     "sos"),
    "ssh":         (f"{WORK_DIR}/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_adt_66.94W-43.06W_58.06N-71.94N_1993-01-01-2025-10-18.nc",          "adt"),
    "u_velocity":  (f"{WORK_DIR}/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_sla-ugos-vgos_69.94W-40.06W_50.06N-69.94N_1993-01-01-2025-10-18.nc",   "ugos"),
    "v_velocity":  (f"{WORK_DIR}/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_sla-ugos-vgos_69.94W-40.06W_50.06N-69.94N_1993-01-01-2025-10-18.nc",   "vgos"),
    "mld":  (f"{WORK_DIR}/cmems_mod_glo_phy_my_0.083deg_P1M-m_mlotst_180.00W-179.92E_80.00S-90.00N_1993-01-01-2026-03-01.nc",   "mlotst"),

}

# Any depth dimension found in a variable will be sliced at index 0 (surface).
# Extend this list if your files use non-standard names.
DEPTH_NAMES = ("deptht", "depthu", "depthv", "depthw",
               "depth", "z", "lev", "level", "olevel",
               "nav_lev", "z_t", "z_w")

# Latitude / longitude coordinate name patterns (case-insensitive substring match)
LAT_PATTERNS = ("nav_lat", "lat",  "latitude",  "y_geostationary")
LON_PATTERNS = ("nav_lon", "lon",  "longitude", "x_geostationary")

# ── HELPERS ────────────────────────────────────────────────────────────────────

def _find_coord(da: xr.DataArray, patterns: tuple[str, ...]) -> str | None:
    """Return the first coordinate name that matches any pattern (case-insensitive)."""
    for pat in patterns:
        for cname in da.coords:
            if pat.lower() in cname.lower():
                return cname
    return None


def _nearest_2d(lat2d: np.ndarray, lon2d: np.ndarray,
                target_lat: float, target_lon: float) -> tuple[int, int]:
    """
    Brute-force nearest-neighbour search on a 2-D curvilinear grid.
    Returns (i, j) indices into the 2-D arrays.
    """
    dist = (lat2d - target_lat) ** 2 + (lon2d - target_lon) ** 2
    return np.unravel_index(np.argmin(dist), dist.shape)


def _drop_depth(da: xr.DataArray) -> xr.DataArray:
    """If a known depth dimension exists, keep only the surface level (index 0)."""
    for dname in DEPTH_NAMES:
        if dname in da.dims:
            da = da.isel({dname: 0})
            break
    return da


def _select_point(da: xr.DataArray,
                  target_lat: float, target_lon: float) -> xr.DataArray:
    """
    Select the nearest grid point, handling both regular (1-D) and
    curvilinear (2-D) grids such as NEMO ORCA grids.
    """
    lat_coord = _find_coord(da, LAT_PATTERNS)
    lon_coord = _find_coord(da, LON_PATTERNS)

    if lat_coord is None or lon_coord is None:
        raise ValueError(
            f"Cannot identify lat/lon coordinates in '{da.name}'.\n"
            f"  Available coords: {list(da.coords)}\n"
            f"  Extend LAT_PATTERNS / LON_PATTERNS in the config if needed."
        )

    lat_vals = da[lat_coord].values
    lon_vals = da[lon_coord].values

    if lat_vals.ndim == 1:
        # ── Regular rectilinear grid ─────────────────────────────────────────
        da = da.sel({lat_coord: target_lat, lon_coord: target_lon},
                    method="nearest")
    else:
        # ── Curvilinear 2-D grid (e.g. NEMO ORCA) ───────────────────────────
        i, j = _nearest_2d(lat_vals, lon_vals, target_lat, target_lon)
        y_dim, x_dim = da[lat_coord].dims          # e.g. ("y", "x")
        da = da.isel({y_dim: i, x_dim: j})

    return da.squeeze(drop=True)


def _to_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    """
    Convert a CFTime or mixed index to a standard DatetimeIndex.
    Handles proleptic_gregorian, 360_day, all_leap, etc.
    """
    if isinstance(index, pd.DatetimeIndex):
        return index
    # cftime objects: convert via string (safe for non-standard calendars)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.DatetimeIndex(
                [t.strftime("%Y-%m-%d %H:%M:%S") for t in index],
                name="time"
            )
        except AttributeError:
            return pd.to_datetime(index)


def load_monthly_series(filepath: str | Path, var: str,
                        target_lat: float, target_lon: float) -> pd.Series:
    """
    Open *filepath*, extract *var* at (*target_lat*, *target_lon*),
    drop the surface depth level if present, resample to monthly means,
    and return a named pandas Series.
    """
    ds = xr.open_dataset(filepath, use_cftime=True)

    if var not in ds:
        raise KeyError(
            f"Variable '{var}' not found in {filepath}.\n"
            f"  Available variables: {list(ds.data_vars)}"
        )

    da = ds[var]
    da = _drop_depth(da)
    da = _select_point(da, target_lat, target_lon)

    # Build pandas Series with a proper DatetimeIndex
    ts = da.to_series()
    ts.index = _to_datetime_index(ts.index)
    ts.index.name = "time"
    ts.name = var

    # Resample to monthly means.
    # "MS" labels each bin at the first of the month; shift by 15 days to
    # give a mid-month timestamp that matches typical model output conventions.
    ts_monthly = ts.resample("MS").mean()
    ts_monthly.index = ts_monthly.index + pd.Timedelta(days=15)
    ts_monthly.index.name = "time"

    ds.close()
    return ts_monthly


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\nTarget point : lat={TARGET_LAT}°N  lon={TARGET_LON}°E")
    print(f"Output CSV   : {OUTPUT_CSV}\n")

    series: dict[str, pd.Series] = {}

    for col, (fpath, vname) in VARIABLE_FILES.items():
        fpath = Path(fpath)
        if not fpath.exists():
            print(f"  [SKIP] {col}: file not found → {fpath}")
            continue

        try:
            ts = load_monthly_series(fpath, vname, TARGET_LAT, TARGET_LON)
            series[col] = ts
            print(f"  {col:<14}  {len(ts):>5} monthly steps"
                  f"  [{ts.index[0]}  →  {ts.index[-1]}]")
        except Exception as exc:
            print(f"  [ERROR] {col}: {exc}", file=sys.stderr)
            raise

    if not series:
        print("No series loaded – check file paths and variable names.")
        sys.exit(1)

    # ── Clip to the common (shortest) time span ──────────────────────────────
    t_start = max(s.index[0]  for s in series.values())
    t_end   = min(s.index[-1] for s in series.values())

    if t_start > t_end:
        print("\n[ERROR] No overlapping time period found across the files.")
        sys.exit(1)

    print(f"\nClipping to common period: {t_start}  →  {t_end}")

    clipped = {col: ts.loc[t_start:t_end] for col, ts in series.items()}

    # Align on a complete monthly grid; any missing steps become NaN
    full_index = pd.date_range(t_start, t_end, freq="MS") + pd.Timedelta(days=15)
    full_index.name = "time"

    df = pd.DataFrame(clipped, index=full_index)
    df.index.name = "time"

    # ------------------------------------------------------------------
    # Remove NaNs at the beginning and end of the series
    # ------------------------------------------------------------------
    valid_rows = ~df.isna().all(axis=1)
    
    if valid_rows.any():
        first_valid = valid_rows.idxmax()
        last_valid = valid_rows[::-1].idxmax()
        df = df.loc[first_valid:last_valid]
    
    # ------------------------------------------------------------------
    # Interpolate internal gaps
    # limit = maximum number of consecutive NaNs to fill
    # ------------------------------------------------------------------
    MAX_GAP = 3
    
    df = df.interpolate(
        method="linear",
        limit=MAX_GAP,
        limit_area="inside",
    )
    
    # ------------------------------------------------------------------
    # Check whether any NaNs remain
    # ------------------------------------------------------------------
    remaining = int(df.isna().any(axis=1).sum())
    
    if remaining:
        raise ValueError(
            f"{remaining} rows still contain NaNs after interpolation. "
            f"Gaps exceed MAX_GAP={MAX_GAP}."
        )
    
    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    df.to_csv(OUTPUT_CSV, float_format="%.7g")
    
    print(
        f"\nWritten {len(df)} rows × {len(df.columns)} columns → {OUTPUT_CSV}"
    )

    # Print a preview
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}\n")


if __name__ == "__main__":
    main()