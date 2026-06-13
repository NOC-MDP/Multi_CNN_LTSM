import os
import glob
import xarray as xr
import pandas as pd
import numpy as np

def extract_location_timeseries(
    data_dir: str,
    atmo_data_dir: str,
    lat: float, 
    lon: float, 
    output_csv_path: str
):
    """
    Reads segmented monthly NetCDF files organized inside nested year subfolders,
    extracts a single point location, joins parameters, and saves a cleaned CSV.
    """
    print(f"--- Starting Multi-Year Extraction for Location (Lat: {lat}, Lon: {lon}) ---")
    
    param_files = {
        'temperature': os.path.join(data_dir, "**", "*grid_T_votemper.nc"),
        'salinity':    os.path.join(data_dir, "**", "*grid_T_vosaline.nc"),
        'ssh':         os.path.join(data_dir, "**", "*grid_T_sossheig.nc"), 
        'u_velocity':  os.path.join(data_dir, "**", "*grid_U_vozocrtx.nc"),
        'v_velocity':  os.path.join(data_dir, "**", "*grid_V_vomecrty.nc"),
        'u_wind':         os.path.join(atmo_data_dir, "**", "*m01s30i201.nc"),
        'v_wind':         os.path.join(atmo_data_dir, "**", "*m01s30i202.nc"),
    }
    
    extracted_series = {}
    
    # ─── OPTIMIZATION 1: PRE-CALCULATE OCEAN GRID SPATIAL INDICES ────────────
    # We still pre-calculate the NEMO indices to save time for ocean files
    nemo_sample = None
    for param in ['temperature', 'salinity', 'ssh', 'u_velocity', 'v_velocity']:
        found = glob.glob(param_files[param], recursive=True)
        if found:
            nemo_sample = sorted(found)[0]
            break
            
    if nemo_sample:
        print(f"Mapping NEMO ocean grid using: {os.path.basename(nemo_sample)}")
        with xr.open_dataset(nemo_sample) as sample_ds:
            dist = (sample_ds.nav_lat - lat)**2 + (sample_ds.nav_lon - lon)**2
            y_idx, x_idx = np.unravel_index(dist.argmin(), dist.shape)
            print(f"  Mapped NEMO indices -> x: {x_idx}, y: {y_idx}")
    else:
        raise FileNotFoundError("Could not find any ocean datasets to establish mapping.")

    # ─── OPTIMIZATION 2: DYNAMIC PREPROCESS FILTER ──────────────────────────
    def slice_pixel_on_load(ds):
        """Slices the dataset to the target pixel dynamically based on grid structure."""
        
        # Scenario A: Ocean Grid (NEMO)
        if 'nav_lat' in ds.coords and 'x' in ds.dims and 'y' in ds.dims:
            ds_sliced = ds.isel(x=x_idx, y=y_idx)
            
        # Scenario B: Atmosphere Grid (Met Office UM Wind variables)
        elif 'lat_um_atmos_grid_uv' in ds.dims:
            # UM dimensions use long custom names. Use .sel(..., method='nearest')
            ds_sliced = ds.sel(
                lat_um_atmos_grid_uv=lat, 
                lon_um_atmos_grid_uv=lon + 360 if lon < 0 and ds['lon_um_atmos_grid_uv'].max() > 180 else lon, 
                method='nearest'
            )
            
        # Scenario C: Fallback standard lat/lon
        elif 'lat' in ds.dims or 'latitude' in ds.dims:
            lat_dim = 'lat' if 'lat' in ds.dims else 'latitude'
            lon_dim = 'lon' if 'lon' in ds.dims else 'longitude'
            ds_sliced = ds.sel({lat_dim: lat, lon_dim: lon}, method='nearest')
            
        else:
            raise KeyError(f"Unknown grid structure. Coordinates present: {list(ds.coords.keys())}")
            
        # Select surface level immediately if vertical dimensions exist
        # Added 'um_atmos_DP36CCM' from your error log to clear atmospheric pressure levels too
        for depth_dim in ['deptht', 'depthu', 'depthv', 'um_atmos_DP36CCM']:
            if depth_dim in ds_sliced.dims:
                ds_sliced = ds_sliced.isel({depth_dim: 0})
                
        return ds_sliced

    # ─── LOOP PARAMETERS ─────────────────────────────────────────────────────
    for param_name, glob_pattern in param_files.items():
        print(f"\nProcessing parameter: {param_name}...")
        
        files = sorted(glob.glob(glob_pattern, recursive=True))
        if not files:
            raise FileNotFoundError(f"No NetCDF files found matching pattern: {glob_pattern}")
            
        print(f"  Found {len(files)} total monthly files. Stitching...")
        
        # We explicitly add 'time' chunks to handle both time_counter (NEMO) and time (UM)
        with xr.open_mfdataset(
            files, 
            combine='by_coords', 
            parallel=True, 
            preprocess=slice_pixel_on_load,
            chunks={'time_counter': 1, 'time': 1} 
        ) as point_ds:
            
            if 'time_counter' in point_ds.dims:
                point_ds = point_ds.rename({'time_counter': 'time'})
            
            nemo_var_map = {
                'temperature': 'votemper',
                'salinity': 'vosaline',
                'ssh': 'sossheig',
                'u_velocity': 'vozocrtx',
                'v_velocity': 'vomecrty',
                'u_wind': 'm01s30i201',
                'v_wind': 'm01s30i202',
            }
            
            nc_var_name = nemo_var_map[param_name]
            if nc_var_name not in point_ds.data_vars:
                raise ValueError(f"Could not find variable '{nc_var_name}'")
            
            series = point_ds[nc_var_name].to_series()
            
            if isinstance(series.index, pd.MultiIndex):
                series = series.droplevel([level for level in series.index.names if level != 'time'])
            
            if series.index.duplicated().any():
                print("  ⚠️ Found duplicate timestamps. Retaining the first entry per timestamp.")
                series = series[~series.index.duplicated(keep='first')]
                
            extracted_series[param_name] = series

    # Merge variables by their shared timestamps
    print("\nMerging all variables into combined dataset...")
    combined_df = pd.DataFrame(extracted_series)
    combined_df = combined_df.sort_index()
    
    print(f"Final Combined Timeline: {combined_df.index.min()} to {combined_df.index.max()}")
    
    # Save output to file
    combined_df.to_csv(output_csv_path, index_label='time')
    print(f"🎉 Success! Exported to: {output_csv_path}")
    
if __name__ == "__main__":
    import gc 
    WORK_DIR = "CANARI"
    ensemble_st = 9
    ensemble_end = 40
    model_types = ["SSP370"]
    retries = 5
    for i in range(ensemble_st-1, ensemble_end, 1):
        for model_type in model_types:
            retry = 0
            while retry < retries:
                print(f"getting {model_type} for ensemble {i+1} on retry {retry}")
                DATA_DIRECTORY = f"/gws/ssde/j25b/canari/shared/large-ensemble/priority/{model_type}/{i+1}/OCN/yearly/"
                ATMO_DATA_DIRECTORY = f"/gws/ssde/j25b/canari/shared/large-ensemble/priority/{model_type}/{i+1}/ATM/yearly/"
                TARGET_LAT = 60.819
                TARGET_LON = -56.506
                
                OUTPUT_FILE = f"{WORK_DIR}/ensembles/CANARI_{model_type}_{i+1}_LON_{TARGET_LON}_LAT_{TARGET_LAT}.csv"
                
                if not os.path.exists(DATA_DIRECTORY):
                    print(f"Directory '{DATA_DIRECTORY}' not found. Please verify your base path.")
                    break 
                else:
                    try:
                        extract_location_timeseries(
                            data_dir=DATA_DIRECTORY,
                            atmo_data_dir=ATMO_DATA_DIRECTORY,
                            lat=TARGET_LAT,
                            lon=TARGET_LON,
                            output_csv_path=OUTPUT_FILE
                        )
                        retry = retries
                    except Exception as e:
                        print(f"\n❌ Extraction stopped due to error: {e}")
                        retry = retry + 1
                        import time
                        time.sleep(600) 
            
            gc.collect()