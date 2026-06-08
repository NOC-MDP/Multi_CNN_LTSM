import os
import glob
import xarray as xr
import pandas as pd
import numpy as np

def extract_location_timeseries(
    data_dir: str, 
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
        'mld':         os.path.join(data_dir, "**", "*grid_T_somxl010.nc"),
    }
    
    extracted_series = {}
    
    # ─── OPTIMIZATION 1: PRE-CALCULATE SPATIAL INDICES ───────────────────────
    # Find the very first available file across any parameter to determine the grid index
    sample_file = None
    for pattern in param_files.values():
        found = glob.glob(pattern, recursive=True)
        if found:
            sample_file = sorted(found)[0]
            break
            
    if not sample_file:
        raise FileNotFoundError("Could not find a single NetCDF file to determine grid mapping.")
    
    print(f"Mapping lat/lon grid using sample file: {os.path.basename(sample_file)}")
    with xr.open_dataset(sample_file) as sample_ds:
        if 'nav_lat' in sample_ds.coords and 'nav_lon' in sample_ds.coords:
            dist = (sample_ds.nav_lat - lat)**2 + (sample_ds.nav_lon - lon)**2
            y_idx, x_idx = np.unravel_index(dist.argmin(), dist.shape)
            has_nav_coords = True
            print(f"  Mapped to grid indices -> x: {x_idx}, y: {y_idx}")
        else:
            has_nav_coords = False
            print("  Standard 'lat'/'lon' dimensions detected. Using coordinate selection.")

    # ─── OPTIMIZATION 2: DEFINE A PREPROCESS FILTER ──────────────────────────
    def slice_pixel_on_load(ds):
        """Slices the dataset to the target pixel immediately upon opening."""
        if has_nav_coords:
            # Slice spatially right away so we don't carry the whole grid into memory
            ds_sliced = ds.isel(x=x_idx, y=y_idx)
        else:
            ds_sliced = ds.sel(lat=lat, lon=lon, method='nearest')
            
        # Select surface level immediately if depth dimensions exist
        for depth_dim in ['deptht', 'depthu', 'depthv']:
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
        
        # Open datasets, chunks={'time_counter': 1} or 'auto' prevents Dask memory bloat
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
                'mld': 'somxl010'
            }
            
            nc_var_name = nemo_var_map[param_name]
            if nc_var_name not in point_ds.data_vars:
                raise ValueError(f"Could not find variable '{nc_var_name}'")
            
            # Convert to series triggers the actual computed data loading, 
            # but now it's only loading a 1x1 pixel time series!
            series = point_ds[nc_var_name].to_series()
            
            if isinstance(series.index, pd.MultiIndex):
                # Clean up any residual index levels leftover from squeezed depth variables
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
    import gc  # <--- Import garbage collection module
    
    ensemble_st = 38
    ensemble_end =40
    model_types = ["SSP370"]#["SSP370","HIST2"]
    retries = 5
    for i in range(ensemble_st-1,ensemble_end,1):
        for model_type in model_types:
            retry = 0
            while retry < retries:
                print(f"getting {model_type} for ensemble {i+1} on retry {retry}")
                DATA_DIRECTORY = f"/gws/ssde/j25b/canari/shared/large-ensemble/priority/{model_type}/{i+1}/OCN/yearly/"
                TARGET_LAT = 60.819
                TARGET_LON = -56.506
                
                OUTPUT_FILE = f"ensembles/CANARI_{model_type}_{i+1}_LON_{TARGET_LON}_LAT_{TARGET_LAT}.csv"
                
                if not os.path.exists(DATA_DIRECTORY):
                    print(f"Directory '{DATA_DIRECTORY}' not found. Please verify your base path.")
                    break # Break the retry loop if the directory straight up doesn't exist
                else:
                    try:
                        extract_location_timeseries(
                            data_dir=DATA_DIRECTORY,
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
            
            # ─── OPTIMIZATION FOR LONG LOOPS ──────────────────────────────────
            # Force Python to release unreferenced memory chunks before starting 
            # the next ensemble iteration
            gc.collect() 
            # ──────────────────────────────────────────────────────────────────