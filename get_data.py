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
    
    # Using '**' allows glob to look recursively down into any subfolders (like /2024/, /2025/)
    param_files = {
            'temperature': os.path.join(data_dir, "**", "*grid_T_votemper.nc"),
            'salinity':    os.path.join(data_dir, "**", "*grid_T_vosaline.nc"),
            'ssh':         os.path.join(data_dir, "**", "*grid_T_sossheig.nc"), # or *grid_T_soshfldo.nc
            'u_velocity':  os.path.join(data_dir, "**", "*grid_U_vozocrtx.nc"),
            'v_velocity':  os.path.join(data_dir, "**", "*grid_V_vomecrty.nc"),
            'mld':  os.path.join(data_dir, "**", "*grid_T_somxl010.nc"),
        }
    
    extracted_series = {}
    
    for param_name, glob_pattern in param_files.items():
        print(f"\nProcessing parameter: {param_name}...")
        
        # Enable recursive matching to search through the nested year folders
        files = sorted(glob.glob(glob_pattern, recursive=True))
        if not files:
            raise FileNotFoundError(f"No NetCDF files found matching pattern: {glob_pattern}")
            
        print(f"  Found {len(files)} total monthly files across all year folders.")
        print(f"  First file found: {os.path.basename(files[0])}")
        print(f"  Last file found:  {os.path.basename(files[-1])}")
        print("  Stitching files along timeline...")
        
        # Lazy load and combine all files across years by coordinate tracking
        with xr.open_mfdataset(files, combine='by_coords', parallel=True) as ds:
            
            # 2. Map target lat/lon to nearest x, y grid indices
            # We look for nav_lat/nav_lon, which are standard in NEMO grid files
            if 'nav_lat' in ds.coords and 'nav_lon' in ds.coords:
                # Calculate absolute difference to target
                dist = (ds.nav_lat - lat)**2 + (ds.nav_lon - lon)**2
                # Find the index of the minimum distance
                y_idx, x_idx = np.unravel_index(dist.argmin(), dist.shape)
                
                # Select using the identified indices
                point_ds = ds.isel(x=x_idx, y=y_idx)
            else:
                # Fallback if standard lat/lon dimensions ARE present
                point_ds = ds.sel(lat=lat, lon=lon, method='nearest')

            # 3. Handle the time dimension (rename 'time_counter' if necessary)
            if 'time_counter' in point_ds.dims:
                point_ds = point_ds.rename({'time_counter': 'time'})
            # --- Robust NEMO Variable Selection ---
            # Map of parameter name to the specific NEMO variable key
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
                raise ValueError(f"Could not find variable '{nc_var_name}' in file. Available: {list(point_ds.data_vars)}")
            
            print(f"  Extracting variable '{nc_var_name}'...")
            
            # Select the variable
            da = point_ds[nc_var_name]
            
            # --- Handle depth dimension safely ---
            # If the variable has a depth dimension, slice it at the first level (surface)
            if 'deptht' in da.dims:
                da = da.isel(deptht=0)
            elif 'depthu' in da.dims:
                da = da.isel(depthu=0)
            elif 'depthv' in da.dims:
                da = da.isel(depthv=0)
            
            # Convert to series
            series = da.to_series()
            
            # Drop structural leftover depth dimensions if present
            if isinstance(series.index, pd.MultiIndex):
                series = series.xs(series.index.get_level_values('depth')[0], level='depth')
            
            # Check for duplicate timestamps caused by overlapping historical files or restarts
            if series.index.duplicated().any():
                print("  ⚠️ Found duplicate timestamps. Retaining the first entry per timestamp.")
                series = series[~series.index.duplicated(keep='first')]
                
            extracted_series[param_name] = series

    # Merge variables by their shared timestamps
    print("\nMerging all variables into combined dataset...")
    combined_df = pd.DataFrame(extracted_series)
    combined_df = combined_df.sort_index()
    
    print(f"Final Combined Timeline: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"Total rows extracted: {len(combined_df)}")
    
    # Save output to file
    combined_df.to_csv(output_csv_path, index_label='time')
    print(f"🎉 Success! Suitable multi-year dataset exported to: {output_csv_path}")


if __name__ == "__main__":
    # ─── CONFIGURATIONS ──────────────────────────────────────────────────────
    # Base folder path containing your subfolders (e.g., /2024/, /2025/, etc.)
    model_type = "SSP370"
    DATA_DIRECTORY = f"/gws/ssde/j25b/canari/shared/large-ensemble/priority/{model_type}/2/OCN/yearly/"
    TARGET_LAT = 55.939
    TARGET_LON = -49.101
    
    OUTPUT_FILE = f"CANARI_{model_type}_LON_{TARGET_LON}_LAT_{TARGET_LAT}.csv"
    # ─────────────────────────────────────────────────────────────────────────
    
    if not os.path.exists(DATA_DIRECTORY):
        print(f"Directory '{DATA_DIRECTORY}' not found. Please verify your base path.")
    else:
        try:
            extract_location_timeseries(
                data_dir=DATA_DIRECTORY,
                lat=TARGET_LAT,
                lon=TARGET_LON,
                output_csv_path=OUTPUT_FILE
            )
        except Exception as e:
            print(f"\n❌ Extraction stopped due to error: {e}")