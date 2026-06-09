import os
import glob
import xarray as xr
import pandas as pd
import numpy as np

def extract_grid_timeseries(
    data_dir: str, 
    lat: float, 
    lon: float, 
    output_nc_path: str
):
    """
    Reads segmented monthly NetCDF files organized inside nested year subfolders,
    extracts a 3x3 grid centered around the target location, joins parameters, 
    and saves a combined NetCDF file.
    """
    print(f"--- Starting Multi-Year Extraction for 3x3 Grid Centered at (Lat: {lat}, Lon: {lon}) ---")
    
    param_files = {
        'temperature': os.path.join(data_dir, "**", "*grid_T_votemper.nc"),
        'salinity':    os.path.join(data_dir, "**", "*grid_T_vosaline.nc"),
        'ssh':         os.path.join(data_dir, "**", "*grid_T_sossheig.nc"), 
        'u_velocity':  os.path.join(data_dir, "**", "*grid_U_vozocrtx.nc"),
        'v_velocity':  os.path.join(data_dir, "**", "*grid_V_vomecrty.nc"),
        'mld':         os.path.join(data_dir, "**", "*grid_T_somxl010.nc"),
    }
    
    extracted_datasets = []
    
    # ─── OPTIMIZATION 1: PRE-CALCULATE SPATIAL INDICES ───────────────────────
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
            print(f"  Mapped to center grid indices -> x: {x_idx}, y: {y_idx}")
        else:
            has_nav_coords = False
            # Fallback for standard rectilinear datasets (lat/lon dimensions)
            lat_dim = 'lat' if 'lat' in sample_ds.dims else 'latitude'
            lon_dim = 'lon' if 'lon' in sample_ds.dims else 'longitude'
            y_idx = int(np.abs(sample_ds[lat_dim] - lat).argmin())
            x_idx = int(np.abs(sample_ds[lon_dim] - lon).argmin())
            print(f"  Standard dimensions ({lat_dim}/{lon_dim}) detected. Center indices -> y: {y_idx}, x: {x_idx}")

    # ─── OPTIMIZATION 2: DEFINE A PREPROCESS FILTER FOR 3x3 SLICING ──────────
    def slice_grid_on_load(ds):
        """Slices the dataset to a 3x3 bounding box centered on target indices."""
        # Python slices are exclusive of the stop index; slice(idx-1, idx+2) extracts: idx-1, idx, idx+1
        if has_nav_coords:
            ds_sliced = ds.isel(
                y=slice(max(0, y_idx - 1), y_idx + 2), 
                x=slice(max(0, x_idx - 1), x_idx + 2)
            )
        else:
            ds_sliced = ds.isel({
                lat_dim: slice(max(0, y_idx - 1), y_idx + 2),
                lon_dim: slice(max(0, x_idx - 1), x_idx + 2)
            })
            
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
        
        with xr.open_mfdataset(
            files, 
            combine='by_coords', 
            parallel=True, 
            preprocess=slice_grid_on_load,
            chunks={'time_counter': 1, 'time': 1} 
        ) as grid_ds:
            
            if 'time_counter' in grid_ds.dims:
                grid_ds = grid_ds.rename({'time_counter': 'time'})
            
            nemo_var_map = {
                'temperature': 'votemper',
                'salinity': 'vosaline',
                'ssh': 'sossheig',
                'u_velocity': 'vozocrtx',
                'v_velocity': 'vomecrty',
                'mld': 'somxl010'
            }
            
            nc_var_name = nemo_var_map[param_name]
            if nc_var_name not in grid_ds.data_vars:
                raise ValueError(f"Could not find variable '{nc_var_name}'")
            
            # Keep only our specific target data variable
            var_ds = grid_ds[[nc_var_name]]
            
            # Deduplicate timestamps along the time coordinate if necessary
            if len(np.unique(var_ds.time)) < len(var_ds.time):
                print("  ⚠️ Found duplicate timestamps. Retaining the first entry per timestamp.")
                _, index = np.unique(var_ds['time'], return_index=True)
                var_ds = var_ds.isel(time=index)
            
            # .load() forces the computation of the tiny 3x3 matrix into memory 
            # so it stays accessible outside the with-context file closure.
            var_ds.load()
            extracted_datasets.append(var_ds)

    # Merge variables along their shared dimensions
    print("\nMerging all variables into combined dataset...")
    # compat='override' is vital here to circumvent minor coordinate offsets across staggered grids
    combined_ds = xr.merge(extracted_datasets, compat='override')
    combined_ds = combined_ds.sortby('time')
    
    print(f"Final Combined Timeline: {combined_ds.time.min().values} to {combined_ds.time.max().values}")
    
    # Save output to a NetCDF file
    combined_ds.to_netcdf(output_nc_path)
    print(f"🎉 Success! Exported NetCDF to: {output_nc_path}")
    
if __name__ == "__main__":
    import gc 
    
    ensemble_st = 1
    ensemble_end = 40
    model_types = ["SSP370"]
    retries = 5
    for i in range(ensemble_st-1, ensemble_end, 1):
        for model_type in model_types:
            retry = 0
            while retry < retries:
                print(f"getting {model_type} for ensemble {i+1} on retry {retry}")
                DATA_DIRECTORY = f"/gws/ssde/j25b/canari/shared/large-ensemble/priority/{model_type}/{i+1}/OCN/yearly/"
                TARGET_LAT = 60.819
                TARGET_LON = -56.506
                
                # Updated the filename string to output `.nc` instead of `.csv`
                OUTPUT_FILE = f"3x3ensembles/CANARI_{model_type}_{i+1}_LON_{TARGET_LON}_LAT_{TARGET_LAT}.nc"
                
                if not os.path.exists(DATA_DIRECTORY):
                    print(f"Directory '{DATA_DIRECTORY}' not found. Please verify your base path.")
                    break 
                else:
                    try:
                        extract_grid_timeseries(
                            data_dir=DATA_DIRECTORY,
                            lat=TARGET_LAT,
                            lon=TARGET_LON,
                            output_nc_path=OUTPUT_FILE
                        )
                        retry = retries
                    except Exception as e:
                        print(f"\n❌ Extraction stopped due to error: {e}")
                        retry = retry + 1
                        import time
                        time.sleep(600) 
            
            gc.collect()