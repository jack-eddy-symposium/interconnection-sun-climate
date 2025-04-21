import os
# Selects GPU device "0". Ensure this is the intended device.
# This needs to be set BEFORE importing jax/numpyro in worker processes too.
# Multiprocessing usually handles environment variable inheritance, but explicit checks can be useful.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import numpy as np
import pandas as pd
import xarray as xr
import time # For basic timing

from tqdm import tqdm # Progress bars for loops

from jax import random
import dill # For saving complex Python objects (like MCMC results)
import logging

import jax
# Enable 64-bit precision in JAX. Important for numerical stability.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpyro
# Enable 64-bit precision in NumPyro.
numpyro.enable_x64()
import numpyro.distributions as dist
# Import NestedSampler for Bayesian inference and model comparison.
from numpyro.contrib.nested_sampling import NestedSampler

# --- Multiprocessing ---
import multiprocessing
from functools import partial # Useful for passing fixed arguments

# --- Logging Setup ---
# Configure logging (same as before)
log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
log_level = logging.INFO
LOG = logging.getLogger(__name__)
LOG.setLevel(log_level)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
handler.setFormatter(log_format)
LOG.addHandler(handler)

logger = logging.getLogger("jax")
logger.setLevel(logging.WARNING)
# Note: Logging from multiple processes to stdout can get interleaved.
# For cleaner logs, consider logging to separate files per worker or using a QueueHandler.

# --- Configuration Constants ---
DATA_SMOOTHING = 30
# --- !!! IMPORTANT: Update DATA_LABEL for precipitation !!! ---
DATA_LABEL = "precipitation" # Changed from "air_temperature"
COARSEN_LEVELS = [8, 2]
CONSTRUCTOR_KWARGS = {
    "max_samples": 40000,
    "parameter_estimation": True,
    "verbose": False, # Keep verbose False, especially with many parallel runs
}

# --- IMPORTANT: Parallelism Configuration ---
# Number of worker processes to run in parallel.
# START SMALL (e.g., 2-4) and monitor GPU memory/utilization (nvidia-smi).
# Increase gradually based on your GPU's capacity.
# Too many workers will lead to OOM errors or slow performance.
NUM_WORKERS = 2 # <--- ADJUST THIS BASED ON YOUR GPU

DATA_INPUT_FILE = (
    "/d0/amunozj/precipitation_data/access/gpcp_v02r03.nc"
)
FIGURE_FOLDER = "/d0/amunozj/git_repos/interconnection-sun-climate/outputs/figures"
# --- !!! IMPORTANT: Update MCMC_OUT_FOLDER for precipitation !!! ---
MCMC_OUT_FOLDER = (
    # Changed from "air_temperature" and "nested_sampling_negative_shift"
    f"/d0/amunozj/git_repos/interconnection-sun-climate/outputs/nested_sampling_negative_shift/{DATA_LABEL}"
)

# Create output directories if they don't exist (main process)
if not os.path.exists(FIGURE_FOLDER):
    os.makedirs(FIGURE_FOLDER) # Use makedirs

if not os.path.exists(MCMC_OUT_FOLDER):
    os.makedirs(MCMC_OUT_FOLDER)


## Define model
# --- !!! IMPORTANT: Review model priors/structure for precipitation !!! ---
# The priors (offset, sigma, amplitudes) were likely set for air temperature (Kelvin).
# They might need significant adjustment for precipitation (e.g., mm/day).
# Consider using a different likelihood distribution if appropriate (e.g., Gamma, LogNormal for positive values).
def single_lat_lon_model(
    time_coords, # Pass time coordinates explicitly
    target_variable=None,
    co2=False,
    seasonal=False,
    oni_norm=None, # Pass normalized data directly
    oni_time=None,
    cr_norm=None,  # Pass normalized data directly
    cr_time=None,
):
    """
    NumPyro model. NOTE: Priors might need adjustment for precipitation data.
    Consider likelihood distribution (e.g., Gamma, LogNormal) if Normal is inappropriate.
    """

    # Convert time to years since the start of the data period.
    min_time_val = np.min(time_coords) # Use passed time_coords (already numpy)
    orbits = ((time_coords - min_time_val) / np.timedelta64(1, "D") / 365.256)

    # --- Priors (REVIEW THESE FOR PRECIPITATION) ---
    # Sigma: Prior scale might be too large for precip (mm/day)
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0)) # Reduced scale guess
    # Offset: Prior mean/std likely wrong for precip. Maybe center around mean precip?
    # Use a wider prior initially if unsure.
    offset = numpyro.sample("offset", dist.Normal(2.0, 2.0)) # Guessed center/scale
    target_variable_estimate = offset

    # --- Model Components (REVIEW AMPLITUDES FOR PRECIPITATION) ---
    if seasonal:
        seasonal_shift = numpyro.sample("seasonal_shift", dist.Normal(0.0, 1.0))
        # Seasonal Amplitude: Prior scale might be too large for precip
        seasonal_amplitude = numpyro.sample(
            "seasonal_amplitude", dist.Normal(1.0, 2.0) # Reduced scale guess
        )
        seasonal_exponent = numpyro.sample("seasonal_exponent", dist.HalfNormal(2.0))
        sin_orbits = jnp.sin((orbits - seasonal_shift) * 2 * jnp.pi)
        target_variable_estimate = (
            target_variable_estimate
            + seasonal_amplitude
            * jnp.sign(sin_orbits)
            * jnp.pow(jnp.abs(sin_orbits), seasonal_exponent)
        )

    if co2:
        # CO2 Gain: Prior scale might be too large for precip trend
        co2_gain = numpyro.sample("co2_gain", dist.Normal(0.0, 2.0)) # Centered at 0, reduced scale
        target_variable_estimate = (
            target_variable_estimate + orbits / jnp.max(orbits) * co2_gain
        )

    if oni_norm is not None:
        # ONI Gain: Prior scale might be okay, but check units
        oni_gain = numpyro.sample("oni_gain", dist.Normal(0.0,  1.0))
        oni_shift = numpyro.sample("oni_shift", dist.Normal(0.0, 0.5))
        # Calculate ONI orbits relative to the start of the target variable time
        orbits_oni = ((oni_time - min_time_val) / np.timedelta64(1, "D") / 365.256)
        target_variable_estimate = target_variable_estimate + oni_gain * jnp.interp(
            orbits - jnp.abs(oni_shift), orbits_oni, oni_norm
        )

    if cr_norm is not None:
        # CR Gain: Prior scale might be okay, but check units
        cr_gain = numpyro.sample("cr_gain", dist.Normal(0.0,  1.0))
        cr_shift = numpyro.sample("cr_shift", dist.Normal(0.0, 1.0))
        # Calculate CR orbits relative to the start of the target variable time
        orbits_cr = ((cr_time - min_time_val) / np.timedelta64(1, "D") / 365.256)
        target_variable_estimate = target_variable_estimate + cr_gain * jnp.interp(
            orbits - jnp.abs(cr_shift), orbits_cr, cr_norm
        )

    # --- Likelihood (REVIEW FOR PRECIPITATION) ---
    # Precipitation is often non-negative and skewed. Normal might not be ideal.
    # Consider dist.Gamma or dist.LogNormal if target_variable > 0.
    # If target_variable can be zero, a Zero-Inflated model might be needed.
    # Using Normal for now, but this is a critical point for review.
    with numpyro.plate("times", target_variable.shape[0]):
        numpyro.sample(
            "obs", dist.Normal(target_variable_estimate, sigma), obs=target_variable
        )

# --- Worker Function ---
# This function performs the analysis for a single task (lat/lon/config)
def run_single_task(args):
    """
    Worker function to run Nested Sampling for one specific task.

    Args:
        args (tuple): A tuple containing all necessary arguments:
            (coarsen_level, lat_index, lon_index, lat, lon, config,
             time_coords_np, target_variable_np,
             oni_norm_np, oni_time_np, cr_norm_np, cr_time_np,
             base_output_folder, prng_key, data_label) # Added data_label
    """
    # Unpack arguments
    (coarsen_level, lat_index, lon_index, lat, lon, config,
     time_coords_np, target_variable_np,
     oni_norm_np, oni_time_np, cr_norm_np, cr_time_np,
     base_output_folder, prng_key, data_label) = args # Added data_label

    # Check for NaN values in target variable
    if np.isnan(target_variable_np).any():
        return f"Skipped NaN: Lat {lat:.2f}, Lon {lon:.2f}"

    # Construct output filename using data_label
    output_file = os.path.join(
        base_output_folder,
        f"{data_label}_smoothing{DATA_SMOOTHING}_level{coarsen_level}_{config['name']}_lat_{lat_index}_lon_{lon_index}.pkl",
    )

    # Check if results already exist
    if os.path.isfile(output_file):
        return f"Skipped Existing: {os.path.basename(output_file)}"

    # Prepare model arguments based on config
    model_kwargs = {
        "time_coords": time_coords_np,
        "target_variable": target_variable_np,
        "co2": config['co2'],
        "seasonal": config['seasonal'],
        "oni_norm": oni_norm_np if config['oni'] else None,
        "oni_time": oni_time_np if config['oni'] else None,
        "cr_norm": cr_norm_np if config['cr'] else None,
        "cr_time": cr_time_np if config['cr'] else None,
    }

    try:
        # Initialize NestedSampler
        ns = NestedSampler(
            single_lat_lon_model, constructor_kwargs=CONSTRUCTOR_KWARGS
        )

        # Split PRNG key for this specific run
        run_key, sample_key = random.split(prng_key)

        # Run the sampler
        ns.run(run_key, **model_kwargs)

        # Get posterior samples (draws samples *from* the posterior distribution defined by the nested sampling run)
        output_dictionary = ns.get_samples(sample_key, num_samples=5000)
        output_dictionary["log_Z_mean"] = ns._results.log_Z_mean

        # Save results using dill
        with open(output_file, "wb") as f:
            dill.dump(output_dictionary, f)

        # --- ADDED LINE: Attempt to clear JAX caches ---
        jax.clear_caches()
        # --- END ADDED LINE ---

        return f"Completed: {os.path.basename(output_file)}"

    except Exception as e:
        # --- ADDED LINE: Attempt cache clearing even on error ---
        # This might help if the error itself left things in a bad state,
        # though it won't fix the root cause of the error.
        jax.clear_caches()
        # --- END ADDED LINE ---
        LOG.error(f"Worker error for {config['name']} at Lat {lat:.2f}, Lon {lon:.2f}: {e}", exc_info=False)
        return f"Error: Lat {lat:.2f}, Lon {lon:.2f} - {e}"


# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Set Multiprocessing Start Method to 'spawn' ---
    # This is crucial for CUDA compatibility with multiprocessing on Linux
    try:
        multiprocessing.set_start_method('spawn', force=True)
        LOG.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        # Might happen if context is already set (e.g., in interactive session)
        LOG.warning("Multiprocessing context already set. Assuming 'spawn' or compatible.")
        pass # Or check multiprocessing.get_start_method()

    LOG.info(f"Starting Bayesian analysis for {DATA_LABEL}.") # Use DATA_LABEL
    LOG.info(f"Using {NUM_WORKERS} parallel workers.")
    LOG.info(f"Using multiprocessing start method: {multiprocessing.get_start_method()}") # Confirm method

    start_time = time.time()

    # --- Load Data (Main Process) ---
    LOG.info(f"Loading {DATA_LABEL} data from: {DATA_INPUT_FILE}") # Use DATA_LABEL
    geo_data = xr.open_dataset(DATA_INPUT_FILE)
    # --- !!! Select the correct variable for precipitation !!! ---
    # Assuming the variable name is 'precip'. Change if necessary.
    precip_variable_name = 'precip'
    if precip_variable_name not in geo_data.variables:
        LOG.error(f"Variable '{precip_variable_name}' not found in {DATA_INPUT_FILE}. Check variable name.")
        sys.exit(1) # Exit if variable not found

    # --- Load and Prepare ONI Data ---
    oni_file = "/d0/amunozj/git_repos/interconnection-sun-climate/ENI/detrend.nino34.ascii.txt"
    LOG.info(f"Loading ONI data from: {oni_file}")
    df_oni = pd.read_table(oni_file, engine="c", sep="\s+")
    df_oni["DAY"] = 15
    df_oni["time"] = pd.to_datetime(df_oni[["YEAR", "MONTH", "DAY"]])
    df_oni = df_oni.set_index("time", drop=True).loc[:, ["TOTAL", "ClimAdjust"]]
    oni_data = xr.Dataset.from_dataframe(df_oni)
    oni_data["oni"] = oni_data.TOTAL - oni_data.ClimAdjust
    oni_data["normalized"] = (oni_data.oni - oni_data.oni.mean()) / oni_data.oni.std()
    # Extract numpy arrays for passing to workers
    oni_norm_np = oni_data.normalized.values
    oni_time_np = oni_data.time.values
    LOG.info("ONI data loaded and normalized.")

    # --- Load and Prepare CR Data ---
    # Using relative path assuming script is run from causality_analysis directory
    crDataPath = "/d0/amunozj/git_repos/interconnection-sun-climate/data/cosmic_rays/OULU_1964_05_01 _00_00_2024_10_28 _23_30.csv"
    LOG.info(f"Loading Cosmic Ray data from: {crDataPath}")
    try:
        cr_df = pd.read_csv(crDataPath)
    except FileNotFoundError:
        LOG.error(f"Cosmic Ray data file not found at: {crDataPath}")
        LOG.error("Ensure the relative path is correct based on where you run the script.")
        sys.exit(1)
    cr_df["time"] = (
        pd.to_datetime(cr_df["Timestamp"]).dt.tz_localize(None).astype("datetime64[ns]")
    )
    cr_df = cr_df.set_index("time", drop=True)
    cr_data = xr.Dataset.from_dataframe(cr_df)
    # --- !!! Check CR variable name if needed !!! ---
    cr_var_name = "CorrectedCountRate"
    if cr_var_name not in cr_data:
         LOG.error(f"Variable '{cr_var_name}' not found in CR data. Check CSV header.")
         sys.exit(1)
    cr_data["normalized"] = (
        cr_data[cr_var_name] - cr_data[cr_var_name].mean()
    ) / cr_data[cr_var_name].std()
    # Extract numpy arrays
    cr_norm_np = cr_data.normalized.values
    cr_time_np = cr_data.time.values
    LOG.info("Cosmic Ray data loaded and normalized.")

    # --- Define Model Configurations ---
    model_configs = [
        {"name": "co2_season", "co2": True, "seasonal": True, "oni": False, "cr": False},
        {"name": "co2_season_oni", "co2": True, "seasonal": True, "oni": True, "cr": False},
        {"name": "co2_season_cr", "co2": True, "seasonal": True, "oni": False, "cr": True},
        {"name": "co2_season_oni_cr", "co2": True, "seasonal": True, "oni": True, "cr": True},
    ]

    # --- Master PRNG Key ---
    master_key = random.PRNGKey(0) # Base key for reproducibility

    # --- Loop Through Coarsening Levels (Sequential) ---
    for coarsen_level in COARSEN_LEVELS:
        LOG.info(f"Processing coarsening level: {coarsen_level}")

        # Apply coarsening
        LOG.info(f"Coarsening data with factor {coarsen_level} (space) and {DATA_SMOOTHING} (time)...")
        # --- !!! Use correct dimension names for precipitation data !!! ---
        # Assuming 'longitude', 'latitude'. Change if necessary.
        try:
            geo_data_coarse = geo_data.coarsen(
                longitude=coarsen_level, latitude=coarsen_level, time=DATA_SMOOTHING, boundary="pad"
            ).mean()
        except ValueError as e:
             LOG.error(f"Error during coarsening. Check dimension names ('longitude', 'latitude', 'time'): {e}")
             sys.exit(1)

        # Extract time coordinates once
        time_coords_np = geo_data_coarse.time.values
        LOG.info(f"Coarsened data shape: {geo_data_coarse[precip_variable_name].shape}")

        # Create output directory for this level
        level_output_folder = os.path.join(MCMC_OUT_FOLDER, f"level_{coarsen_level}")
        os.makedirs(level_output_folder, exist_ok=True) # Use makedirs with exist_ok=True

        # --- Prepare Tasks for Parallel Execution ---
        tasks = []
        LOG.info("Generating tasks for parallel processing...")
        # --- !!! Use correct dimension names !!! ---
        lon_dim_name = 'longitude'
        lat_dim_name = 'latitude'
        num_lon = geo_data_coarse[lon_dim_name].shape[0]
        num_lat = geo_data_coarse[lat_dim_name].shape[0]

        pbar_tasks = tqdm(total=num_lon * num_lat * len(model_configs), desc="Task Generation")

        # Split the master key for all tasks at this level
        num_total_tasks = num_lon * num_lat * len(model_configs)
        # Handle case where num_total_tasks might be 0
        if num_total_tasks == 0:
            LOG.warning("No tasks to generate (check data dimensions).")
            continue # Skip to next coarsening level if no tasks

        task_keys = random.split(master_key, num_total_tasks + 1) # Need +1 to advance master_key
        master_key = task_keys[0] # Advance the master state
        task_keys = task_keys[1:] # Use the rest for the current tasks
        key_idx = 0

        for lon_index in range(num_lon):
            lon = geo_data_coarse[lon_dim_name].values[lon_index]
            for lat_index in range(num_lat):
                lat = geo_data_coarse[lat_dim_name].values[lat_index]

                # Extract target variable (as numpy array)
                # --- !!! Use correct dimension names and variable name !!! ---
                target_variable_np = geo_data_coarse[precip_variable_name].sel({lat_dim_name: lat, lon_dim_name: lon}).values

                # Skip NaN locations early
                if np.isnan(target_variable_np).any():
                    # LOG.warning(f"Skipping Lat {lat:.2f}, Lon {lon:.2f} at task generation due to NaN values.") # Can be noisy
                    key_idx += len(model_configs) # Increment key index
                    pbar_tasks.update(len(model_configs))
                    continue

                for config in model_configs:
                    # Check if enough keys remain (debugging)
                    if key_idx >= len(task_keys):
                         LOG.error(f"Key index {key_idx} out of bounds ({len(task_keys)} keys available). Mismatch in task count?")
                         # This shouldn't happen if num_total_tasks calculation is correct
                         break # Avoid crashing

                    # Arguments for the worker function
                    task_args = (
                        coarsen_level, lat_index, lon_index, lat, lon, config,
                        time_coords_np, target_variable_np,
                        oni_norm_np, oni_time_np, cr_norm_np, cr_time_np,
                        level_output_folder,
                        task_keys[key_idx], # Assign a unique key
                        DATA_LABEL # Pass data label for filename
                    )
                    tasks.append(task_args)
                    key_idx += 1
                    pbar_tasks.update(1)
            if key_idx >= len(task_keys) and lon_index < num_lon -1: # Break outer loop if keys run out early
                 break
        pbar_tasks.close()
        LOG.info(f"Generated {len(tasks)} tasks for level {coarsen_level}.")

        # --- Execute Tasks in Parallel ---
        if tasks:
            LOG.info(f"Starting parallel execution with {NUM_WORKERS} workers...")
            with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
                results = list(tqdm(pool.imap_unordered(run_single_task, tasks), total=len(tasks), desc=f"Level {coarsen_level} Processing"))

            success_count = sum(1 for r in results if r.startswith("Completed") or r.startswith("Skipped"))
            error_count = sum(1 for r in results if r.startswith("Error"))
            LOG.info(f"Level {coarsen_level} finished. Success/Skipped: {success_count}, Errors: {error_count}")
        else:
            LOG.info(f"No valid tasks to run for level {coarsen_level}.")


    end_time = time.time()
    LOG.info(f"Bayesian analysis finished. Total time: {end_time - start_time:.2f} seconds.")

