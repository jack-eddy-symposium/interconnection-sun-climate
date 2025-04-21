import os
# GPU selection handled by CUDA_VISIBLE_DEVICES environment variable
# Set it *before* running the script, e.g.,
# CUDA_VISIBLE_DEVICES=1 python flexible_bayesian_ns.py --config config_precip.yaml

import sys
import numpy as np
import pandas as pd
import xarray as xr
import time
import argparse
import yaml     
import multiprocessing
from tqdm import tqdm
import dill
import logging

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random

import numpyro
numpyro.enable_x64()
import numpyro.distributions as dist
from numpyro.contrib.nested_sampling import NestedSampler

# --- Logging Setup ---
log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
log_level = logging.INFO
LOG = logging.getLogger(__name__)
LOG.setLevel(log_level)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
handler.setFormatter(log_format)
LOG.addHandler(handler)

logger = logging.getLogger("jax._src.xla_bridge")
logger.setLevel(logging.ERROR)

def single_lat_lon_model(
    time_coords,
    target_variable=None,
    co2=False,
    seasonal=False,
    oni_norm=None,
    oni_time=None,
    cr_norm=None,
    cr_time=None,
    # --- Prior Parameters ---
    sigma_scale: float = 1.0,
    offset_mean: float = 0.0,
    offset_std: float = 1.0,
    seasonal_amp_mean: float = 0.0,
    seasonal_amp_std: float = 1.0,
    seasonal_exp_scale: float = 2.0,
    co2_gain_mean: float = 0.0,
    co2_gain_std: float = 1.0,
    oni_gain_mean: float = 0.0,
    oni_gain_std: float = 1.0,
    oni_shift_mean: float = 0.0,
    oni_shift_std: float = 0.5,
    cr_gain_mean: float = 0.0,
    cr_gain_std: float = 1.0,
    cr_shift_mean: float = 0.0,
    cr_shift_std: float = 1.0,
):
    """
    NumPyro model with parameterized priors.
    WARNING: The choice of likelihood (currently Normal) and the model terms
             (seasonal, co2, oni, cr) might need adjustment for different variables.
    """
    min_time_val = np.min(time_coords)
    orbits = ((time_coords - min_time_val) / np.timedelta64(1, "D") / 365.256)
    max_orbits = jnp.max(orbits) # Calculate once

    # --- Offset priors ---
    sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_scale))
    offset = numpyro.sample("offset", dist.Normal(offset_mean, offset_std))
    target_variable_estimate = offset

    # --- Model Components ---
    if seasonal:
        seasonal_shift = numpyro.sample("seasonal_shift", dist.Normal(0.0, 1.0))
        seasonal_amplitude = numpyro.sample(
            "seasonal_amplitude", dist.Normal(seasonal_amp_mean, seasonal_amp_std)
        )
        seasonal_exponent = numpyro.sample("seasonal_exponent", dist.HalfNormal(seasonal_exp_scale))
        sin_orbits = jnp.sin((orbits - seasonal_shift) * 2 * jnp.pi)
        target_variable_estimate = (
            target_variable_estimate
            + seasonal_amplitude
            * jnp.sign(sin_orbits)
            * jnp.pow(jnp.abs(sin_orbits), seasonal_exponent)
        )

    if co2:
        co2_gain = numpyro.sample("co2_gain", dist.Normal(co2_gain_mean, co2_gain_std))
        target_variable_estimate = (
            target_variable_estimate + orbits / max_orbits * co2_gain
        )

    if oni_norm is not None and oni_time is not None:
        oni_gain = numpyro.sample("oni_gain", dist.Normal(oni_gain_mean, oni_gain_std))
        oni_shift = numpyro.sample("oni_shift", dist.Normal(oni_shift_mean, oni_shift_std))
        orbits_oni = ((oni_time - min_time_val) / np.timedelta64(1, "D") / 365.256)
        target_variable_estimate = target_variable_estimate + oni_gain * jnp.interp(
            orbits - jnp.abs(oni_shift), orbits_oni, oni_norm
        )

    if cr_norm is not None and cr_time is not None:
        cr_gain = numpyro.sample("cr_gain", dist.Normal(cr_gain_mean, cr_gain_std))
        cr_shift = numpyro.sample("cr_shift", dist.Normal(cr_shift_mean, cr_shift_std))
        orbits_cr = ((cr_time - min_time_val) / np.timedelta64(1, "D") / 365.256)
        target_variable_estimate = target_variable_estimate + cr_gain * jnp.interp(
            orbits - jnp.abs(cr_shift), orbits_cr, cr_norm
        )

    # --- Likelihood ---
    with numpyro.plate("times", target_variable.shape[0]):
        numpyro.sample(
            "obs", dist.Normal(target_variable_estimate, sigma), obs=target_variable
        )


# --- Worker Function ---
def run_single_task(args_tuple):
    """
    Worker function to run Nested Sampling for one specific task.
    Receives config values directly instead of the full config dict.
    """
    # Unpack arguments (ensure order matches task_args creation)
    (coarsen_level, lat_index, lon_index, lat, lon, config_model, # Renamed config -> config_model
     time_coords_np, target_variable_np,
     oni_norm_np, oni_time_np, cr_norm_np, cr_time_np,
     base_output_folder, prng_key, data_label, smoothing, # Passed from main config
     prior_params, # Passed from main config
     constructor_kwargs, num_posterior_samples # Passed from main config
     ) = args_tuple

    # Check for NaN values in target variable
    if np.isnan(target_variable_np).any():
        return f"Skipped NaN: Lat {lat:.2f}, Lon {lon:.2f}"

    # Construct output filename
    output_file = os.path.join(
        base_output_folder,
        f"{data_label}_smoothing{smoothing}_level{coarsen_level}_{config_model['name']}_lat_{lat_index}_lon_{lon_index}.pkl",
    )

    # Check if results already exist
    if os.path.isfile(output_file):
        return f"Skipped Existing: {os.path.basename(output_file)}"

    # Prepare model arguments based on config_model and priors
    model_kwargs = {
        "time_coords": time_coords_np,
        "target_variable": target_variable_np,
        "co2": config_model['co2'],
        "seasonal": config_model['seasonal'],
        "oni_norm": oni_norm_np if config_model['oni'] else None,
        "oni_time": oni_time_np if config_model['oni'] else None,
        "cr_norm": cr_norm_np if config_model['cr'] else None,
        "cr_time": cr_time_np if config_model['cr'] else None,
        **prior_params # Unpack the prior parameters dictionary
    }

    try:
        # Initialize NestedSampler
        ns = NestedSampler(
            single_lat_lon_model, constructor_kwargs=constructor_kwargs
        )

        # Split PRNG key for this specific run
        run_key, sample_key = random.split(prng_key)

        # Run the sampler
        ns.run(run_key, **model_kwargs)

        # Get posterior samples
        output_dictionary = ns.get_samples(sample_key, num_samples=num_posterior_samples)
        output_dictionary["log_Z_mean"] = ns._results.log_Z_mean

        # Save results using dill
        with open(output_file, "wb") as f:
            dill.dump(output_dictionary, f)

        jax.clear_caches()
        return f"Completed: {os.path.basename(output_file)}"

    except Exception as e:
        jax.clear_caches()
        # Log the full traceback for debugging worker errors
        LOG.error(f"Worker error for {config_model['name']} at Lat {lat:.2f}, Lon {lon:.2f}: {e}", exc_info=True)
        return f"Error: Lat {lat:.2f}, Lon {lon:.2f} - {e}"


# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Argument Parser (Only for Config File Path) ---
    parser = argparse.ArgumentParser(description="Run Bayesian Nested Sampling analysis using a YAML config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # --- Load Configuration from YAML ---
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        LOG.info(f"Loaded configuration from: {args.config}")
    except FileNotFoundError:
        LOG.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        LOG.error(f"Error parsing YAML configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        LOG.error(f"An unexpected error occurred loading the config: {e}")
        sys.exit(1)

    # --- Validate Essential Config Sections/Keys ---
    required_sections = ['data', 'processing', 'execution', 'priors']
    for section in required_sections:
        if section not in config:
            LOG.error(f"Missing required section '{section}' in configuration file.")
            sys.exit(1)
    required_data_keys = ['label', 'input_file', 'var_name', 'dimensions']
    for key in required_data_keys:
         if key not in config['data']:
              LOG.error(f"Missing required key '{key}' in 'data' section.")
              sys.exit(1)
    required_dim_keys = ['lon', 'lat', 'time']
    if not isinstance(config['data']['dimensions'], dict) or \
       any(key not in config['data']['dimensions'] for key in required_dim_keys):
        LOG.error(f"Key 'data.dimensions' must be a dictionary containing keys: {required_dim_keys}")
        sys.exit(1)
    # Add more validation as needed, especially for prior keys

    # --- Extract Config Values ---
    data_label = config['data']['label']
    input_file = config['data']['input_file']
    var_name = config['data']['var_name']
    output_dir = config['execution']['output_dir']
    lon_dim = config['data']['dimensions']['lon']
    lat_dim = config['data']['dimensions']['lat']
    time_dim = config['data']['dimensions']['time']
    coarsen_levels = config['processing']['coarsen_levels']
    smoothing = config['processing']['smoothing']
    workers = config['execution']['workers']
    max_samples = config['execution']['nested_sampler']['max_samples']
    num_posterior_samples = config['execution']['nested_sampler']['num_posterior_samples']
    prior_params = config['priors'] # The whole dictionary

    # Optional external data paths and names
    oni_file_path = config.get('external_data', {}).get('oni', {}).get('file_path')
    cr_config = config.get('external_data', {}).get('cr', {})
    cr_file_path = cr_config.get('file_path')
    cr_var_name = cr_config.get('var_name', 'CorrectedCountRate') # Default if not specified

    # --- Set Multiprocessing Start Method ---
    try:
        multiprocessing.set_start_method('spawn', force=True)
        LOG.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        LOG.warning("Multiprocessing context already set. Assuming 'spawn' or compatible.")
        pass

    LOG.info(f"Starting Bayesian analysis for variable: {data_label}")
    LOG.info(f"Using {workers} parallel workers.")
    LOG.info(f"Input file: {input_file}")
    LOG.info(f"Variable name: {var_name}")
    LOG.info(f"Output directory: {output_dir}")
    LOG.info(f"Coarsening levels: {coarsen_levels}")
    LOG.info(f"Temporal smoothing: {smoothing}")
    LOG.info(f"Using multiprocessing start method: {multiprocessing.get_start_method()}")

    start_time = time.time()

    # --- Prepare NestedSampler Kwargs ---
    constructor_kwargs = {
        "max_samples": max_samples,
        "parameter_estimation": True,
        "verbose": False, # Keep verbose False for parallel runs
    }

    # --- Log Prior Parameters ---
    LOG.info(f"Using Prior Parameters: {prior_params}")
    LOG.warning("Ensure these prior parameters are appropriate for the scale and expected behavior of your variable!")

    # --- Load Main Data ---
    LOG.info(f"Loading {data_label} data from: {input_file}")
    try:
        geo_data = xr.open_dataset(input_file)
    except FileNotFoundError:
        LOG.error(f"Input data file not found: {input_file}")
        sys.exit(1)

    # Check if variable exists
    if var_name not in geo_data.variables:
        LOG.error(f"Variable '{var_name}' not found in {input_file}.")
        LOG.error(f"Available variables: {list(geo_data.variables.keys())}")
        sys.exit(1)
    # Check if dimensions exist
    for dim in [lon_dim, lat_dim, time_dim]:
         if dim not in geo_data.dims:
              LOG.error(f"Dimension '{dim}' not found in {input_file}.")
              LOG.error(f"Available dimensions: {list(geo_data.dims.keys())}")
              sys.exit(1)


    # --- Load and Prepare ONI Data ---
    oni_norm_np = None
    oni_time_np = None
    if oni_file_path:
        LOG.info(f"Loading ONI data from: {oni_file_path}")
        try:
            df_oni = pd.read_table(oni_file_path, engine="c", sep="\s+")
            df_oni["DAY"] = 15
            df_oni["time"] = pd.to_datetime(df_oni[["YEAR", "MONTH", "DAY"]])
            df_oni = df_oni.set_index("time", drop=True).loc[:, ["TOTAL", "ClimAdjust"]]
            oni_data = xr.Dataset.from_dataframe(df_oni)
            oni_data["oni"] = oni_data.TOTAL - oni_data.ClimAdjust
            oni_data["normalized"] = (oni_data.oni - oni_data.oni.mean()) / oni_data.oni.std()
            oni_norm_np = oni_data.normalized.values
            oni_time_np = oni_data.time.values
            LOG.info("ONI data loaded and normalized.")
        except FileNotFoundError:
            LOG.warning(f"ONI data file not found at {oni_file_path}. ONI component will not be used.")
        except Exception as e:
            LOG.error(f"Error loading or processing ONI data: {e}", exc_info=True)
    else:
        LOG.info("ONI file path not specified in config. ONI component will not be used.")


    # --- Load and Prepare CR Data ---
    cr_norm_np = None
    cr_time_np = None
    if cr_file_path:
        LOG.info(f"Loading Cosmic Ray data from: {cr_file_path}")
        try:
            cr_df = pd.read_csv(cr_file_path)
            cr_df["time"] = (
                pd.to_datetime(cr_df["Timestamp"]).dt.tz_localize(None).astype("datetime64[ns]")
            )
            cr_df = cr_df.set_index("time", drop=True)
            cr_data = xr.Dataset.from_dataframe(cr_df)
            if cr_var_name not in cr_data:
                 LOG.error(f"Variable '{cr_var_name}' (specified in config or default) not found in CR data file {cr_file_path}. Check CSV header.")
            else:
                cr_data["normalized"] = (
                    cr_data[cr_var_name] - cr_data[cr_var_name].mean()
                ) / cr_data[cr_var_name].std()
                cr_norm_np = cr_data.normalized.values
                cr_time_np = cr_data.time.values
                LOG.info("Cosmic Ray data loaded and normalized.")
        except FileNotFoundError:
            LOG.warning(f"Cosmic Ray data file not found at: {cr_file_path}. CR component will not be used.")
        except Exception as e:
            LOG.error(f"Error loading or processing CR data: {e}", exc_info=True)
    else:
        LOG.info("CR file path not specified in config. CR component will not be used.")


    # --- Define Model Configurations ---
    model_configs = [
        {"name": "co2_season", "co2": True, "seasonal": True, "oni": False, "cr": False},
        {"name": "co2_season_oni", "co2": True, "seasonal": True, "oni": True, "cr": False},
        {"name": "co2_season_cr", "co2": True, "seasonal": True, "oni": False, "cr": True},
        {"name": "co2_season_oni_cr", "co2": True, "seasonal": True, "oni": True, "cr": True},
    ]
    # Filter configs based on available ONI/CR data
    if oni_norm_np is None or oni_time_np is None:
        LOG.warning("Filtering model configurations: ONI data not available or not loaded.")
        model_configs = [cfg for cfg in model_configs if not cfg['oni']]
    if cr_norm_np is None or cr_time_np is None:
        LOG.warning("Filtering model configurations: CR data not available or not loaded.")
        model_configs = [cfg for cfg in model_configs if not cfg['cr']]

    if not model_configs:
         LOG.error("No valid model configurations remaining after checking ONI/CR data. Exiting.")
         sys.exit(1)
    LOG.info(f"Using model configurations: {[cfg['name'] for cfg in model_configs]}")


    # --- Master PRNG Key ---
    master_key = random.PRNGKey(0) # Base key for reproducibility

    # --- Create Base Output Directory ---
    # Specific output folder using the data label from config
    MCMC_OUT_FOLDER = os.path.join(output_dir, f"nested_sampling_{data_label}")
    os.makedirs(MCMC_OUT_FOLDER, exist_ok=True)
    LOG.info(f"Main output folder: {MCMC_OUT_FOLDER}")


    # --- Loop Through Coarsening Levels (Sequential) ---
    for coarsen_level in coarsen_levels:
        LOG.info(f"Processing coarsening level: {coarsen_level}")

        # Apply coarsening (using dimension names from config)
        LOG.info(f"Coarsening data with factor {coarsen_level} (space) and {smoothing} (time)...")
        try:
            coarsen_dims = {
                lon_dim: coarsen_level,
                lat_dim: coarsen_level,
                time_dim: smoothing
            }
            # Filter out dims not present or with factor 1
            coarsen_dims = {k: v for k, v in coarsen_dims.items() if k in geo_data.dims and v > 1}
            if not coarsen_dims:
                 LOG.warning(f"No coarsening applied for level {coarsen_level} / smoothing {smoothing} (factors <= 1 or dims not present). Using original data resolution for this level.")
                 geo_data_coarse = geo_data
            else:
                 geo_data_coarse = geo_data.coarsen(**coarsen_dims, boundary="pad").mean()

        except ValueError as e:
             LOG.error(f"Error during coarsening. Check dimension names ('{lon_dim}', '{lat_dim}', '{time_dim}') and factors: {e}")
             continue # Skip to next level

        # Extract time coordinates once
        time_coords_np = geo_data_coarse[time_dim].values
        LOG.info(f"Coarsened data shape for variable '{var_name}': {geo_data_coarse[var_name].shape}")

        # Create output directory for this level
        level_output_folder = os.path.join(MCMC_OUT_FOLDER, f"level_{coarsen_level}")
        os.makedirs(level_output_folder, exist_ok=True)

        # --- Prepare Tasks for Parallel Execution ---
        tasks = []
        LOG.info("Generating tasks for parallel processing...")
        num_lon = geo_data_coarse[lon_dim].shape[0]
        num_lat = geo_data_coarse[lat_dim].shape[0]

        num_total_tasks_expected = num_lon * num_lat * len(model_configs)
        if num_total_tasks_expected == 0:
            LOG.warning(f"No tasks to generate for level {coarsen_level} (check data dimensions). Skipping level.")
            continue

        # Split the master key for all tasks at this level
        task_keys = random.split(master_key, num_total_tasks_expected + 1)
        master_key = task_keys[0] # Advance the master state
        task_keys = task_keys[1:] # Use the rest for the current tasks
        key_idx = 0

        pbar_tasks = tqdm(total=num_total_tasks_expected, desc=f"Lvl {coarsen_level} Task Gen")

        for lon_index in range(num_lon):
            lon = geo_data_coarse[lon_dim].values[lon_index]
            for lat_index in range(num_lat):
                lat = geo_data_coarse[lat_dim].values[lat_index]

                # Extract target variable (as numpy array)
                target_variable_np = geo_data_coarse[var_name].sel({lat_dim: lat, lon_dim: lon}).values

                # Skip NaN locations early (check before generating tasks for all configs)
                if np.isnan(target_variable_np).any():
                    key_idx += len(model_configs) # Increment key index past keys for this location
                    pbar_tasks.update(len(model_configs))
                    continue

                for config_model in model_configs: # Renamed loop variable
                    if key_idx >= len(task_keys):
                         LOG.error(f"Key index {key_idx} out of bounds ({len(task_keys)} keys available). Mismatch in task count calculation? Stopping task generation.")
                         break # Avoid crashing

                    # Arguments for the worker function (ensure order matches run_single_task unpack)
                    task_args = (
                        coarsen_level, lat_index, lon_index, lat, lon, config_model,
                        time_coords_np, target_variable_np,
                        oni_norm_np, oni_time_np, cr_norm_np, cr_time_np,
                        level_output_folder,
                        task_keys[key_idx], # Assign a unique key
                        data_label,       # Pass data label from config
                        smoothing,        # Pass smoothing value from config
                        prior_params,     # Pass the dictionary of prior parameters from config
                        constructor_kwargs, # Pass sampler settings from config
                        num_posterior_samples # Pass num samples from config
                    )
                    tasks.append(task_args)
                    key_idx += 1
                    pbar_tasks.update(1)

                if key_idx >= len(task_keys) and lat_index < num_lat -1: # Break inner loop if keys run out
                     break
            if key_idx >= len(task_keys) and lon_index < num_lon -1: # Break outer loop if keys run out
                 break
        pbar_tasks.close()
        LOG.info(f"Generated {len(tasks)} tasks for level {coarsen_level}.")
        if key_idx != num_total_tasks_expected and len(tasks) < num_total_tasks_expected :
             LOG.warning(f"Expected {num_total_tasks_expected} tasks but generated {len(tasks)}. Key index ended at {key_idx}. Check for early NaNs or key generation issues.")


        # --- Execute Tasks in Parallel ---
        if tasks:
            LOG.info(f"Starting parallel execution with {workers} workers...")
            # Use try-finally to ensure pool closure
            pool = None
            try:
                # Pass initializer if needed for env vars, though spawn often handles it
                pool = multiprocessing.Pool(processes=workers)
                results = list(tqdm(pool.imap_unordered(run_single_task, tasks), total=len(tasks), desc=f"Level {coarsen_level} Processing"))
            finally:
                if pool:
                    pool.close() # Prevent new tasks
                    pool.join() # Wait for workers to finish

            # Summarize results
            completed_count = sum(1 for r in results if r.startswith("Completed"))
            skipped_nan_count = sum(1 for r in results if r.startswith("Skipped NaN"))
            skipped_exist_count = sum(1 for r in results if r.startswith("Skipped Existing"))
            error_count = sum(1 for r in results if r.startswith("Error"))
            LOG.info(f"Level {coarsen_level} finished. Completed: {completed_count}, Skipped (NaN): {skipped_nan_count}, Skipped (Exists): {skipped_exist_count}, Errors: {error_count}")
            # Optionally print error messages collected
            for r in results:
                if r.startswith("Error"):
                    LOG.warning(f"Worker reported: {r}") # Log errors again for visibility
        else:
            LOG.info(f"No valid tasks to run for level {coarsen_level}.")


    end_time = time.time()
    LOG.info(f"Bayesian analysis for {data_label} finished. Total time: {end_time - start_time:.2f} seconds.")

