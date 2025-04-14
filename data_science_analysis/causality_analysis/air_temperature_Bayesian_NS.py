import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys

import netCDF4 as nc
import glob
import numpy as np
import pandas as pd
import xarray as xr
import itertools
import cftime
import cartopy.crs as ccrs
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl

from tqdm import tqdm

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax.experimental import mesh_utils

import numpyro

numpyro.enable_x64()
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.contrib.nested_sampling import NestedSampler


import dill
import logging

log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
log_level = logging.INFO
LOG = logging.getLogger(__name__)
LOG.setLevel(log_level)

# writing to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
handler.setFormatter(log_format)
LOG.addHandler(handler)

DATA_LABEL = "air_temperature"
COARSEN_LEVELS = [16, 8, 4, 2]
CONSTRUCTOR_KWARGS = {
    "max_samples": 40000,
    "parameter_estimation": True,
    "verbose": False,
}

DATA_INPUT_FILE = (
    "/d0/amunozj/git_repos/interconnection-sun-climate/air_temperature/air.2m.gauss.nc"
)
FIGURE_FOLDER = "/d0/amunozj/git_repos/interconnection-sun-climate/outputs/figures"
MCMC_OUT_FOLDER = (
    f"/d0/amunozj/git_repos/interconnection-sun-climate/outputs/nested_sampling/{DATA_LABEL}"
)

if not os.path.exists(FIGURE_FOLDER):
    os.mkdir(FIGURE_FOLDER)

if not os.path.exists(MCMC_OUT_FOLDER):
    os.mkdir(MCMC_OUT_FOLDER)


## Define model
def single_lat_lon_model(
    time, target_variable=None, co2=False, seasonal=False, oni=None, cr=None
):

    orbits = ((time - np.min(time)) / np.timedelta64(1, "D") / 365.256).values

    # Intrinsic noise
    sigma = numpyro.sample("sigma", dist.HalfNormal(20.0))

    # Offset
    offset = numpyro.sample("offset", dist.Normal(270.0, 40.0))
    target_variable_estimate = offset

    # Seasonal
    if seasonal:
        seasonal_shift = numpyro.sample("seasonal_shift", dist.Normal(0.0, 1.0))
        seasonal_amplitude = numpyro.sample(
            "seasonal_amplitude", dist.Normal(20.0, 20.0)
        )
        seasonal_exponent = numpyro.sample("seasonal_exponent", dist.HalfNormal(2.0))

        sin_orbits = jnp.sin((orbits - seasonal_shift) * 2 * jnp.pi)
        target_variable_estimate = (
            target_variable_estimate
            + seasonal_amplitude
            * jnp.sign(sin_orbits)
            * jnp.pow(jnp.abs(sin_orbits), seasonal_exponent)
        )

    # CO2
    if co2:
        co2_gain = numpyro.sample("co2_gain", dist.Normal(20.0, 20.0))
        target_variable_estimate = (
            target_variable_estimate + orbits / jnp.max(orbits) * co2_gain
        )

    # ONI
    if oni is not None:
        oni_gain = numpyro.sample("oni_gain", dist.Normal(20.0, 40.0))
        oni_shift = numpyro.sample("oni_shift", dist.Normal(0.0, 5.0))

        orbits_oni = (
            (oni.time - np.min(time)) / np.timedelta64(1, "D") / 365.256
        ).values
        target_variable_estimate = target_variable_estimate + oni_gain * jnp.interp(
            orbits - oni_shift, orbits_oni, oni.normalized.values
        )

    # Cosmic Rays
    if cr is not None:
        cr_gain = numpyro.sample("cr_gain", dist.Normal(20.0, 40.0))
        cr_shift = numpyro.sample("cr_shift", dist.Normal(0.0, 5.0))

        orbits_cr = ((cr.time - np.min(time)) / np.timedelta64(1, "D") / 365.256).values
        target_variable_estimate = target_variable_estimate + cr_gain * jnp.interp(
            orbits - cr_shift, orbits_cr, cr.normalized.values
        )

    with numpyro.plate("times", target_variable.shape[0]):
        numpyro.sample(
            "obs", dist.Normal(target_variable_estimate, sigma), obs=target_variable
        )


if __name__ == "__main__":

    ## Read climate data
    geo_data = xr.open_dataset(DATA_INPUT_FILE)

    ## Read El Niño Index
    df = pd.read_table(
        "/d0/amunozj/git_repos/interconnection-sun-climate/ENI/detrend.nino34.ascii.txt",
        engine="c",
        sep="\s+",
    )
    df["DAY"] = 15
    df["time"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]])
    df = df.set_index("time", drop=True).loc[:, ["TOTAL", "ClimAdjust"]]
    oni_data = xr.Dataset.from_dataframe(df)
    oni_data["oni"] = oni_data.TOTAL - oni_data.ClimAdjust
    oni_data["normalized"] = (oni_data.oni - oni_data.oni.mean()) / oni_data.oni.std()

    ## Read cosmic ray data
    crDataPath = "../../data/cosmic_rays/OULU_1964_05_01 _00_00_2024_10_28 _23_30.csv"
    cr = pd.read_csv(crDataPath)
    cr["time"] = (
        pd.to_datetime(cr["Timestamp"]).dt.tz_localize(None).astype("datetime64[ns]")
    )
    cr = cr.set_index("time", drop=True)
    cr = xr.Dataset.from_dataframe(cr)
    cr["normalized"] = (
        cr.CorrectedCountRate - cr.CorrectedCountRate.mean()
    ) / cr.CorrectedCountRate.std()

    pbar1 = tqdm(COARSEN_LEVELS, position=0, total=len(COARSEN_LEVELS))
    for coarsen_level in pbar1:
        pbar1.set_postfix_str(f"Processing level {coarsen_level}", refresh=True)
        geo_data_coarse = geo_data.coarsen(
            lon=coarsen_level, lat=coarsen_level, boundary="pad"
        ).mean()

        output_folder = os.path.join(MCMC_OUT_FOLDER, f"level_{coarsen_level}")
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        pbar2 = tqdm(
            range(geo_data_coarse.lon.shape[0]),
            position=1,
            total=geo_data_coarse.lon.shape[0],
            leave=False,
        )
        for lon_index in pbar2:
            pbar2.set_postfix_str(
                f"Processing {geo_data_coarse.lon.values[lon_index]}° longitude",
                refresh=True,
            )

            pbar3 = tqdm(
                range(geo_data_coarse.lat.shape[0]),
                position=2,
                total=geo_data_coarse.lat.shape[0],
                leave=False,
            )
            for lat_index in pbar3:
                pbar3.set_postfix_str(
                    f"Processing {geo_data_coarse.lat.values[lat_index]}° latitude",
                    refresh=True,
                )

                # Run with only co2 increase and seasonal variation
                output_file = os.path.join(
                    output_folder,
                    f"{DATA_LABEL}_level{coarsen_level}_co2_season_lat_{lat_index}_lon_{lon_index}.pkl",
                )
                if not os.path.isfile(output_file):
                    ns = NestedSampler(
                        single_lat_lon_model, constructor_kwargs=CONSTRUCTOR_KWARGS
                    )
                    ns.run(
                        random.PRNGKey(2),
                        geo_data_coarse.time,
                        geo_data_coarse.air.values[:, 0, lat_index, lon_index],
                        co2=True,
                        seasonal=True,
                        oni=None,
                        cr=None,
                    )
                    output_dictionary = ns.get_samples(
                        random.PRNGKey(3), num_samples=5000
                    )
                    output_dictionary["log_Z_mean"] = ns._results.log_Z_mean
                    with open(output_file, "wb") as f:  # open a text file
                        dill.dump(output_dictionary, f)  # serialize the mcmc run
                    f.close()

                # Run with co2 increase, seasonal variation, and ONI
                output_file = os.path.join(
                    output_folder,
                    f"{DATA_LABEL}_level{coarsen_level}_co2_season_oni_lat_{lat_index}_lon_{lon_index}.pkl",
                )
                if not os.path.isfile(output_file):
                    ns = NestedSampler(
                        single_lat_lon_model, constructor_kwargs=CONSTRUCTOR_KWARGS
                    )
                    ns.run(
                        random.PRNGKey(2),
                        geo_data_coarse.time,
                        geo_data_coarse.air.values[:, 0, lat_index, lon_index],
                        co2=True,
                        seasonal=True,
                        oni=oni_data,
                        cr=None,
                    )
                    output_dictionary = ns.get_samples(
                        random.PRNGKey(3), num_samples=5000
                    )
                    output_dictionary["log_Z_mean"] = ns._results.log_Z_mean
                    with open(output_file, "wb") as f:  # open a text file
                        dill.dump(output_dictionary, f)  # serialize the mcmc run
                    f.close()

                # Run with co2 increase, seasonal variation, and cr
                output_file = os.path.join(
                    output_folder,
                    f"{DATA_LABEL}_level{coarsen_level}_co2_season_cr_lat_{lat_index}_lon_{lon_index}.pkl",
                )
                if not os.path.isfile(output_file):
                    ns = NestedSampler(
                        single_lat_lon_model, constructor_kwargs=CONSTRUCTOR_KWARGS
                    )
                    ns.run(
                        random.PRNGKey(2),
                        geo_data_coarse.time,
                        geo_data_coarse.air.values[:, 0, lat_index, lon_index],
                        co2=True,
                        seasonal=True,
                        oni=None,
                        cr=cr,
                    )
                    output_dictionary = ns.get_samples(
                        random.PRNGKey(3), num_samples=5000
                    )
                    output_dictionary["log_Z_mean"] = ns._results.log_Z_mean
                    with open(output_file, "wb") as f:  # open a text file
                        dill.dump(output_dictionary, f)  # serialize the mcmc run
                    f.close()

                # Run with co2 increase, seasonal variation, oni, and cr
                output_file = os.path.join(
                    output_folder,
                    f"{DATA_LABEL}_level{coarsen_level}_co2_season_oni_cr_lat_{lat_index}_lon_{lon_index}.pkl",
                )
                if not os.path.isfile(output_file):
                    ns = NestedSampler(
                        single_lat_lon_model, constructor_kwargs=CONSTRUCTOR_KWARGS
                    )
                    ns.run(
                        random.PRNGKey(2),
                        geo_data_coarse.time,
                        geo_data_coarse.air.values[:, 0, lat_index, lon_index],
                        co2=True,
                        seasonal=True,
                        oni=oni_data,
                        cr=cr,
                    )
                    output_dictionary = ns.get_samples(
                        random.PRNGKey(3), num_samples=5000
                    )
                    output_dictionary["log_Z_mean"] = ns._results.log_Z_mean
                    with open(output_file, "wb") as f:  # open a text file
                        dill.dump(output_dictionary, f)  # serialize the mcmc run
                    f.close()
