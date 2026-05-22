"""Auxiliary dataset handling for profile processing.

An auxiliary dataset carries time series (lat, lon, T, S, density) that get
interpolated onto each profile. The merge needs decoded (datetime64) times to
match the auxiliary dataset's time axis; profile processing itself needs raw
(float-second) times because ``window_mean`` averages the time coordinate.
This module hides that asymmetry behind a single ``attach_auxiliary`` call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from .profile import ProfileConfig

logger = logging.getLogger(__name__)

__all__ = [
    "AUX_VARS",
    "load_auxiliary",
    "merge_auxiliary_data",
    "attach_auxiliary",
]

# Names of auxiliary variables once interpolated onto the profile.
AUX_VARS = (
    "aux_latitude",
    "aux_longitude",
    "aux_temperature",
    "aux_salinity",
    "aux_density",
)


def load_auxiliary(
    auxiliary_file: Union[str, Path],
    config: "ProfileConfig",
) -> xr.Dataset:
    """Load and validate an auxiliary NetCDF dataset.

    Requires a CF-decodable 'time' coordinate. Sorts by time, drops NaT
    times, and linearly interpolates over NaN values in the variables named
    by ``config`` (lat/lon always; T/S/density only when explicitly set).
    """
    auxiliary_file = Path(auxiliary_file)
    if not auxiliary_file.exists():
        raise FileNotFoundError(f"Auxiliary file not found: {auxiliary_file}")

    logger.info(f"Loading auxiliary dataset from {auxiliary_file}")

    aux_ds = xr.open_dataset(auxiliary_file, decode_times=True)

    if "time" not in aux_ds.coords:
        raise ValueError(
            "Auxiliary dataset must have a coordinate named 'time' for interpolation"
        )
    if not np.issubdtype(aux_ds["time"].dtype, np.datetime64):
        raise ValueError(
            "Auxiliary dataset 'time' coordinate must be CF-decodable to datetimes "
            "(e.g., 'seconds since 1970-01-01')"
        )

    aux_ds = aux_ds.dropna(dim="time", subset=["time"]).sortby("time")

    aux_vars = [config.aux_latitude, config.aux_longitude]
    opt_vars = [config.aux_temperature, config.aux_salinity, config.aux_density]
    aux_vars.extend(v for v in opt_vars if v is not None)

    for var in aux_vars:
        if var in aux_ds and aux_ds[var].isnull().any():
            aux_ds[var] = aux_ds[var].interpolate_na(
                dim="time", method="linear", fill_value="extrapolate"
            )
            logger.info(f"Interpolated NaN values in auxiliary variable '{var}'")

    return aux_ds


def merge_auxiliary_data(
    ds: xr.Dataset,
    aux_ds: xr.Dataset,
    config: Optional["ProfileConfig"] = None,
) -> xr.Dataset:
    """Interpolate auxiliary variables onto a profile's ``t_slow`` axis.

    ``ds`` must have decoded (datetime64) ``t_slow`` for interpolation to
    align with the auxiliary dataset's time axis. Use :func:`attach_auxiliary`
    if you need to merge into a dataset that was loaded with raw times.
    """
    if config is None:
        from .profile import ProfileConfig

        config = ProfileConfig()

    ds = ds.copy()

    profile_time = ds.t_slow
    aux_time_var = config.aux_time
    if aux_time_var not in aux_ds.dims and aux_time_var not in aux_ds.coords:
        raise ValueError(
            f"Auxiliary time variable '{aux_time_var}' not found in auxiliary dataset"
        )

    var_mappings = [
        (config.aux_latitude, "aux_latitude"),
        (config.aux_longitude, "aux_longitude"),
    ]
    if config.aux_temperature is not None:
        var_mappings.append((config.aux_temperature, "aux_temperature"))
    if config.aux_salinity is not None:
        var_mappings.append((config.aux_salinity, "aux_salinity"))
    if config.aux_density is not None:
        var_mappings.append((config.aux_density, "aux_density"))

    for aux_var, output_var in var_mappings:
        if aux_var in aux_ds:
            interp_data = aux_ds[aux_var].interp(
                {aux_time_var: profile_time},
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            ds[output_var] = ("t_slow", interp_data.values)
            logger.debug(f"Interpolated {aux_var} -> {output_var}")
        else:
            logger.debug(f"Auxiliary variable '{aux_var}' not found, skipping")

    return ds


def attach_auxiliary(
    ds: xr.Dataset,
    aux_ds: xr.Dataset,
    config: "ProfileConfig",
) -> xr.Dataset:
    """Attach auxiliary variables to a profile loaded with raw times.

    The profile dataset is loaded with ``decode_times=False`` so that
    ``window_mean`` can numerically average ``t_slow``. The auxiliary
    interpolation, however, needs datetime64 times to match. This helper
    handles that by decoding for the merge and copying the resulting
    auxiliary arrays back onto the original (raw-time) dataset.
    """
    decoded = xr.decode_cf(ds)
    decoded = merge_auxiliary_data(decoded, aux_ds, config)
    for var in AUX_VARS:
        if var in decoded:
            ds[var] = ("t_slow", decoded[var].values)
    return ds
