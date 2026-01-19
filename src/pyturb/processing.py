"""Batch processing functions for microstructure data."""

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Literal, Optional, Union

import gsw  # type: ignore[import]
import numpy as np
import xarray as xr

from .io import load_profile_nc
from .profile import (
    ProfileConfig,
    merge_auxiliary_data,
    prepare_profile,
    process_profile,
    split_into_profiles,
)

logger = logging.getLogger(__name__)

__all__ = [
    "batch_compute_epsilon",
    "bin_profiles",
]


def _process_file(
    input_file: Path,
    output_dir: Path,
    config: ProfileConfig,
    overwrite: bool,
    aux_ds: Optional[xr.Dataset] = None,
) -> list[tuple]:
    """
    Process a file that may contain multiple profiles.

    Returns list of (input_path, output_path, profile_index, error) tuples.
    """
    results: list[tuple[Path, Optional[Path], int, Optional[str]]] = []
    stem = input_file.stem

    try:
        ds = load_profile_nc(input_file)

        # Merge auxiliary data if provided (needs decoded times)
        if aux_ds is not None:
            ds_decoded = xr.decode_cf(ds)
            ds_decoded = merge_auxiliary_data(ds_decoded, aux_ds, config)
            # Copy auxiliary variables back to original dataset
            for var in [
                "aux_latitude",
                "aux_longitude",
                "aux_temperature",
                "aux_salinity",
                "aux_density",
            ]:
                if var in ds_decoded:
                    ds[var] = ("t_slow", ds_decoded[var].values)

        # Prepare the profile (smooth speed/pressure, scale shear/gradT)
        ds = prepare_profile(ds, config)

        # Split into individual profiles
        profile_count = 0
        for profile_idx, profile_ds in split_into_profiles(ds, config):
            profile_count += 1
            # Generate output filename with profile index
            output_file = output_dir / f"{stem}_p{profile_idx:04d}.nc"

            if output_file.exists() and not overwrite:
                results.append(
                    (input_file, output_file, profile_idx, "skipped (exists)")
                )
                continue

            try:
                result = process_profile(profile_ds, config)

                # Keep only variables on time dimension (epsilon results)
                vars_to_keep = [
                    v
                    for v in result.data_vars
                    if "time" in result[v].dims and len(result[v].dims) > 0
                ]
                result_filtered = result[vars_to_keep]

                # Add coordinates
                result_filtered = result_filtered.assign_coords(
                    frequency=result.frequency,
                    k=result.k,
                )

                # Ensure time coordinate has units attribute
                if "time" in result.coords and "units" in result.time.attrs:
                    result_filtered.time.attrs["units"] = result.time.attrs["units"]
                if "time" in result.coords and "long_name" in result.time.attrs:
                    result_filtered.time.attrs["long_name"] = result.time.attrs[
                        "long_name"
                    ]

                # Add source file metadata
                result_filtered.attrs["source_file"] = input_file.name
                result_filtered.attrs["profile_index"] = profile_idx
                result_filtered.attrs["profile_direction"] = config.profile_direction

                # Copy instrument info from source dataset
                for attr in ["instrument_vehicle", "instrument_model", "instrument_sn"]:
                    if attr in ds.attrs:
                        result_filtered.attrs[attr] = ds.attrs[attr]

                result_filtered.to_netcdf(output_file)
                results.append((input_file, output_file, profile_idx, None))

            except Exception as e:
                results.append((input_file, None, profile_idx, str(e)))

        # If no profiles were found, try single-profile processing as fallback
        if profile_count == 0:
            output_file = output_dir / f"{stem}_p0000.nc"

            if output_file.exists() and not overwrite:
                return [(input_file, output_file, 0, "skipped (exists)")]

            # Fall back to original single-profile processing
            result = process_profile(ds, config)

            vars_to_keep = [
                v
                for v in result.data_vars
                if "time" in result[v].dims and len(result[v].dims) > 0
            ]
            result_filtered = result[vars_to_keep]
            result_filtered = result_filtered.assign_coords(
                frequency=result.frequency,
                k=result.k,
            )
            if "time" in result.coords and "units" in result.time.attrs:
                result_filtered.time.attrs["units"] = result.time.attrs["units"]
            if "time" in result.coords and "long_name" in result.time.attrs:
                result_filtered.time.attrs["long_name"] = result.time.attrs["long_name"]

            result_filtered.attrs["source_file"] = input_file.name
            result_filtered.attrs["profile_index"] = 0

            result_filtered.to_netcdf(output_file)
            results.append((input_file, output_file, 0, None))

    except Exception as e:
        results.append((input_file, None, -1, str(e)))

    return results


def _unpack_epsilon_args(args: tuple) -> list[tuple]:
    """Unpack arguments for imap_unordered."""
    input_file, output_dir, config, overwrite, aux_ds = args
    return _process_file(input_file, output_dir, config, overwrite, aux_ds)


def batch_compute_epsilon(
    files: Union[str, Path, list[Path]],
    output_dir: Optional[Union[str, Path]] = None,
    diss_len_sec: float = 4.0,
    fft_len_sec: float = 1.0,
    min_speed: float = 0.2,
    pressure_smoothing_period: float = 0.25,
    temperature: str = "JAC_T",
    speed: str = "W",
    angle_of_attack: float = 3.0,
    use_pitch_correction: bool = False,
    profile_direction: Literal["down", "up", "both"] = "down",
    min_profile_pressure: float = 0.0,
    peaks_kwargs: Optional[dict] = None,
    auxiliary_file: Optional[Union[str, Path]] = None,
    aux_latitude: str = "lat",
    aux_longitude: str = "lon",
    aux_temperature: Optional[str] = None,
    aux_salinity: Optional[str] = None,
    aux_density: Optional[str] = None,
    despike_max_passes: int = 6,
    n_workers: Optional[int] = None,
    verbose: bool = False,
    overwrite: bool = False,
) -> list[dict]:
    """
    Batch compute epsilon from converted NetCDF files.

    This function processes raw p2nc output by:
    1. Detecting multiple profiles within each file (for glider data)
    2. Smoothing speed and pressure data
    3. Scaling shear probes by 1/U^2 and gradT probes by 1/U
    4. Computing epsilon using the Nasmyth spectrum fit

    Each input file may produce multiple output files if it contains multiple
    dive cycles. Output files are named {original_stem}_p{NNN}.nc where NNN
    is the 0-indexed profile number.

    Parameters
    ----------
    files : str, Path, or list of Path
        Either a glob pattern to match NetCDF files (e.g., '/path/to/data/*.nc'),
        a directory path (in which case '*.nc' is appended), or a list of
        Path objects pointing to specific files.
    output_dir : str or Path, optional
        Directory for output NetCDF files. If None, uses current directory.
    diss_len_sec : float, optional
        Dissipation window length in seconds. Default 4.0.
    fft_len_sec : float, optional
        FFT window length in seconds. Default 1.0.
    min_speed : float, optional
        Minimum speed threshold for valid data. Default 0.2 m/s.
    pressure_smoothing_period : float, optional
        Low-pass filter cutoff period for pressure smoothing. Default 0.25 s.
    temperature : str, optional
        Name of temperature variable for viscosity calculation. Default 'JAC_T'.
    speed : str, optional
        Name of speed variable. If not found in dataset, speed is estimated
        from pressure derivative. Default 'W'.
    angle_of_attack : float, optional
        Angle of attack in degrees, used when estimating speed from pressure.
        Default 3.0.
    use_pitch_correction : bool, optional
        Whether to apply pitch correction when estimating speed from pressure.
        Default False.
    profile_direction : {'down', 'up', 'both'}, optional
        Which cast directions to process. Default 'down'.
    min_profile_pressure : float, optional
        Minimum pressure (dbar) for profile detection. Default 10.0.
    peaks_kwargs : dict, optional
        Keyword arguments for scipy.signal.find_peaks used in profile detection.
        Default uses height=25, distance=200, width=200, prominence=25.
    auxiliary_file : str or Path, optional
        Path to auxiliary NetCDF file containing time series of latitude,
        longitude, temperature, salinity, and/or density. These are interpolated
        onto each profile and used for viscosity calculation and output.
    n_workers : int, optional
        Number of parallel workers. Default is number of CPU cores.
    verbose : bool, optional
        Print progress information. Default False.
    overwrite : bool, optional
        Whether to overwrite existing files. Default False.

    Returns
    -------
    list of dict
        Results for each profile with keys: 'input', 'output', 'profile_index',
        'success', 'error'

    Examples
    --------
    >>> from pyturb.processing import batch_compute_epsilon
    >>> # Using glob pattern - processes all profiles in each file
    >>> results = batch_compute_epsilon('/path/to/data/*.nc', output_dir='/path/to/output')
    >>> # Process only up casts
    >>> results = batch_compute_epsilon('/path/to/data/*.nc', profile_direction='up')
    >>> # Process both up and down casts
    >>> results = batch_compute_epsilon('/path/to/data/*.nc', profile_direction='both')
    """
    # Handle different input types
    if isinstance(files, list):
        # Already a list of files
        nc_files = sorted(files)
    else:
        # It's a pattern or directory
        pattern = Path(files)

        if pattern.is_dir():
            pattern = pattern / "*.nc"

        if pattern.is_absolute():
            nc_files = sorted(pattern.parent.glob(pattern.name))
        else:
            nc_files = sorted(Path.cwd().glob(str(pattern)))

    if not nc_files:
        if verbose:
            logger.info("No NetCDF files found.")
        return []

    if verbose:
        logger.info(f"Found {len(nc_files)} NetCDF files to process")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()

    # Note: Skip logic for existing files is now handled per-profile in _process_file

    if n_workers is None:
        n_workers = mp.cpu_count()

    # Set default peaks_kwargs if not provided
    if peaks_kwargs is None:
        peaks_kwargs = {
            "height": 25,
            "distance": 200,
            "width": 200,
            "prominence": 25,
        }

    config = ProfileConfig(
        diss_len_sec=diss_len_sec,
        fft_len_sec=fft_len_sec,
        min_speed=min_speed,
        pressure_smoothing_period=pressure_smoothing_period,
        temperature=temperature,
        speed=speed,
        angle_of_attack=angle_of_attack,
        use_pitch_correction=use_pitch_correction,
        profile_direction=profile_direction,
        min_profile_pressure=min_profile_pressure,
        peaks_kwargs=peaks_kwargs,
        aux_latitude=aux_latitude,
        aux_longitude=aux_longitude,
        aux_temperature=aux_temperature,
        aux_salinity=aux_salinity,
        aux_density=aux_density,
        despike_max_passes=despike_max_passes,
        verbose=verbose,
    )

    # Load auxiliary dataset if provided
    aux_ds = None
    if auxiliary_file is not None:
        auxiliary_file = Path(auxiliary_file)
        if not auxiliary_file.exists():
            raise FileNotFoundError(f"Auxiliary file not found: {auxiliary_file}")

        logger.info(f"Loading auxiliary dataset from {auxiliary_file}")

        aux_ds = xr.open_dataset(auxiliary_file, decode_times=True)

        # Require a time coordinate named 'time' (CF time decoding is expected)
        if "time" not in aux_ds.coords:
            raise ValueError(
                "Auxiliary dataset must have a coordinate named 'time' for interpolation"
            )

        # Ensure time coordinate decodes to datetime64
        if not np.issubdtype(aux_ds["time"].dtype, np.datetime64):
            raise ValueError(
                "Auxiliary dataset 'time' coordinate must be CF-decodable to datetimes (e.g., 'seconds since 1970-01-01')"
            )

        # Drop NaN times, sort by time, and remove duplicate times (keep first occurrence)
        aux_ds = aux_ds.dropna(dim="time", subset=["time"]).sortby("time")

        # Interpolate over NaN values in auxiliary variables (use configured names).
        # Latitude/longitude are always considered; temperature/salinity/density
        # are only used if explicitly provided by the user (config value not None).
        aux_vars = [config.aux_latitude, config.aux_longitude]
        opt_vars = [config.aux_temperature, config.aux_salinity, config.aux_density]
        aux_vars.extend([v for v in opt_vars if v is not None])

        for var in aux_vars:
            if var in aux_ds and aux_ds[var].isnull().any():
                aux_ds[var] = aux_ds[var].interpolate_na(
                    dim="time", method="linear", fill_value="extrapolate"
                )
                logger.info(f"Interpolated NaN values in auxiliary variable '{var}'")

        logger.info(f"Loaded auxiliary dataset from {auxiliary_file}")

    args = [(f, output_dir, config, overwrite, aux_ds) for f in nc_files]

    results = []
    total_profiles = 0

    # Limit workers to number of files (no benefit having more workers than files)
    effective_workers = min(n_workers, len(nc_files))

    if verbose:
        logger.info(
            f"Using {effective_workers} parallel workers for {len(nc_files)} files"
        )

    with mp.Pool(processes=effective_workers) as pool:
        results_iter = pool.imap_unordered(_unpack_epsilon_args, args)
        for i, file_results in enumerate(results_iter):
            for input_path, output_path, profile_idx, error in file_results:
                success = error is None
                results.append(
                    {
                        "input": input_path,
                        "output": output_path,
                        "profile_index": profile_idx,
                        "success": success,
                        "error": error,
                    }
                )
                total_profiles += 1
                if verbose:
                    if success:
                        status = f"profile {profile_idx} processed"
                    else:
                        status = f"profile {profile_idx} failed ({error})"
                    logger.info(
                        f"[{i + 1}/{len(nc_files)}] {input_path.name}: {status}"
                    )

    # Summary
    if verbose:
        n_success = sum(1 for r in results if r["success"])
        n_failed = len(results) - n_success
        logger.info(
            f"Completed: {n_success} profiles succeeded, {n_failed} failed from {len(nc_files)} files"
        )

    return results


def _bin_single_profile(
    file: Path,
    depth_bins: np.ndarray,
    variables: list[str],
    default_latitude: float = 45.0,
    bin_by_pressure: bool = False,
) -> Optional[xr.Dataset]:
    """
    Bin a single profile dataset by depth (or pressure).

    Parameters
    ----------
    file : Path
        Path to the NetCDF file.
    depth_bins : np.ndarray
        Bin edges for depth (or pressure if bin_by_pressure=True).
    variables : list of str
        Variables to include in the binned output.
    default_latitude : float
        Default latitude for pressure-to-depth conversion if not in data.
    bin_by_pressure : bool
        If True, bin by pressure instead of depth.

    Returns binned dataset or None if an error occurs.
    """
    try:
        ds = xr.load_dataset(file, decode_times=False)

        # Determine which variables exist in the dataset
        vars_to_bin = [v for v in variables if v in ds]
        if not vars_to_bin:
            return None

        # Convert time coordinate to epoch seconds (seconds since 1970-01-01)
        time_epoch = None
        if "time" in ds.coords:
            time_values = ds.time.values
            time_units = ds.time.attrs.get("units", "")

            if time_units.startswith("seconds since "):
                # Parse the reference time from units string
                ref_time_str = time_units.replace("seconds since ", "")
                try:
                    ref_time = np.datetime64(ref_time_str)
                    # Convert reference time to epoch seconds
                    epoch = np.datetime64("1970-01-01T00:00:00")
                    ref_epoch_sec = (ref_time - epoch) / np.timedelta64(1, "s")
                    # Add to time values to get epoch seconds
                    time_epoch = time_values + ref_epoch_sec
                except ValueError:
                    # If parsing fails, just use raw values
                    time_epoch = time_values
            else:
                time_epoch = time_values

            ds["time_var"] = ("time", time_epoch)
            vars_to_bin_with_time = vars_to_bin + ["time_var"]
        else:
            vars_to_bin_with_time = vars_to_bin

        # Subset to variables of interest
        ds_subset = ds[vars_to_bin_with_time]

        if bin_by_pressure:
            # Bin by pressure directly
            bin_var = ds.pressure
            bin_name = "pressure_bins"
            coord_name = "pressure"
        else:
            # Convert pressure to depth using gsw
            # Get latitude - use data if available, otherwise default
            if "lat" in ds and not np.all(np.isnan(ds.lat.values)):
                lat = np.nanmean(ds.lat.values)
            else:
                lat = default_latitude

            # Calculate depth from pressure
            depth = gsw.z_from_p(ds.pressure.values, lat)
            # gsw returns negative depths (below surface), convert to positive
            depth = -depth
            ds_subset["_depth_for_binning"] = ("time", depth)
            bin_var = ds_subset["_depth_for_binning"]
            bin_name = "_depth_for_binning_bins"
            coord_name = "depth"

        # Group by bins and compute mean
        ds_binned = ds_subset.groupby_bins(bin_var, bins=depth_bins).mean()

        # Convert bin intervals to midpoints
        ds_binned[bin_name] = np.array(
            [interval.mid for interval in ds_binned[bin_name].values]
        )
        ds_binned = ds_binned.rename({bin_name: coord_name})

        # Remove the temporary depth variable if we added it
        if not bin_by_pressure and "_depth_for_binning" in ds_binned:
            ds_binned = ds_binned.drop_vars("_depth_for_binning", errors="ignore")

        # Rename time_var back to time if it exists and add epoch units
        if "time_var" in ds_binned:
            ds_binned = ds_binned.rename({"time_var": "time"})
            ds_binned["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"
            ds_binned["time"].attrs["long_name"] = "Time"
            ds_binned["time"].attrs["calendar"] = "proleptic_gregorian"

        # Add source file as attribute
        ds_binned.attrs["source_file"] = file.name

        # Add instrument serial number as a data variable (to be used as coordinate)
        if "instrument_sn" in ds.attrs:
            ds_binned["instrument_sn"] = ds.attrs["instrument_sn"]
        if "instrument_vehicle" in ds.attrs:
            ds_binned["instrument_vehicle"] = ds.attrs["instrument_vehicle"]

        return ds_binned

    except Exception as e:
        logger.error(f"Error binning {file}: {e}")
        return None


def _unpack_bin_args(args: tuple) -> Optional[xr.Dataset]:
    """Unpack arguments for imap_unordered."""
    file, depth_bins, variables, default_latitude, bin_by_pressure = args
    return _bin_single_profile(
        file, depth_bins, variables, default_latitude, bin_by_pressure
    )


def bin_profiles(
    files: Union[str, Path, list[Path]],
    output_file: Union[str, Path] = "binned_profiles.nc",
    depth_min: float = 0.0,
    depth_max: float = 1000.0,
    bin_width: float = 2.0,
    variables: Optional[list[str]] = None,
    default_latitude: float = 45.0,
    bin_by_pressure: bool = False,
    n_workers: Optional[int] = None,
    verbose: bool = False,
) -> Optional[xr.Dataset]:
    """
    Bin multiple profile datasets by depth (or pressure) and concatenate.

    This function reads epsilon output files, bins them by depth (default)
    or pressure, and concatenates them along a 'profile' dimension.
    Depth is calculated from pressure using gsw.z_from_p().

    Parameters
    ----------
    files : str, Path, or list of Path
        Either a glob pattern to match NetCDF files (e.g., '/path/to/data/*.nc'),
        a directory path (in which case '*.nc' is appended), or a list of
        Path objects pointing to specific files.
    output_file : str or Path, optional
        Output file path. Default 'binned_profiles.nc'.
    depth_min : float, optional
        Minimum depth for binning. Default 0.0 m.
    depth_max : float, optional
        Maximum depth for binning. Default 1000.0 m.
    bin_width : float, optional
        Width of depth bins. Default 2.0 m.
    variables : list of str, optional
        Variables to include in binned output. Default includes eps_1, eps_2,
        W, temperature, salinity, density, nu, latitude, longitude.
    default_latitude : float, optional
        Latitude to use for pressure-to-depth conversion if not available
        in the data. Default 45.0 degrees.
    bin_by_pressure : bool, optional
        If True, bin by pressure (dbar) instead of depth (m). Default False.
    n_workers : int, optional
        Number of parallel workers. Default is number of CPU cores.
    verbose : bool, optional
        Print progress information. Default False.

    Returns
    -------
    xr.Dataset
        Binned and concatenated dataset with dimensions (profile, depth)
        or (profile, pressure) if bin_by_pressure=True.

    Examples
    --------
    >>> from pyturb.processing import bin_profiles
    >>> ds = bin_profiles('/path/to/eps_output/*.nc', output_file='binned.nc')
    >>> # Bin by pressure instead of depth
    >>> ds = bin_profiles('/path/to/eps_output/*.nc', bin_by_pressure=True)
    """
    # Default variables to bin
    if variables is None:
        variables = [
            "eps_1",
            "eps_2",
            "W",
            "temperature",
            "conductivity",
            "salinity",
            "density",
            "nu",
            "lat",
            "lon",
        ]

    # Handle different input types
    if isinstance(files, list):
        nc_files = sorted(files)
    else:
        pattern = Path(files)
        if pattern.is_dir():
            pattern = pattern / "*.nc"
        if pattern.is_absolute():
            nc_files = sorted(pattern.parent.glob(pattern.name))
        else:
            nc_files = sorted(Path.cwd().glob(str(pattern)))

    if not nc_files:
        if verbose:
            logger.info("No NetCDF files found.")
        return None

    if verbose:
        logger.info(f"Found {len(nc_files)} NetCDF files to bin")
        coord_type = "pressure" if bin_by_pressure else "depth"
        logger.info(
            f"Binning by {coord_type} from {depth_min} to {depth_max} m with {bin_width} m bins"
        )

    # Create depth (or pressure) bins
    depth_bins = np.arange(depth_min, depth_max + bin_width, bin_width)

    if n_workers is None:
        n_workers = mp.cpu_count()

    args = [
        (f, depth_bins, variables, default_latitude, bin_by_pressure) for f in nc_files
    ]

    binned_datasets = []

    # Use serial processing for small batches
    if len(nc_files) <= min(n_workers, 4):
        if verbose:
            logger.info("Using serial processing for small batch")
        for i, arg_tuple in enumerate(args):
            result = _unpack_bin_args(arg_tuple)
            if result is not None:
                binned_datasets.append(result)
            if verbose:
                status = "binned" if result is not None else "skipped"
                logger.info(f"[{i + 1}/{len(nc_files)}] {status}: {nc_files[i].name}")
    else:
        if verbose:
            logger.info(f"Using {n_workers} parallel workers")
        with mp.Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap(_unpack_bin_args, args)):
                if result is not None:
                    binned_datasets.append(result)
                if verbose:
                    status = "binned" if result is not None else "skipped"
                    logger.info(
                        f"[{i + 1}/{len(nc_files)}] {status}: {nc_files[i].name}"
                    )

    if not binned_datasets:
        if verbose:
            logger.info("No datasets were successfully binned.")
        return None

    if verbose:
        logger.info(f"Concatenating {len(binned_datasets)} binned profiles...")

    # Concatenate along profile dimension
    combined = xr.concat(binned_datasets, dim="profile")

    # Sort profiles by time (use minimum time per profile to handle NaT values)
    if "time" in combined:
        # Get representative time for each profile (min time, skipping NaT)
        profile_times = combined.time.min(dim="depth", skipna=True)
        # Sort by time
        sort_order = np.argsort(profile_times.values)
        combined = combined.isel(profile=sort_order)
        if verbose:
            logger.info("Sorted profiles by time")

    # Save to file
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_netcdf(output_file)

    if verbose:
        logger.info(f"Saved binned data to {output_file}")

    return combined
