"""Batch processing functions for microstructure data."""

import logging
import multiprocessing as mp
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Optional, Union

import gsw  # type: ignore[import]
import numpy as np
import xarray as xr

from . import __version__
from .auxiliary import attach_auxiliary, load_auxiliary
from .io import load_profile_nc, resolve_input_files
from .profile import (
    ProfileConfig,
    prepare_profile,
    process_profile,
    split_into_profiles,
)

_log = logging.getLogger(__name__)

__all__ = [
    "batch_compute_epsilon",
    "bin_profiles",
]


_INSTRUMENT_ATTRS = ("instrument_vehicle", "instrument_model", "instrument_sn")


def _write_epsilon_profile(
    result: xr.Dataset,
    source_ds: xr.Dataset,
    output_file: Path,
    source_file_name: str,
    profile_idx: int,
    config: ProfileConfig,
) -> None:
    """Write a processed-profile dataset to NetCDF with stamped metadata.

    Keeps only time-dimensioned data variables, reattaches the ``frequency``
    and ``k`` coordinates, carries over ``time`` attrs, and stamps source
    file / profile / instrument metadata.
    """
    vars_to_keep = [
        v
        for v in result.data_vars
        if "time" in result[v].dims and len(result[v].dims) > 0
    ]
    out = result[vars_to_keep].assign_coords(
        frequency=result.frequency,
        k=result.k,
    )

    if "time" in result.coords:
        for attr in ("units", "long_name"):
            if attr in result.time.attrs:
                out.time.attrs[attr] = result.time.attrs[attr]

    out.attrs["source_file"] = source_file_name
    out.attrs["profile_index"] = profile_idx
    out.attrs["profile_direction"] = config.profile_direction
    out.attrs["pyturb_version"] = __version__
    out.attrs["pyturb_processed_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    out.attrs["pyturb_config"] = config.to_yaml()
    for attr in _INSTRUMENT_ATTRS:
        if attr in source_ds.attrs:
            out.attrs[attr] = source_ds.attrs[attr]

    out.to_netcdf(output_file)


def _process_file(
    input_file: Path,
    output_dir: Path,
    config: ProfileConfig,
    overwrite: bool,
    aux_ds: Optional[xr.Dataset] = None,
) -> list[tuple]:
    """Process a file that may contain multiple profiles.

    Returns list of (input_path, output_path, profile_index, error) tuples.
    """
    results: list[tuple[Path, Optional[Path], int, Optional[str]]] = []
    stem = input_file.stem

    try:
        ds = load_profile_nc(input_file)
        if aux_ds is not None:
            ds = attach_auxiliary(ds, aux_ds, config)
        ds = prepare_profile(ds, config)

        # Iterate detected profiles, falling back to one whole-file profile.
        profile_iter = list(split_into_profiles(ds, config))
        if not profile_iter:
            _log.info("Processing the whole file as a single profile.")
            profile_iter = [(0, ds)]

        for profile_idx, profile_ds in profile_iter:
            output_file = output_dir / f"{stem}_p{profile_idx:04d}.nc"

            if output_file.exists() and not overwrite:
                results.append(
                    (input_file, output_file, profile_idx, "skipped (exists)")
                )
                continue

            try:
                result = process_profile(profile_ds, config)
                _write_epsilon_profile(
                    result, ds, output_file, input_file.name, profile_idx, config
                )
                results.append((input_file, output_file, profile_idx, None))
            except Exception as e:
                results.append((input_file, None, profile_idx, str(e)))

    except Exception as e:
        results.append((input_file, None, -1, str(e)))

    return results


def _ensure_output_dir(output_dir: Optional[Union[str, Path]]) -> Path:
    """Resolve and create the output directory (defaults to cwd)."""
    if output_dir is None:
        return Path.cwd()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _init_worker_logging(level: int) -> None:
    """Pool initializer: mirror the parent's logging config in each worker.

    Without this, INFO/DEBUG messages emitted from inside ``_process_file``
    (e.g., the despike-reconcile notices) are silently dropped because each
    worker is a fresh Python process with no ``basicConfig``.
    """
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)


def _run_epsilon_pool(
    nc_files: list[Path],
    output_dir: Path,
    config: ProfileConfig,
    overwrite: bool,
    aux_ds: Optional[xr.Dataset],
    n_workers: int,
) -> list[dict]:
    """Dispatch ``_process_file`` across ``nc_files`` and collect results."""
    worker = partial(
        _process_file,
        output_dir=output_dir,
        config=config,
        overwrite=overwrite,
        aux_ds=aux_ds,
    )
    effective_workers = min(n_workers, len(nc_files))
    _log.info(f"Using {effective_workers} parallel workers for {len(nc_files)} files")

    log_level = logging.getLogger().getEffectiveLevel()
    results: list[dict] = []
    with mp.Pool(
        processes=effective_workers,
        initializer=_init_worker_logging,
        initargs=(log_level,),
    ) as pool:
        for i, file_results in enumerate(pool.imap_unordered(worker, nc_files)):
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
                status = (
                    f"profile {profile_idx} processed"
                    if success
                    else f"profile {profile_idx} failed ({error})"
                )
                _log.info(f"[{i + 1}/{len(nc_files)}] {input_path.name}: {status}")

    n_success = sum(1 for r in results if r["success"])
    n_failed = len(results) - n_success
    _log.info(
        f"Completed: {n_success} profiles succeeded, {n_failed} failed "
        f"from {len(nc_files)} files"
    )
    return results


def batch_compute_epsilon(
    files: Union[str, Path, list[Path]],
    *,
    config: Optional[ProfileConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
    auxiliary_file: Optional[Union[str, Path]] = None,
    n_workers: Optional[int] = None,
    overwrite: bool = False,
) -> list[dict]:
    """Batch compute epsilon from converted NetCDF files.

    This function processes raw p2nc output by:
    1. Detecting multiple profiles within each file (for glider data)
    2. Smoothing speed and pressure data
    3. Scaling shear probes by 1/U^2 and gradT probes by 1/U
    4. Computing epsilon using the Nasmyth spectrum fit

    Each input file may produce multiple output files if it contains multiple
    dive cycles. Output files are named ``{stem}_p{NNN}.nc``.

    Parameters
    ----------
    files : str, Path, or list of Path
        Either a glob pattern, a directory (``*.nc`` is appended), or a list
        of Path objects.
    config : ProfileConfig, optional
        Processing configuration. Defaults to ``ProfileConfig()``. All
        algorithm knobs live here.
    output_dir : str or Path, optional
        Directory for output NetCDF files. Defaults to the current directory.
    auxiliary_file : str or Path, optional
        Auxiliary NetCDF file (lat, lon, T, S, density time series) to
        interpolate onto each profile.
    n_workers : int, optional
        Number of parallel workers. Defaults to ``mp.cpu_count()``.
    overwrite : bool, optional
        Whether to overwrite existing output files. Default False.

    Returns
    -------
    list of dict
        Per-profile results with keys ``input``, ``output``, ``profile_index``,
        ``success``, ``error``.

    Examples
    --------
    >>> from pyturb.profile import ProfileConfig
    >>> from pyturb.processing import batch_compute_epsilon
    >>> cfg = ProfileConfig(profile_direction="both", min_speed=0.15)
    >>> batch_compute_epsilon("/data/*.nc", config=cfg, output_dir="/out")
    """
    nc_files = resolve_input_files(files, "*.nc")
    if not nc_files:
        _log.info("No NetCDF files found.")
        return []
    _log.info(f"Found {len(nc_files)} NetCDF files to process")

    if config is None:
        config = ProfileConfig()

    output_dir = _ensure_output_dir(output_dir)

    aux_ds = (
        load_auxiliary(auxiliary_file, config) if auxiliary_file is not None else None
    )

    if n_workers is None:
        n_workers = mp.cpu_count()

    return _run_epsilon_pool(nc_files, output_dir, config, overwrite, aux_ds, n_workers)


_QC_SENTINEL_EXCLUDED = np.int8(-1)
_QC_MISSING = np.int8(9)
_EPS_AGREEMENT_FACTOR = 10.0


def _mask_low_quality_eps(
    ds: xr.Dataset,
    probes: tuple[str, ...],
    questionable_thresh: float,
    bad_thresh: float,
) -> xr.Dataset:
    """NaN out epsilon (and sentinel its QC) using separate questionable / bad
    rejection thresholds.

    A QC-flagged questionable (qc=2) window is excluded when its epsilon
    exceeds ``questionable_thresh``; a QC-flagged bad (qc=4) window is
    excluded when its epsilon exceeds ``bad_thresh``. Below the respective
    threshold the value is kept (low-epsilon flagged windows are usually
    noise-floor artifacts rather than instrument problems). The QC sentinel
    ``-1`` is a stand-in for "excluded from binning"; it is mapped back to
    9 (missing) after groupby_bins. Using a sentinel lets the per-bin ``max``
    aggregator ignore excluded windows naturally (any kept flag is >= 0 and
    dominates).
    """
    for probe in probes:
        eps_name = f"eps_{probe}"
        qc_name = f"eps_{probe}_qc"
        if eps_name not in ds or qc_name not in ds:
            continue
        eps = ds[eps_name].values.copy()
        qc = ds[qc_name].values.astype("i1", copy=True)
        excluded = ((qc == 2) & (eps > questionable_thresh)) | (
            (qc == 4) & (eps > bad_thresh)
        )
        if not excluded.any():
            continue
        eps[excluded] = np.nan
        qc[excluded] = _QC_SENTINEL_EXCLUDED
        ds[eps_name] = (ds[eps_name].dims, eps)
        ds[qc_name] = (ds[qc_name].dims, qc)
    return ds


def _restore_qc_missing(ds: xr.Dataset, qc_vars: list[str]) -> xr.Dataset:
    """Map sentinel ``-1`` and empty-bin NaN values in QC vars back to 9.

    After groupby_bins.max(), QC vars may contain:
      * -1 (sentinel): all contributing windows were excluded → missing
      * NaN: the bin had no contributing windows at all → missing
      * one of {0, 1, 2, 4, 9}: a real flag from at least one window
    """
    for v in qc_vars:
        if v not in ds:
            continue
        arr = ds[v].values
        out = np.where(np.isnan(arr) | (arr == _QC_SENTINEL_EXCLUDED), _QC_MISSING, arr)
        ds[v] = (ds[v].dims, out.astype("i1"))
    return ds


def _combine_eps_pair(
    eps1: np.ndarray, eps2: np.ndarray, qc1: np.ndarray, qc2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Combine two probe estimates into (eps_best, eps_qc).

    eps_best:
      * both finite & within factor of ``_EPS_AGREEMENT_FACTOR`` → mean
      * both finite, disagreement larger → element-wise minimum
      * exactly one finite → that value
      * neither finite → NaN

    eps_qc:
      * both eps finite → max(qc1, qc2)
      * exactly one finite → the surviving probe's qc
      * neither finite → 9 (missing)
    """
    e1_ok = np.isfinite(eps1)
    e2_ok = np.isfinite(eps2)
    both = e1_ok & e2_ok

    hi = np.fmax(eps1, eps2)
    lo = np.fmin(eps1, eps2)
    within = both & (hi <= _EPS_AGREEMENT_FACTOR * lo)

    eps_best = np.where(
        within,
        0.5 * (eps1 + eps2),
        np.where(both, lo, np.where(e1_ok, eps1, np.where(e2_ok, eps2, np.nan))),
    )

    eps_qc = np.full(eps1.shape, _QC_MISSING, dtype="i1")
    only1 = e1_ok & ~e2_ok
    only2 = e2_ok & ~e1_ok
    eps_qc[only1] = qc1[only1]
    eps_qc[only2] = qc2[only2]
    eps_qc[both] = np.maximum(qc1[both], qc2[both])
    return eps_best.astype("f4"), eps_qc


def _attach_combined_eps(ds: xr.Dataset) -> xr.Dataset:
    """Build ``eps_best`` and ``eps_qc`` from binned ``eps_1``/``eps_2`` (+ QCs).

    No-op if either probe's eps or QC variable is missing from the dataset.
    """
    needed = ("eps_1", "eps_2", "eps_1_qc", "eps_2_qc")
    if any(v not in ds for v in needed):
        return ds
    e1 = ds["eps_1"].values
    e2 = ds["eps_2"].values
    q1 = ds["eps_1_qc"].values.astype("i1")
    q2 = ds["eps_2_qc"].values.astype("i1")
    eps_best, eps_qc = _combine_eps_pair(e1, e2, q1, q2)
    dims = ds["eps_1"].dims
    ds["eps_best"] = (dims, eps_best)
    ds["eps_best"].attrs = {
        "long_name": "Best epsilon estimate combined from eps_1 and eps_2",
        "units": "W kg-1",
        "comment": (
            "Mean of eps_1 and eps_2 where they agree within a factor of "
            f"{_EPS_AGREEMENT_FACTOR:g}; element-wise minimum otherwise. "
            "Falls back to the single surviving probe when one is missing."
        ),
    }
    ds["eps_qc"] = (dims, eps_qc)
    ds["eps_qc"].attrs = {
        "long_name": "Combined QC flag for eps_best",
        "flag_values": np.array([0, 1, 2, 4, 9], dtype="i1"),
        "flag_meanings": "unknown good questionable bad missing",
        "valid_min": np.int8(0),
        "valid_max": np.int8(9),
        "comment": (
            "Per-bin max of eps_1_qc and eps_2_qc where both probes have a "
            "finite epsilon; the surviving probe's flag when one is missing; "
            "9 (missing) when both are missing."
        ),
    }
    return ds


def _bin_single_profile(
    file: Path,
    depth_bins: np.ndarray,
    variables: list[str],
    default_latitude: float = 45.0,
    bin_by_pressure: bool = False,
    questionable_thresh: float = 1e-7,
    bad_thresh: float = 1e-9,
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
    questionable_thresh : float, default 1e-7
        Epsilon threshold above which qc=2 (questionable) windows are dropped
        before binning. Pass ``inf`` to keep them all.
    bad_thresh : float, default 1e-9
        Epsilon threshold above which qc=4 (bad) windows are dropped before
        binning. Pass ``inf`` to keep them all.

    Returns binned dataset or None if an error occurs.
    """
    try:
        ds = xr.load_dataset(file, decode_times=False)

        # Pre-bin masking: drop high-eps windows flagged questionable/bad
        # (those are likely real instrument problems, not noise-floor noise).
        ds = _mask_low_quality_eps(ds, ("1", "2"), questionable_thresh, bad_thresh)

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

        # Group by bins. QC vars (suffix "_qc") take the worst (max) flag in
        # the bin; everything else is mean-averaged.
        qc_vars = [v for v in vars_to_bin_with_time if v.endswith("_qc")]
        mean_vars = [v for v in vars_to_bin_with_time if v not in qc_vars]

        grouped_mean = ds_subset[mean_vars].groupby_bins(bin_var, bins=depth_bins)
        ds_binned = grouped_mean.mean()
        if qc_vars:
            grouped_max = ds_subset[qc_vars].groupby_bins(bin_var, bins=depth_bins)
            ds_binned = xr.merge([ds_binned, grouped_max.max()])
            # Post-bin: restore sentinel -1 (all windows excluded) and any
            # empty-bin NaN to 9 (missing) so the on-disk QC is always a
            # valid IODE flag in {0, 1, 2, 4, 9}.
            ds_binned = _restore_qc_missing(ds_binned, qc_vars)

        # Build combined eps_best / eps_qc from binned probe pair (if present).
        ds_binned = _attach_combined_eps(ds_binned)

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
        _log.error(f"Error binning {file}: {e}")
        return None


def _unpack_bin_args(args: tuple) -> Optional[xr.Dataset]:
    """Unpack arguments for imap_unordered."""
    (
        file,
        depth_bins,
        variables,
        default_latitude,
        bin_by_pressure,
        questionable_thresh,
        bad_thresh,
    ) = args
    return _bin_single_profile(
        file,
        depth_bins,
        variables,
        default_latitude,
        bin_by_pressure,
        questionable_thresh,
        bad_thresh,
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
    questionable_thresh: float = 1e-7,
    bad_thresh: float = 1e-9,
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
    questionable_thresh : float, default 1e-7
        Drop qc=2 (questionable) windows whose epsilon exceeds this value
        before binning. Pass ``inf`` to keep them all.
    bad_thresh : float, default 1e-9
        Drop qc=4 (bad) windows whose epsilon exceeds this value before
        binning. Pass ``inf`` to keep them all.

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
    # Default variables to bin. Names ending in "_qc" are aggregated with
    # max (worst flag wins per bin); others with mean.
    if variables is None:
        variables = [
            "eps_1",
            "eps_2",
            "eps_1_fm",
            "eps_2_fm",
            "eps_1_qc",
            "eps_2_qc",
            "sh1_despike_frac",
            "sh2_despike_frac",
            "gradT1_despike_frac",
            "gradT2_despike_frac",
            "W",
            "temperature",
            "conductivity",
            "salinity",
            "density",
            "nu",
            "lat",
            "lon",
        ]

    nc_files = resolve_input_files(files, "*.nc")
    if not nc_files:
        _log.info("No NetCDF files found.")
        return None

    _log.info(f"Found {len(nc_files)} NetCDF files to bin")
    coord_type = "pressure" if bin_by_pressure else "depth"
    _log.info(
        f"Binning by {coord_type} from {depth_min} to {depth_max} m "
        f"with {bin_width} m bins"
    )

    # Create depth (or pressure) bins
    depth_bins = np.arange(depth_min, depth_max + bin_width, bin_width)

    if n_workers is None:
        n_workers = mp.cpu_count()

    args = [
        (
            f,
            depth_bins,
            variables,
            default_latitude,
            bin_by_pressure,
            questionable_thresh,
            bad_thresh,
        )
        for f in nc_files
    ]

    binned_datasets = []

    # Use serial processing for small batches
    if len(nc_files) <= min(n_workers, 4):
        _log.info("Using serial processing for small batch")
        for i, arg_tuple in enumerate(args):
            result = _unpack_bin_args(arg_tuple)
            if result is not None:
                binned_datasets.append(result)
            status = "binned" if result is not None else "skipped"
            _log.info(f"[{i + 1}/{len(nc_files)}] {status}: {nc_files[i].name}")
    else:
        _log.info(f"Using {n_workers} parallel workers")
        with mp.Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap(_unpack_bin_args, args)):
                if result is not None:
                    binned_datasets.append(result)
                status = "binned" if result is not None else "skipped"
                _log.info(f"[{i + 1}/{len(nc_files)}] {status}: {nc_files[i].name}")

    if not binned_datasets:
        _log.info("No datasets were successfully binned.")
        return None

    _log.info(f"Concatenating {len(binned_datasets)} binned profiles...")

    # Concatenate along profile dimension
    combined = xr.concat(binned_datasets, dim="profile")

    # Sort profiles by time (use minimum time per profile to handle NaT values)
    if "time" in combined:
        # Get representative time for each profile (min time, skipping NaT)
        profile_times = combined.time.min(dim="depth", skipna=True)
        # Sort by time
        sort_order = np.argsort(profile_times.values)
        combined = combined.isel(profile=sort_order)
        _log.info("Sorted profiles by time")

    # Save to file
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_netcdf(output_file)

    _log.info(f"Saved binned data to {output_file}")

    return combined
