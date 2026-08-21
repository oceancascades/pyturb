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
    _EPS_AGREEMENT_FACTOR,
    ProfileConfig,
    _combine_eps_pair,
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

    Keeps time- and ctd_time-dimensioned data variables, plus the scalar
    ``lat``/``lon`` attached for a stationary platform (see
    :func:`~pyturb.profile._attach_scalar_position`). Reattaches the
    ``frequency`` and ``k`` coordinates, carries over ``time``/``ctd_time``
    attrs, and stamps source file / profile / instrument metadata.
    """
    vars_to_keep = [
        v
        for v in result.data_vars
        if "time" in result[v].dims
        or "ctd_time" in result[v].dims
        or (v in ("lat", "lon") and len(result[v].dims) == 0)
    ]
    out = result[vars_to_keep].assign_coords(
        frequency=result.frequency,
        k=result.k,
    )

    for coord in ("time", "ctd_time"):
        if coord in result.coords and coord in out.coords:
            for attr in ("units", "long_name"):
                if attr in result[coord].attrs:
                    out[coord].attrs[attr] = result[coord].attrs[attr]

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


def _has_existing_outputs(stem: str, output_dir: Path) -> bool:
    """True if any file in output_dir has a name starting with stem."""
    if not output_dir.exists():
        return False
    return any(p.name.startswith(stem) for p in output_dir.iterdir() if p.is_file())


def _process_file(
    input_file: Path,
    output_dir: Path,
    config: ProfileConfig,
    overwrite: bool,
    aux_ds: Optional[xr.Dataset] = None,
    skip_existing: bool = False,
) -> list[tuple]:
    """Process a file that may contain multiple profiles.

    Returns list of (input_path, output_path, profile_index, error) tuples.
    """
    results: list[tuple[Path, Optional[Path], int, Optional[str]]] = []
    stem = input_file.stem

    if skip_existing and not overwrite and _has_existing_outputs(stem, output_dir):
        _log.info(f"Skipping {input_file.name}: outputs already exist for '{stem}'")
        return [(input_file, None, -1, "skipped (existing outputs found)")]

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
    skip_existing: bool = False,
) -> list[dict]:
    """Dispatch ``_process_file`` across ``nc_files`` and collect results."""
    worker = partial(
        _process_file,
        output_dir=output_dir,
        config=config,
        overwrite=overwrite,
        aux_ds=aux_ds,
        skip_existing=skip_existing,
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
    skip_existing: bool = False,
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
    skip_existing : bool, optional
        Skip a file entirely, without detecting profiles, if any file in
        ``output_dir`` already starts with its stem. Faster than the default
        per-profile overwrite check, which still has to load and split the
        file to find its profile names. Ignored if ``overwrite`` is True.
        Default False.

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

    return _run_epsilon_pool(
        nc_files, output_dir, config, overwrite, aux_ds, n_workers, skip_existing
    )


_QC_SENTINEL_EXCLUDED = np.int8(-1)
_QC_MISSING = np.int8(9)


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


def _attach_combined_eps(ds: xr.Dataset) -> xr.Dataset:
    """Build ``eps`` and ``eps_qc`` from binned ``eps_1``/``eps_2`` (+ QCs).

    No-op if either probe's eps or QC variable is missing from the dataset.
    """
    needed = ("eps_1", "eps_2", "eps_1_qc", "eps_2_qc")
    if any(v not in ds for v in needed):
        return ds
    e1 = ds["eps_1"].values
    e2 = ds["eps_2"].values
    q1 = ds["eps_1_qc"].values.astype("i1")
    q2 = ds["eps_2_qc"].values.astype("i1")
    eps, eps_qc = _combine_eps_pair(e1, e2, q1, q2)
    dims = ds["eps_1"].dims
    ds["eps"] = (dims, eps)
    ds["eps"].attrs = {
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
        "long_name": "Combined QC flag for eps",
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


def _attach_combined_chi(ds: xr.Dataset) -> xr.Dataset:
    """Build ``chi`` and ``chi_qc`` from binned ``chi_1``/``chi_2`` (+ QCs).

    Uses the same combination rule as epsilon. No-op if either probe's chi
    or QC variable is missing from the dataset.
    """
    needed = ("chi_1", "chi_2", "chi_1_qc", "chi_2_qc")
    if any(v not in ds for v in needed):
        return ds
    c1 = ds["chi_1"].values
    c2 = ds["chi_2"].values
    q1 = ds["chi_1_qc"].values.astype("i1")
    q2 = ds["chi_2_qc"].values.astype("i1")
    chi, chi_qc = _combine_eps_pair(c1, c2, q1, q2)
    dims = ds["chi_1"].dims
    ds["chi"] = (dims, chi)
    ds["chi"].attrs = {
        "long_name": "Best chi estimate combined from chi_1 and chi_2",
        "units": "K2 s-1",
        "comment": (
            "Mean of chi_1 and chi_2 where they agree within a factor of "
            f"{_EPS_AGREEMENT_FACTOR:g}; element-wise minimum otherwise. "
            "Falls back to the single surviving probe when one is missing."
        ),
    }
    ds["chi_qc"] = (dims, chi_qc)
    ds["chi_qc"].attrs = {
        "long_name": "Combined QC flag for chi",
        "flag_values": np.array([0, 1, 2, 4, 9], dtype="i1"),
        "flag_meanings": "unknown good questionable bad missing",
        "valid_min": np.int8(0),
        "valid_max": np.int8(9),
        "comment": (
            "Per-bin max of chi_1_qc and chi_2_qc where both probes have a "
            "finite chi; the surviving probe's flag when one is missing; "
            "9 (missing) when both are missing."
        ),
    }
    return ds


def _depth_from_pressure(
    pressure: np.ndarray,
    lat: Optional[np.ndarray],
    default_latitude: float,
) -> np.ndarray:
    """Depth (m) computed from pressure via gsw."""
    if lat is not None and not np.all(np.isnan(lat)):
        lat_val = np.nanmean(lat)
    else:
        lat_val = default_latitude
    return -gsw.z_from_p(pressure, lat_val)


_EPOCH = np.datetime64("1970-01-01T00:00:00", "ns")


def _time_to_epoch_seconds(ds: xr.Dataset, time_dim: str) -> np.ndarray:
    """Absolute float seconds since 1970-01-01 for a raw (per-file relative)
    time coordinate, NaT-safe.

    Eps files are loaded with ``decode_times=False`` (raw floats relative to
    each file's own ``filetime``), so different profile files are not
    directly comparable as numbers. This decodes using the coordinate's own
    ``units``/``calendar`` attrs to get an absolute instant, then converts to
    a fixed, common epoch so the result can be safely averaged (via
    groupby-mean) and compared across files.
    """
    decoded = xr.decode_cf(ds[[time_dim]])[time_dim].values.astype("datetime64[ns]")
    return (decoded - _EPOCH) / np.timedelta64(1, "s")


def _bin_var_group(
    ds: xr.Dataset,
    time_dim: str,
    pressure_name: str,
    var_names: list[str],
    rename_map: dict[str, str],
    depth_bins: np.ndarray,
    default_latitude: float,
    coord_name: str,
    include_time: bool,
) -> Optional[xr.Dataset]:
    """Bin var_names (all on time_dim) against depth_bins using pressure_name.

    QC vars (suffix "_qc") take the worst (max) flag in the bin; everything
    else is mean-averaged. Output vars are renamed per rename_map.
    """
    if pressure_name not in ds:
        return None
    present = [v for v in var_names if v in ds]
    if not present:
        return None

    lat = ds["lat"].values if "lat" in ds else None
    bin_values = _depth_from_pressure(ds[pressure_name].values, lat, default_latitude)

    ds_subset = ds[present].copy()
    bin_var_name = f"_bin_var_{time_dim}"
    ds_subset[bin_var_name] = (time_dim, bin_values)

    if include_time and time_dim in ds.coords:
        ds_subset["time_var"] = (time_dim, _time_to_epoch_seconds(ds, time_dim))
        present = present + ["time_var"]

    qc_vars = [v for v in present if v.endswith("_qc")]
    mean_vars = [v for v in present if v not in qc_vars]

    grouped_mean = ds_subset[mean_vars].groupby_bins(
        ds_subset[bin_var_name], bins=depth_bins
    )
    ds_binned = grouped_mean.mean()
    if qc_vars:
        grouped_max = ds_subset[qc_vars].groupby_bins(
            ds_subset[bin_var_name], bins=depth_bins
        )
        ds_binned = xr.merge([ds_binned, grouped_max.max()])
        # Post-bin: restore sentinel -1 (all windows excluded) and any
        # empty-bin NaN to 9 (missing) so the on-disk QC is always a valid
        # IODE flag in {0, 1, 2, 4, 9}.
        ds_binned = _restore_qc_missing(ds_binned, qc_vars)

    bin_name = f"{bin_var_name}_bins"
    ds_binned[bin_name] = np.array(
        [interval.mid for interval in ds_binned[bin_name].values]
    )
    ds_binned = ds_binned.rename({bin_name: coord_name})
    if rename_map:
        ds_binned = ds_binned.rename(
            {k: v for k, v in rename_map.items() if k in ds_binned}
        )
    return ds_binned


def _bin_single_profile(
    file: Path,
    depth_bins: np.ndarray,
    variables: list[str],
    default_latitude: float = 45.0,
    questionable_thresh: float = 1e-7,
    bad_thresh: float = 1e-9,
    ctd_depth_bins: Optional[np.ndarray] = None,
) -> Optional[xr.Dataset]:
    """
    Bin a single profile dataset by depth.

    Any requested variable with a higher-resolution ``<var>_hires`` (on
    ``ctd_time``) counterpart is binned from that instead of the coarser
    dissipation-bin version, keeping the requested output name.

    Parameters
    ----------
    file : Path
        Path to the NetCDF file.
    depth_bins : np.ndarray
        Bin edges for depth.
    variables : list of str
        Variables to include in the binned output.
    default_latitude : float
        Default latitude for pressure-to-depth conversion if not in data.
    questionable_thresh : float, default 1e-7
        Epsilon threshold above which qc=2 (questionable) windows are dropped
        before binning. Pass ``inf`` to keep them all.
    bad_thresh : float, default 1e-9
        Epsilon threshold above which qc=4 (bad) windows are dropped before
        binning. Pass ``inf`` to keep them all.
    ctd_depth_bins : np.ndarray, optional
        If given, also bins the ``_hires`` CTD variables onto this separate,
        typically finer grid, keeping the ``_hires`` suffix and attached on
        a separate ``ctd_depth`` coordinate.

    Returns binned dataset or None if an error occurs.
    """
    try:
        ds = xr.load_dataset(file, decode_times=False)

        # Pre-bin masking: drop high-eps windows flagged questionable/bad
        # (those are likely real instrument problems, not noise-floor noise).
        ds = _mask_low_quality_eps(ds, ("1", "2"), questionable_thresh, bad_thresh)

        coarse_vars = [v for v in variables if v in ds and f"{v}_hires" not in ds]
        hires_bases = [v for v in variables if f"{v}_hires" in ds]
        hires_names = [f"{v}_hires" for v in hires_bases]

        if not coarse_vars and not hires_bases:
            return None

        pieces = []
        if coarse_vars:
            piece = _bin_var_group(
                ds,
                "time",
                "pressure",
                coarse_vars,
                {},
                depth_bins,
                default_latitude,
                "depth",
                include_time=True,
            )
            if piece is not None:
                pieces.append(piece)

        if hires_bases:
            piece = _bin_var_group(
                ds,
                "ctd_time",
                "pressure_hires",
                hires_names,
                {f"{v}_hires": v for v in hires_bases},
                depth_bins,
                default_latitude,
                "depth",
                include_time=False,
            )
            if piece is not None:
                pieces.append(piece)

        if not pieces:
            return None
        ds_binned = (
            pieces[0] if len(pieces) == 1 else xr.merge(pieces, compat="override")
        )

        # Build combined eps / eps_qc and chi / chi_qc from binned probe pairs.
        ds_binned = _attach_combined_eps(ds_binned)
        ds_binned = _attach_combined_chi(ds_binned)

        # Rename time_var back to time if it exists and add epoch units
        if "time_var" in ds_binned:
            ds_binned = ds_binned.rename({"time_var": "time"})
            ds_binned["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"
            ds_binned["time"].attrs["long_name"] = "Time"
            ds_binned["time"].attrs["calendar"] = "proleptic_gregorian"

        # Separate, finer CTD-only grid (keeps the "_hires" names).
        if ctd_depth_bins is not None and hires_bases:
            piece_ctd = _bin_var_group(
                ds,
                "ctd_time",
                "pressure_hires",
                hires_names,
                {},
                ctd_depth_bins,
                default_latitude,
                "ctd_depth",
                include_time=False,
            )
            if piece_ctd is not None:
                ds_binned = xr.merge([ds_binned, piece_ctd], compat="override")

        # Add source file as attribute
        ds_binned.attrs["source_file"] = file.name

        # Add instrument serial number as a data variable (to be used as coordinate)
        if "instrument_sn" in ds.attrs:
            ds_binned["instrument_sn"] = ds.attrs["instrument_sn"]
        if "instrument_vehicle" in ds.attrs:
            ds_binned["instrument_vehicle"] = ds.attrs["instrument_vehicle"]

        return ds_binned

    except Exception:
        _log.error(f"Error binning {file}, skipping.")
        return None


def _unpack_bin_args(args: tuple) -> Optional[xr.Dataset]:
    """Unpack arguments for imap_unordered."""
    (
        file,
        depth_bins,
        variables,
        default_latitude,
        questionable_thresh,
        bad_thresh,
        ctd_depth_bins,
    ) = args
    return _bin_single_profile(
        file,
        depth_bins,
        variables,
        default_latitude,
        questionable_thresh,
        bad_thresh,
        ctd_depth_bins,
    )


def bin_profiles(
    files: Union[str, Path, list[Path]],
    output_file: Union[str, Path] = "binned_profiles.nc",
    depth_min: float = 0.0,
    depth_max: float = 1000.0,
    bin_width: float = 2.0,
    variables: Optional[list[str]] = None,
    default_latitude: float = 45.0,
    n_workers: Optional[int] = None,
    questionable_thresh: float = 1e-7,
    bad_thresh: float = 1e-9,
    ctd_bin_width: Optional[float] = None,
) -> Optional[xr.Dataset]:
    """
    Bin multiple profile datasets by depth and concatenate.

    This function reads epsilon output files, bins them by depth, and
    concatenates them along a 'profile' dimension. Depth is calculated from
    pressure using gsw.z_from_p().

    Any requested variable with a higher-resolution ``<var>_hires`` version
    (from ``eps``'s ``ctd_bin_sec``) is binned from that instead of the
    coarser dissipation-bin version, keeping the requested output name.

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
    n_workers : int, optional
        Number of parallel workers. Default is number of CPU cores.
    questionable_thresh : float, default 1e-7
        Drop qc=2 (questionable) windows whose epsilon exceeds this value
        before binning. Pass ``inf`` to keep them all.
    bad_thresh : float, default 1e-9
        Drop qc=4 (bad) windows whose epsilon exceeds this value before
        binning. Pass ``inf`` to keep them all.
    ctd_bin_width : float, optional
        If given, also bins the ``_hires`` CTD variables onto a separate,
        typically finer grid of this width, over the same depth_min/max
        range. Output on a separate ``ctd_depth`` coordinate, keeping the
        ``_hires`` suffix. Default None (skipped).

    Returns
    -------
    xr.Dataset
        Binned and concatenated dataset with dimensions (profile, depth).

    Examples
    --------
    >>> from pyturb.processing import bin_profiles
    >>> ds = bin_profiles('/path/to/eps_output/*.nc', output_file='binned.nc')
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
            "chi_1",
            "chi_2",
            "chi_1_fm",
            "chi_2_fm",
            "chi_1_qc",
            "chi_2_qc",
            "sh1_despike_frac",
            "sh2_despike_frac",
            "gradT1_despike_frac",
            "gradT2_despike_frac",
            "W",
            "temperature",
            "conductivity",
            "salinity",
            "density",
            "z",
            "absolute_salinity",
            "conservative_temperature",
            "potential_density",
            "N2",
            "nu",
            "lat",
            "lon",
        ]

    nc_files = resolve_input_files(files, "*.nc")
    if not nc_files:
        _log.info("No NetCDF files found.")
        return None

    _log.info(f"Found {len(nc_files)} NetCDF files to bin")
    _log.info(
        f"Binning by depth from {depth_min} to {depth_max} m with {bin_width} m bins"
    )

    # Create depth bins
    depth_bins = np.arange(depth_min, depth_max + bin_width, bin_width)

    ctd_depth_bins = None
    if ctd_bin_width is not None:
        ctd_depth_bins = np.arange(depth_min, depth_max + ctd_bin_width, ctd_bin_width)
        _log.info(f"Also binning CTD variables at {ctd_bin_width} m resolution")

    if n_workers is None:
        n_workers = mp.cpu_count()

    args = [
        (
            f,
            depth_bins,
            variables,
            default_latitude,
            questionable_thresh,
            bad_thresh,
            ctd_depth_bins,
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
    # and attach it as a "profile_time" coordinate so profiles are directly
    # organizable/selectable by start time even where per-depth-bin "time"
    # is patchy (missing/NaT bins near the surface, seafloor, etc).
    if "time" in combined:
        # Get representative time for each profile (min time, skipping NaT)
        profile_times = combined.time.min(dim="depth", skipna=True)
        # Sort by time
        sort_order = np.argsort(profile_times.values)
        combined = combined.isel(profile=sort_order)
        combined = combined.assign_coords(
            profile_time=("profile", profile_times.values[sort_order])
        )
        combined["profile_time"].attrs = {
            "long_name": "Profile start time",
            "units": "seconds since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        }
        _log.info("Sorted profiles by time")

    # Save to file
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_netcdf(output_file)

    _log.info(f"Saved binned data to {output_file}")

    return combined
