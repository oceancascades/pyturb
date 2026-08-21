"""Lightweight profile detection index.

Detecting profiles only needs the slow-channel pressure record, but
materializing full-resolution profile data means copying the (much larger)
fast-channel probe arrays. This module separates the two: `batch_index_profiles`
runs detection and writes a small per-file index of profile boundaries;
`extract_profile` uses a saved index to slice a single profile's full-resolution
data out of the original converted file on demand, without re-running detection.
"""

import logging
import multiprocessing as mp
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr

from . import __version__
from .io import load_profile_nc, resolve_input_files
from .processing import _ensure_output_dir, _init_worker_logging
from .profile import (
    ProfileConfig,
    extract_profile_by_indices,
    find_all_profiles,
    prepare_profile,
)

_log = logging.getLogger(__name__)

__all__ = ["batch_index_profiles", "extract_profile"]

_INSTRUMENT_ATTRS = ("instrument_vehicle", "instrument_model", "instrument_sn")


def _segment_direction(pressure: np.ndarray, idx_start: int, idx_end: int) -> str:
    """Return "down" if pressure increases start->end, else "up"."""
    return "down" if pressure[idx_end] >= pressure[idx_start] else "up"


def _index_from_segments(
    ds: xr.Dataset, config: ProfileConfig, segments: list[tuple[int, int]]
) -> xr.Dataset:
    pressure = ds[config.pressure_smooth].values
    t_slow = ds.t_slow.values

    n = len(segments)
    start_idx = np.empty(n, dtype="i8")
    end_idx = np.empty(n, dtype="i8")
    start_time = np.empty(n, dtype="f8")
    end_time = np.empty(n, dtype="f8")
    direction = np.empty(n, dtype=object)

    for i, (s, e) in enumerate(segments):
        start_idx[i] = s
        end_idx[i] = e
        start_time[i] = t_slow[s]
        end_time[i] = t_slow[e]
        direction[i] = _segment_direction(pressure, s, e)

    idx_ds = xr.Dataset(
        {
            "start_idx": ("profile", start_idx),
            "end_idx": ("profile", end_idx),
            "start_time": ("profile", start_time),
            "end_time": ("profile", end_time),
            "direction": ("profile", direction.astype(str)),
        },
        coords={"profile": np.arange(n)},
    )
    idx_ds["start_idx"].attrs["long_name"] = "t_slow index of profile start (inclusive)"
    idx_ds["end_idx"].attrs["long_name"] = "t_slow index of profile end (inclusive)"
    idx_ds["direction"].attrs["long_name"] = "cast direction"
    for attr in ("units", "calendar", "long_name"):
        if attr in ds.t_slow.attrs:
            idx_ds["start_time"].attrs[attr] = ds.t_slow.attrs[attr]
            idx_ds["end_time"].attrs[attr] = ds.t_slow.attrs[attr]

    return idx_ds


def build_profile_index(ds: xr.Dataset, config: ProfileConfig) -> xr.Dataset:
    """Detect profiles in ds and return a small per-profile index dataset.

    Only touches the slow-channel pressure record (via find_all_profiles);
    never slices the fast-channel probe arrays.

    Returns a Dataset with a ``profile`` dimension holding ``start_idx``,
    ``end_idx`` (inclusive t_slow indices), ``start_time``, ``end_time``,
    and ``direction`` ("down"/"up").
    """
    segments = find_all_profiles(ds, config)
    return _index_from_segments(ds, config, segments)


def _write_profile_index(
    idx_ds: xr.Dataset,
    output_file: Path,
    source_file_name: str,
    config: ProfileConfig,
    source_attrs: dict,
) -> None:
    idx_ds = idx_ds.copy()
    idx_ds.attrs["source_file"] = source_file_name
    idx_ds.attrs["pyturb_version"] = __version__
    idx_ds.attrs["pyturb_processed_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    idx_ds.attrs["pyturb_config"] = config.to_yaml()
    for attr in _INSTRUMENT_ATTRS:
        if attr in source_attrs:
            idx_ds.attrs[attr] = source_attrs[attr]

    idx_ds.to_netcdf(output_file, format="NETCDF4")


def _write_hires_profile(
    profile_ds: xr.Dataset,
    output_file: Path,
    source_file_name: str,
    profile_idx: int,
    direction: str,
    config: ProfileConfig,
    compress: bool,
    compression_level: int,
) -> None:
    out = profile_ds.copy()
    out.attrs["source_file"] = source_file_name
    out.attrs["profile_index"] = profile_idx
    out.attrs["profile_direction"] = direction
    out.attrs["pyturb_version"] = __version__
    out.attrs["pyturb_processed_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    out.attrs["pyturb_config"] = config.to_yaml()

    encoding = {}
    if compress:
        for var in out.data_vars:
            encoding[var] = {
                "zlib": True,
                "complevel": compression_level,
                "dtype": "float32",
            }
        for coord in ("t_fast", "t_slow"):
            if coord in out.coords:
                encoding[coord] = {"zlib": True, "complevel": compression_level}

    out.to_netcdf(output_file, encoding=encoding, format="NETCDF4")


def _materialize_profiles(
    prepared: xr.Dataset,
    idx_ds: xr.Dataset,
    output_dir: Path,
    stem: str,
    source_file_name: str,
    config: ProfileConfig,
    overwrite: bool,
    compress: bool,
    compression_level: int,
) -> list[tuple[Path, Optional[str]]]:
    """Write each indexed profile's full-resolution data to its own NetCDF file.

    Output files are named ``{stem}_p{NNNN}_hires.nc``. Returns a list of
    (output_file, error) tuples, one per detected profile.
    """
    results: list[tuple[Path, Optional[str]]] = []
    for i in range(idx_ds.sizes["profile"]):
        output_file = output_dir / f"{stem}_p{i:04d}_hires.nc"
        if output_file.exists() and not overwrite:
            results.append((output_file, "skipped (exists)"))
            continue
        try:
            idx_start = int(idx_ds["start_idx"].values[i])
            idx_end = int(idx_ds["end_idx"].values[i])
            direction = str(idx_ds["direction"].values[i])
            profile_ds = extract_profile_by_indices(prepared, idx_start, idx_end)
            _write_hires_profile(
                profile_ds,
                output_file,
                source_file_name,
                i,
                direction,
                config,
                compress,
                compression_level,
            )
            results.append((output_file, None))
        except Exception as e:
            results.append((output_file, str(e)))
    return results


def _process_file_for_index(
    input_file: Path,
    output_dir: Path,
    config: ProfileConfig,
    overwrite: bool,
    materialize: bool = False,
    compress: bool = False,
    compression_level: int = 4,
) -> tuple:
    """Detect profiles in one file, write its index, and optionally materialize
    each profile's full-resolution data.

    Returns (input, index_file, error, hires_results), where hires_results is
    a list of (output_file, error) tuples -- empty unless materialize=True.
    """
    index_file = output_dir / f"{input_file.stem}_profiles.nc"

    if index_file.exists() and not overwrite and not materialize:
        return (input_file, index_file, "skipped (exists)", [])

    try:
        ds = load_profile_nc(input_file)
        prepared = prepare_profile(ds, config)
        idx_ds = build_profile_index(prepared, config)

        if idx_ds.sizes["profile"] == 0:
            _log.info("No profiles detected; indexing the whole file as one profile.")
            n_slow = prepared.sizes["t_slow"]
            idx_ds = _index_from_segments(prepared, config, [(0, n_slow - 1)])

        if not index_file.exists() or overwrite:
            _write_profile_index(idx_ds, index_file, input_file.name, config, ds.attrs)

        hires_results: list[tuple[Path, Optional[str]]] = []
        if materialize:
            hires_results = _materialize_profiles(
                prepared,
                idx_ds,
                output_dir,
                input_file.stem,
                input_file.name,
                config,
                overwrite,
                compress,
                compression_level,
            )

        return (input_file, index_file, None, hires_results)
    except Exception as e:
        return (input_file, None, str(e), [])


def batch_index_profiles(
    files: Union[str, Path, list[Path]],
    *,
    config: Optional[ProfileConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
    n_workers: Optional[int] = None,
    overwrite: bool = False,
    materialize: bool = False,
    compress: bool = False,
    compression_level: int = 4,
) -> list[dict]:
    """Detect profiles in converted NetCDF files and write a boundary index for each.

    Output files are named ``{stem}_profiles.nc`` and contain the detected
    profile boundaries (t_slow indices and times) plus the detection config,
    without copying any fast-channel probe data. Use :func:`extract_profile`
    to pull a single profile's full-resolution data out of the original file
    using a saved index.

    Parameters
    ----------
    files : str, Path, or list of Path
        Either a glob pattern, a directory (``*.nc`` is appended), or a list
        of Path objects.
    config : ProfileConfig, optional
        Profile-detection configuration. Defaults to ``ProfileConfig()``.
    output_dir : str or Path, optional
        Directory for output index files. Defaults to the current directory.
    n_workers : int, optional
        Number of parallel workers. Defaults to ``mp.cpu_count()``.
    overwrite : bool, optional
        Whether to overwrite an existing index or hires file. Default False.
    materialize : bool, optional
        Also write each detected profile's full-resolution data to its own
        ``{stem}_p{NNNN}_hires.nc`` file. Default False.
    compress : bool, optional
        Compress materialized profile NetCDF output. Only used when
        ``materialize`` is True. Default False.
    compression_level : int, optional
        Compression level (1-9) for materialized output. Default 4.

    Returns
    -------
    list of dict
        Per-file results with keys ``input``, ``output``, ``success``,
        ``error``, ``materialized`` (list of (output_file, error) tuples,
        one per detected profile, empty unless ``materialize`` is True).
    """
    nc_files = resolve_input_files(files, "*.nc")
    if not nc_files:
        _log.info("No NetCDF files found.")
        return []
    _log.info(f"Found {len(nc_files)} NetCDF files to index")

    if config is None:
        config = ProfileConfig()

    output_dir = _ensure_output_dir(output_dir)

    if n_workers is None:
        n_workers = mp.cpu_count()
    effective_workers = min(n_workers, len(nc_files))

    worker = partial(
        _process_file_for_index,
        output_dir=output_dir,
        config=config,
        overwrite=overwrite,
        materialize=materialize,
        compress=compress,
        compression_level=compression_level,
    )

    log_level = logging.getLogger().getEffectiveLevel()
    results: list[dict] = []
    with mp.Pool(
        processes=effective_workers,
        initializer=_init_worker_logging,
        initargs=(log_level,),
    ) as pool:
        for i, (input_path, output_path, error, hires_results) in enumerate(
            pool.imap_unordered(worker, nc_files)
        ):
            success = error is None
            results.append(
                {
                    "input": input_path,
                    "output": output_path,
                    "success": success,
                    "error": error,
                    "materialized": hires_results,
                }
            )
            status = "indexed" if success else f"failed ({error})"
            if hires_results:
                n_ok = sum(1 for _, e in hires_results if e is None)
                status += f", materialized {n_ok}/{len(hires_results)} profiles"
            _log.info(f"[{i + 1}/{len(nc_files)}] {input_path.name}: {status}")

    n_success = sum(1 for r in results if r["success"])
    _log.info(
        f"Completed: {n_success} succeeded, {len(results) - n_success} failed "
        f"from {len(nc_files)} files"
    )
    return results


def extract_profile(
    indices_file: Union[str, Path],
    raw_file: Union[str, Path],
    profile_index: int,
) -> xr.Dataset:
    """Extract one profile's full-resolution data using a saved index.

    Looks up ``profile_index`` in ``indices_file`` (written by
    :func:`batch_index_profiles`) and slices the matching profile out of
    ``raw_file`` directly, without re-running detection.
    """
    idx_ds = xr.load_dataset(indices_file)
    if profile_index not in idx_ds.profile.values:
        raise ValueError(f"profile_index {profile_index} not found in {indices_file}")

    row = idx_ds.sel(profile=profile_index)
    idx_start = int(row.start_idx.values)
    idx_end = int(row.end_idx.values)

    ds = load_profile_nc(raw_file)
    profile_ds = extract_profile_by_indices(ds, idx_start, idx_end)
    profile_ds.attrs["profile_index"] = profile_index
    profile_ds.attrs["profile_start_idx"] = idx_start
    profile_ds.attrs["profile_end_idx"] = idx_end
    return profile_ds
