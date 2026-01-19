"""
Merge multiple NetCDF files along t_fast and t_slow dimensions.

This module provides functionality to concatenate NetCDF files produced by
the p2nc command into a single continuous dataset with unified POSIX timestamps.
"""

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

import netCDF4 as nc

logger = logging.getLogger(__name__)

__all__ = ["merge_netcdf"]


def _parse_time_units(units_str: str) -> float:
    """
    Parse time units string and return POSIX offset.

    Parameters
    ----------
    units_str : str
        Units string like "seconds since 2025-02-03T04:22:13.432"

    Returns
    -------
    float
        POSIX timestamp of the reference time (UTC)
    """
    pattern = r"seconds\s+since\s+(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    match = re.search(pattern, units_str, re.IGNORECASE)

    if not match:
        raise ValueError(f"Cannot parse time units: {units_str}")

    datetime_str = match.group(1).replace(" ", "T")

    # Parse with or without fractional seconds
    if "." in datetime_str:
        dt = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")
    else:
        dt = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")

    # All times are UTC - explicitly set timezone to avoid local timezone conversion
    dt = dt.replace(tzinfo=timezone.utc)

    return dt.timestamp()


def _get_file_info(filepath: Path) -> tuple[dict, dict, dict, dict]:
    """Extract dimension and variable information from a NetCDF file."""
    with nc.Dataset(filepath, "r") as ds:
        # Check required dimensions exist
        for required_dim in ["t_fast", "t_slow"]:
            if required_dim not in ds.dimensions:
                raise ValueError(
                    f"Required dimension '{required_dim}' not found in {filepath}"
                )

        dims: dict[str, int | None] = {}
        concat_dims = {"t_fast", "t_slow"}
        for name, dim in ds.dimensions.items():
            dims[name] = None if name in concat_dims else len(dim)

        var_info = {}
        for name, var in ds.variables.items():
            var_info[name] = {
                "dimensions": var.dimensions,
                "dtype": var.dtype,
                "attrs": {attr: var.getncattr(attr) for attr in var.ncattrs()},
            }

        global_attrs = {attr: ds.getncattr(attr) for attr in ds.ncattrs()}

        sizes = {
            "t_fast": ds.dimensions["t_fast"].size,
            "t_slow": ds.dimensions["t_slow"].size,
        }

    return dims, var_info, global_attrs, sizes


def merge_netcdf(
    files: Union[list[Path], list[str]],
    output_file: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """
    Merge multiple NetCDF files along t_fast and t_slow dimensions.

    Concatenates files produced by the p2nc command into a single dataset,
    converting all timestamps to POSIX time (seconds since 1970-01-01).

    Parameters
    ----------
    files : list of Path or str
        List of input NetCDF file paths to merge. Will be sorted by filename.
    output_file : str or Path
        Path for the output merged file.
    verbose : bool, optional
        Print progress information. Default False.
    overwrite : bool, optional
        Overwrite output file if it exists. Default False.

    Returns
    -------
    Path
        Path to the output file.

    Raises
    ------
    ValueError
        If no input files provided or files lack required dimensions.
    FileExistsError
        If output file exists and overwrite is False.

    Examples
    --------
    >>> from pyturb.merge import merge_netcdf
    >>> merge_netcdf(['file1.nc', 'file2.nc'], 'combined.nc')
    """
    # Convert to Path objects and sort
    file_list = sorted([Path(f) for f in files])
    output_file = Path(output_file)

    if not file_list:
        raise ValueError("No input files provided")

    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output file '{output_file}' already exists. Use overwrite=True to replace."
        )

    logger.info(f"Merging {len(file_list)} files")
    logger.info(f"Output: {output_file}")
    logger.debug("-" * 50)

    # Get structure from first file
    dims, var_info, global_attrs, _ = _get_file_info(file_list[0])

    # Collect sizes and time offsets from all files
    file_info: list[dict[str, Any]] = []
    for f in file_list:
        with nc.Dataset(f, "r") as ds:
            info = {
                "path": f,
                "t_fast_size": ds.dimensions["t_fast"].size,
                "t_slow_size": ds.dimensions["t_slow"].size,
            }

            # Get time offsets
            for time_var in ["t_fast", "t_slow"]:
                if time_var in ds.variables:
                    units = ds.variables[time_var].units
                    info[f"{time_var}_offset"] = _parse_time_units(units)
                else:
                    raise ValueError(f"Time variable '{time_var}' not found in {f}")

            file_info.append(info)

    total_t_fast = sum(f["t_fast_size"] for f in file_info)
    total_t_slow = sum(f["t_slow_size"] for f in file_info)

    logger.info(f"Total t_fast: {total_t_fast}")
    logger.info(f"Total t_slow: {total_t_slow}")
    for info in file_info:
        logger.debug(
            f"  {info['path'].name}: "
            f"t_fast={info['t_fast_size']}, t_slow={info['t_slow_size']}"
        )
    logger.debug("-" * 50)

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create output file
    with nc.Dataset(output_file, "w", format="NETCDF4") as dst:
        # Create dimensions
        for name, size in dims.items():
            if name == "t_fast":
                dst.createDimension(name, total_t_fast)
            elif name == "t_slow":
                dst.createDimension(name, total_t_slow)
            else:
                dst.createDimension(name, size)

        # Create variables
        for name, info in var_info.items():
            var_dims = info["dimensions"]

            # Calculate chunk sizes
            chunksizes = None
            if len(var_dims) > 0:
                chunksizes = []
                for d in var_dims:
                    if d == "t_fast":
                        chunksizes.append(min(10000, total_t_fast))
                    elif d == "t_slow":
                        chunksizes.append(min(1000, total_t_slow))
                    elif dims[d] is not None:
                        chunksizes.append(min(5000, dims[d]))
                    else:
                        chunksizes.append(1000)

            var = dst.createVariable(
                name,
                info["dtype"],
                var_dims,
                chunksizes=chunksizes if chunksizes else None,
            )

            # Copy variable attributes, updating time units
            attrs = info["attrs"].copy()
            if name in ["t_fast", "t_slow"]:
                attrs["units"] = "seconds since 1970-01-01T00:00:00"
            var.setncatts(attrs)

        # Set global attributes
        dst.setncatts(global_attrs)
        dst.history = (
            f"Merged from {len(file_list)} files on {datetime.now().isoformat()}"
        )

        # Copy data file by file
        offset_t_fast = 0
        offset_t_slow = 0

        for idx, info in enumerate(file_info):
            f = info["path"]
            t_fast_size = info["t_fast_size"]
            t_slow_size = info["t_slow_size"]

            progress = (idx + 1) / len(file_list) * 100
            logger.info(f"[{progress:5.1f}%] Processing {f.name}...")

            with nc.Dataset(f, "r") as src:
                for name in src.variables:
                    var = src.variables[name]
                    var_dims = var.dimensions

                    # Handle time variables - convert to POSIX
                    if name == "t_fast":
                        data = var[:] + info["t_fast_offset"]
                        dst.variables[name][
                            offset_t_fast : offset_t_fast + t_fast_size
                        ] = data
                        continue

                    if name == "t_slow":
                        data = var[:] + info["t_slow_offset"]
                        dst.variables[name][
                            offset_t_slow : offset_t_slow + t_slow_size
                        ] = data
                        continue

                    # Handle other variables based on their dimensions
                    has_t_fast = "t_fast" in var_dims
                    has_t_slow = "t_slow" in var_dims

                    if has_t_fast or has_t_slow:
                        # Build slice for output array
                        slices = []
                        for d in var_dims:
                            if d == "t_fast":
                                slices.append(
                                    slice(offset_t_fast, offset_t_fast + t_fast_size)
                                )
                            elif d == "t_slow":
                                slices.append(
                                    slice(offset_t_slow, offset_t_slow + t_slow_size)
                                )
                            else:
                                slices.append(slice(None))

                        dst.variables[name][tuple(slices)] = var[:]

                    elif idx == 0:
                        # Only copy non-time-dependent variables once
                        dst.variables[name][:] = var[:]

            offset_t_fast += t_fast_size
            offset_t_slow += t_slow_size

    output_size = os.path.getsize(output_file) / (1024 * 1024)
    logger.debug("-" * 50)
    logger.info(f"Done! Output file size: {output_size:.2f} MB")

    return output_file
