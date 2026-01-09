"""
Read RSI P-file (ODAS binary format) data files.

This module provides functions to read binary data files from RSI microstructure
instruments in the ODAS format. It is a Python translation of the MATLAB code
from read_pfile.m and open_pfile.m.

Author: Translated from MATLAB ODAS library v4.5.1
"""

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Optional, Union

from ._pfile import (
    SetupConfig,
    adis_extract,
    convert_all_channels,
    convert_channel,
    deconvolve,
    make_gradT,
    open_pfile,
    read_pfile,
    to_xarray,
)
from ._pfile.preprocess import preprocess_shear

logger = logging.getLogger(__name__)

# Re-export for public API
__all__ = [
    "SetupConfig",
    "read_pfile",
    "open_pfile",
    "load_pfile",
    "load_pfile_phys",
    "save_netcdf",
    "batch_convert_to_netcdf",
    "to_xarray",
    "convert_channel",
    "convert_all_channels",
    "deconvolve",
    "adis_extract",
    "make_gradT",
]


def load_pfile(filename: Union[str, Path]) -> Dict:
    """
    Convenience function to read a P-file and return only the data dictionary.

    Parameters
    ----------
    filename : str or Path
        Path to the P-file

    Returns
    -------
    dict
        Dictionary containing channel data and metadata

    Examples
    --------
    >>> data = load_pfile('my_file.p')
    >>> shear = data['sh1']
    >>> pressure = data['P']
    """
    data = read_pfile(filename)
    return data


def load_pfile_phys(
    filename: Union[str, Path],
    exclude_types: Optional[list] = None,
    verbose: bool = False,
) -> Dict:
    """
    Read a P-file and convert all channels to physical units.

    This is a convenience function that combines read_pfile() with
    convert_all_channels() to provide data in calibrated physical units.

    Parameters
    ----------
    filename : str or Path
        Path to the P-file
    exclude_types : list, optional
        List of channel types to skip conversion (default: ['gnd', 'raw'])
    verbose : bool, optional
        If True, print the address matrix and other diagnostic info. Default False.

    Returns
    -------
    dict
        Dictionary containing:
        - Converted channel data in physical units
        - 'units' : dict - Units for each converted channel
        - 'fs_fast' : float - Fast sampling rate (Hz)
        - 'fs_slow' : float - Slow sampling rate (Hz)
        - 't_fast' : ndarray - Time vector for fast channels
        - 't_slow' : ndarray - Time vector for slow channels
        - 'filetime' : datetime - File timestamp
        - 'date' : str - Date string
        - 'time' : str - Time string
        - 'header_version' : float - Header version
        - 'fullPath' : str - Full path to file
        - 'cfgobj' : SetupConfig - Configuration object
        - 'gradT1', 'gradT2', etc. : Temperature time derivatives (dT/dt in K/s)

    Examples
    --------
    >>> from pyturb import pfile
    >>> data = pfile.load_pfile_phys('myfile.p')
    >>> shear = data['sh1']  # Shear in s^-1
    >>> temp = data['T1']    # Temperature in °C
    >>> gradT = data['gradT1']  # Temperature time derivative in K/s
    >>> print(f"Shear units: {data['units']['sh1']}")
    >>> print(f"Temp units: {data['units']['T1']}")

    Notes
    -----
    This function automatically converts all channels to physical units
    using the calibration coefficients in the configuration file.
    Channels of type 'gnd' and 'raw' are excluded by default.

    Temperature time derivatives (gradT1, gradT2, etc.) are computed
    automatically for all pre-emphasized thermistor channels.
    """

    data = read_pfile(filename, verbose=verbose)

    data_phys = convert_all_channels(data, exclude_types=exclude_types)

    return data_phys


def save_netcdf(
    data: Dict,
    output_file: Union[str, Path],
    variables: Optional[list] = None,
    compress: bool = False,
    compression_level: int = 4,
    overwrite: bool = False,
) -> None:
    """
    Save P-file data to a CF-compliant NetCDF file.

    Parameters
    ----------
    data : dict
        Data dictionary from load_pfile_phys() containing channel data,
        metadata, and the configuration object.
    output_file : str or Path
        Path to the output NetCDF file (.nc extension recommended)
    variables : list, optional
        List of variable names to save. If None, saves default variables:
        P, sh1, sh2, gradT1, gradT2, U_EM, JAC_T, JAC_C, T1, T2, plus
        accelerometers and inclinometers if present.
    compress : bool, optional
        Whether to compress the data. Default False.
    compression_level : int, optional
        Compression level (1-9). Default 4.
    overwrite : bool, optional
        Whether to overwrite existing files. Default False.

    Notes
    -----
    The output file follows CF-1.8 conventions where applicable:
    - Time coordinates use the 'seconds since' convention
    - Standard names are used for recognized variables
    - The P-file configuration string is stored as a global attribute

    Data is saved as float32 to reduce file size while maintaining
    sufficient precision for microstructure data.

    Examples
    --------
    >>> from pyturb import pfile
    >>> data = pfile.load_pfile_phys('myfile.p')
    >>> pfile.save_netcdf(data, 'myfile.nc')

    >>> # Save only specific variables
    >>> pfile.save_netcdf(data, 'myfile.nc', variables=['P', 'sh1', 'sh2'])
    """
    output_file = Path(output_file)

    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {output_file}. Set overwrite=True to replace."
        )

    ds = to_xarray(data, variables=variables)

    encoding = {}
    if compress:
        for var in ds.data_vars:
            encoding[var] = {
                "zlib": True,
                "complevel": compression_level,
                "dtype": "float32",
            }
        for coord in ["t_fast", "t_slow"]:
            encoding[coord] = {
                "zlib": True,
                "complevel": compression_level,
                "dtype": "float64",
            }

    ds.to_netcdf(output_file, encoding=encoding, format="NETCDF4")


def _process_single_file(
    pfile_path: Path,
    output_dir: Path,
    variables: Optional[list],
    compress: bool,
    compression_level: int,
    exclude_types: Optional[list],
    overwrite: bool,
    preprocess: bool = False,
    despike_passes: int = 6,
    hp_cutoff_hz: float = 0.5,
) -> tuple:
    """Worker function for parallel processing. Returns (input_path, output_path, error)."""
    try:
        output_file = output_dir / (pfile_path.stem + ".nc")
        data = load_pfile_phys(pfile_path, exclude_types=exclude_types)

        # Log data dimensions
        n_fast = len(data.get("t_fast", []))
        n_slow = len(data.get("t_slow", []))
        logger.info(f"Loaded {pfile_path.name}: t_fast={n_fast}, t_slow={n_slow}")

        # Apply shear preprocessing if requested
        if preprocess:
            data = preprocess_shear(
                data,
                despike_passes=despike_passes,
                hp_cutoff_hz=hp_cutoff_hz,
            )

        save_netcdf(
            data,
            output_file,
            variables=variables,
            compress=compress,
            compression_level=compression_level,
            overwrite=overwrite,
        )
        return (pfile_path, output_file, None)
    except Exception as e:
        return (pfile_path, None, str(e))


# Global variable for worker log level (set by initializer)
_worker_log_level = logging.INFO


def _init_worker(log_level: int):
    """Initialize logging in worker processes."""
    global _worker_log_level
    _worker_log_level = log_level
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        force=True,
    )


def _unpack_worker_args(args: tuple) -> tuple:
    """Unpack arguments for imap_unordered (needed because lambdas can't be pickled)."""
    (
        pfile_path,
        output_dir,
        variables,
        compress,
        compression_level,
        exclude_types,
        overwrite,
        preprocess,
        despike_passes,
        hp_cutoff_hz,
    ) = args
    return _process_single_file(
        pfile_path,
        output_dir,
        variables,
        compress,
        compression_level,
        exclude_types,
        overwrite,
        preprocess,
        despike_passes,
        hp_cutoff_hz,
    )


def batch_convert_to_netcdf(
    files: Union[str, Path, list[Path]],
    output_dir: Optional[Union[str, Path]] = None,
    variables: Optional[list] = None,
    compress: bool = False,
    compression_level: int = 4,
    exclude_types: Optional[list] = None,
    n_workers: Optional[int] = None,
    overwrite: bool = False,
    min_file_size: int = 100_000,
    preprocess: bool = False,
    despike_passes: int = 6,
    hp_cutoff_hz: float = 0.5,
) -> None:
    """
    Batch convert multiple P-files to NetCDF using parallel processing.

    Parameters
    ----------
    files : str, Path, or list of Path
        Either a glob pattern to match P-files (e.g., '/path/to/data/*.p'),
        a directory path (in which case '*.p' is appended), or a list of
        Path objects pointing to specific files.
    output_dir : str or Path, optional
        Directory for output NetCDF files. If None, uses current directory.
    variables : list, optional
        Variable names to save. If None, saves default variables.
    compress : bool, optional
        Whether to compress the NetCDF files. Default False.
    compression_level : int, optional
        Compression level (1-9). Default 4.
    exclude_types : list, optional
        Channel types to skip during conversion.
    n_workers : int, optional
        Number of parallel workers. Default is number of CPU cores.
    overwrite : bool, optional
        Whether to overwrite existing files. Default False.
    min_file_size : int, optional
        Minimum file size in bytes. Files smaller than this are skipped.
        Default 100000 (100 kB). Set to 0 to process all files.
    preprocess : bool, optional
        Whether to preprocess shear probes (despike + HP filter). Default False.
        When enabled, creates sh1_hp and sh2_hp variables that can be used
        directly by epsilon processing, skipping redundant filtering.
    despike_passes : int, optional
        Maximum despike iterations for preprocessing. Default 6.
    hp_cutoff_hz : float, optional
        High-pass filter cutoff frequency in Hz for preprocessing. Default 0.5.

    Examples
    --------
    >>> from pyturb import pfile
    >>> # Convert all P-files in a directory
    >>> results = pfile.batch_convert_to_netcdf('/path/to/data/*.p')

    >>> # Convert with compression and specific output directory
    >>> results = pfile.batch_convert_to_netcdf(
    ...     '/path/to/data/**/*.p',
    ...     output_dir='/path/to/output',
    ...     compress=True,
    ...     n_workers=4
    ... )

    >>> # Using list of files with preprocessing
    >>> results = pfile.batch_convert_to_netcdf(
    ...     [Path('file1.p'), Path('file2.p')],
    ...     preprocess=True,
    ...     despike_passes=6,
    ... )
    """
    # Handle different input types
    if isinstance(files, list):
        # Already a list of files
        pfiles = sorted(files)
    else:
        # It's a pattern or directory
        pattern = Path(files)

        if pattern.is_dir():
            pattern = pattern / "*.p"

        if pattern.is_absolute():
            pfiles = sorted(pattern.parent.glob(pattern.name))
        else:
            pfiles = sorted(Path.cwd().glob(str(pattern)))

    if not pfiles:
        raise RuntimeError("No p files found.")

    # Filter out small files
    if min_file_size > 0:
        original_count = len(pfiles)
        pfiles = [pf for pf in pfiles if pf.stat().st_size >= min_file_size]
        skipped_small = original_count - len(pfiles)
        if skipped_small:
            logger.info(
                f"Skipping {skipped_small} files smaller than {min_file_size / 1000:.0f} kB"
            )

    if not pfiles:
        logger.info("No files to process after size filtering")
        return

    logger.info(f"Found {len(pfiles)} P-files to convert")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()

    # Filter out files that already have output if not overwriting
    if not overwrite:
        files_to_process = []
        skipped = []
        for pf in pfiles:
            out_file = output_dir / (pf.stem + ".nc")
            if out_file.exists():
                skipped.append(pf)
            else:
                files_to_process.append(pf)

        if skipped:
            logger.info(f"Skipping {len(skipped)} files (output already exists)")

        pfiles = files_to_process

        if not pfiles:
            logger.info("No files to process (all outputs already exist)")
            return

    if n_workers is None:
        n_workers = mp.cpu_count()

    args = [
        (
            pf,
            output_dir,
            variables,
            compress,
            compression_level,
            exclude_types,
            overwrite,
            preprocess,
            despike_passes,
            hp_cutoff_hz,
        )
        for pf in pfiles
    ]

    # Always use parallel processing
    results = []
    log_level = logger.getEffectiveLevel()
    with mp.Pool(
        processes=n_workers, initializer=_init_worker, initargs=(log_level,)
    ) as pool:
        # Use imap_unordered for streaming results as they complete
        results_iter = pool.imap_unordered(_unpack_worker_args, args)
        for i, (input_path, output_path, error) in enumerate(results_iter):
            success = error is None
            results.append(
                {
                    "input": input_path,
                    "output": output_path,
                    "success": success,
                    "error": error,
                }
            )
            status = "successfully converted" if success else "failed to convert"
            logger.info(f"[{i + 1}/{len(pfiles)}] {status} {input_path.name}")
            if error:
                logger.error(f"    Error: {error}")

    # Summary
    n_success = sum(1 for r in results if r["success"])
    n_failed = len(results) - n_success
    logger.info(f"Completed: {n_success} succeeded, {n_failed} failed")
