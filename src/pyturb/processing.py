"""Batch processing functions for microstructure data."""

import multiprocessing as mp
from pathlib import Path
from typing import Optional, Union

from .io import load_profile_nc
from .profile import PrepareConfig, ProfileConfig, prepare_profile, process_profile

__all__ = [
    "batch_compute_epsilon",
]


def _process_single_profile(
    input_file: Path,
    output_dir: Path,
    prepare_config: PrepareConfig,
    profile_config: ProfileConfig,
    overwrite: bool,
) -> tuple:
    """
    Process a single profile file and save results.

    Returns (input_path, output_path, error).
    """
    try:
        output_file = output_dir / input_file.name

        if output_file.exists() and not overwrite:
            return (input_file, output_file, "skipped (exists)")

        ds = load_profile_nc(input_file)

        # Prepare the profile (smooth speed/pressure, scale shear/gradT)
        ds = prepare_profile(ds, prepare_config)

        result = process_profile(ds, profile_config)

        # Keep only variables on t_diss dimension (epsilon results)
        vars_to_keep = [
            v
            for v in result.data_vars
            if "t_diss" in result[v].dims and len(result[v].dims) > 0
        ]
        result_filtered = result[vars_to_keep]

        # Add coordinates
        result_filtered = result_filtered.assign_coords(
            frequency=result.frequency,
            k=result.k,
        )

        result_filtered.to_netcdf(output_file)

        return (input_file, output_file, None)

    except Exception as e:
        return (input_file, None, str(e))


def _unpack_epsilon_args(args: tuple) -> tuple:
    """Unpack arguments for imap_unordered."""
    input_file, output_dir, prepare_config, profile_config, overwrite = args
    return _process_single_profile(
        input_file, output_dir, prepare_config, profile_config, overwrite
    )


def batch_compute_epsilon(
    files: Union[str, Path, list[Path]],
    output_dir: Optional[Union[str, Path]] = None,
    diss_len_sec: float = 4.0,
    fft_len_sec: float = 1.0,
    min_speed: float = 0.2,
    smoothing_period: float = 0.25,
    temperature: str = "JAC_T",
    n_workers: Optional[int] = None,
    verbose: bool = False,
    overwrite: bool = False,
) -> list[dict]:
    """
    Batch compute epsilon from converted NetCDF files.

    This function processes raw p2nc output by:
    1. Smoothing speed and pressure data
    2. Scaling shear probes by 1/U^2 and gradT probes by 1/U
    3. Computing epsilon using the Nasmyth spectrum fit

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
    smoothing_period : float, optional
        Low-pass filter cutoff period for speed/pressure smoothing. Default 0.25 s.
    temperature : str, optional
        Name of temperature variable for viscosity calculation. Default 'JAC_T'.
    n_workers : int, optional
        Number of parallel workers. Default is number of CPU cores.
    verbose : bool, optional
        Print progress information. Default False.
    overwrite : bool, optional
        Whether to overwrite existing files. Default False.

    Returns
    -------
    list of dict
        Results for each file with keys: 'input', 'output', 'success', 'error'

    Examples
    --------
    >>> from pyturb.processing import batch_compute_epsilon
    >>> # Using glob pattern
    >>> results = batch_compute_epsilon('/path/to/data/*.nc', output_dir='/path/to/output')
    >>> # Using list of files
    >>> results = batch_compute_epsilon([Path('file1.nc'), Path('file2.nc')])
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
            print("No NetCDF files found.")
        return []

    if verbose:
        print(f"Found {len(nc_files)} NetCDF files to process")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()

    # Filter out files that already have output if not overwriting
    if not overwrite:
        files_to_process = []
        skipped = []
        for f in nc_files:
            out_file = output_dir / f.name
            if out_file.exists():
                skipped.append(f)
            else:
                files_to_process.append(f)

        if verbose and skipped:
            print(f"Skipping {len(skipped)} files (output already exists)")

        nc_files = files_to_process

        if not nc_files:
            if verbose:
                print("No files to process (all outputs already exist)")
            return []

    if n_workers is None:
        n_workers = mp.cpu_count()

    prepare_config = PrepareConfig(
        smoothing_period=smoothing_period,
    )

    profile_config = ProfileConfig(
        diss_len_sec=diss_len_sec,
        fft_len_sec=fft_len_sec,
        min_speed=min_speed,
        # Use smoothed variables from prepare_profile
        pressure="P_smooth",
        speed="U_EM_smooth",
        temperature=temperature,
        verbose=verbose,
    )

    args = [
        (f, output_dir, prepare_config, profile_config, overwrite) for f in nc_files
    ]

    results = []

    # Use serial processing for small batches
    if len(nc_files) <= min(n_workers, 4):
        if verbose:
            print("Using serial processing for small batch")
        for i, arg_tuple in enumerate(args):
            input_path, output_path, error = _unpack_epsilon_args(arg_tuple)
            success = error is None
            results.append(
                {
                    "input": input_path,
                    "output": output_path,
                    "success": success,
                    "error": error,
                }
            )
            if verbose:
                status = "successfully processed" if success else f"failed ({error})"
                print(f"[{i + 1}/{len(nc_files)}] {status}: {input_path.name}")
    else:
        if verbose:
            print(f"Using {n_workers} parallel workers")
        with mp.Pool(processes=n_workers) as pool:
            results_iter = pool.imap_unordered(_unpack_epsilon_args, args)
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
                if verbose:
                    status = (
                        "successfully processed" if success else f"failed ({error})"
                    )
                    print(f"[{i + 1}/{len(nc_files)}] {status}: {input_path.name}")

    # Summary
    if verbose:
        n_success = sum(1 for r in results if r["success"])
        n_failed = len(results) - n_success
        print(f"\nCompleted: {n_success} succeeded, {n_failed} failed")

    return results
