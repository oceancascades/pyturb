"""Batch processing functions for microstructure data."""

import multiprocessing as mp
from pathlib import Path
from typing import Optional, Union

import xarray as xr

from .io import load_profile_nc
from .profile import (
    ProfileConfig,
    merge_auxiliary_data,
    prepare_profile,
    process_profile,
)

__all__ = [
    "batch_compute_epsilon",
]


def _process_single_profile(
    input_file: Path,
    output_dir: Path,
    config: ProfileConfig,
    overwrite: bool,
    aux_ds: Optional[xr.Dataset] = None,
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

        # Merge auxiliary data if provided (needs decoded times)
        if aux_ds is not None:
            ds_decoded = xr.decode_cf(ds)
            ds_decoded = merge_auxiliary_data(ds_decoded, aux_ds, config)
            # Copy auxiliary variables back to original dataset
            # Use ("t_slow", values) to avoid coordinate type mismatch
            # (ds uses float64 epoch, ds_decoded uses datetime64)
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

        result = process_profile(ds, config)

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

        result_filtered.to_netcdf(output_file)

        return (input_file, output_file, None)

    except Exception as e:
        return (input_file, None, str(e))


def _unpack_epsilon_args(args: tuple) -> tuple:
    """Unpack arguments for imap_unordered."""
    input_file, output_dir, config, overwrite, aux_ds = args
    return _process_single_profile(input_file, output_dir, config, overwrite, aux_ds)


def batch_compute_epsilon(
    files: Union[str, Path, list[Path]],
    output_dir: Optional[Union[str, Path]] = None,
    diss_len_sec: float = 4.0,
    fft_len_sec: float = 1.0,
    min_speed: float = 0.2,
    smoothing_period: float = 0.25,
    temperature: str = "JAC_T",
    speed: str = "W",
    angle_of_attack: float = 3.0,
    use_pitch_correction: bool = False,
    auxiliary_file: Optional[Union[str, Path]] = None,
    aux_latitude: str = "lat",
    aux_longitude: str = "lon",
    aux_temperature: str = "temperature",
    aux_salinity: str = "salinity",
    aux_density: str = "density",
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
    speed : str, optional
        Name of speed variable. If not found in dataset, speed is estimated
        from pressure derivative. Default 'W'.
    angle_of_attack : float, optional
        Angle of attack in degrees, used when estimating speed from pressure.
        Default 3.0.
    use_pitch_correction : bool, optional
        Whether to apply pitch correction when estimating speed from pressure.
        Default False.
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

    config = ProfileConfig(
        diss_len_sec=diss_len_sec,
        fft_len_sec=fft_len_sec,
        min_speed=min_speed,
        smoothing_period=smoothing_period,
        temperature=temperature,
        speed=speed,
        angle_of_attack=angle_of_attack,
        use_pitch_correction=use_pitch_correction,
        aux_latitude=aux_latitude,
        aux_longitude=aux_longitude,
        aux_temperature=aux_temperature,
        aux_salinity=aux_salinity,
        aux_density=aux_density,
        verbose=verbose,
    )

    # Load auxiliary dataset if provided
    aux_ds = None
    if auxiliary_file is not None:
        auxiliary_file = Path(auxiliary_file)
        if not auxiliary_file.exists():
            raise FileNotFoundError(f"Auxiliary file not found: {auxiliary_file}")
        aux_ds = xr.open_dataset(auxiliary_file)

        # Interpolate over NaN values in auxiliary variables (use configured names)
        aux_vars = [
            config.aux_latitude,
            config.aux_longitude,
            config.aux_temperature,
            config.aux_salinity,
            config.aux_density,
        ]
        for var in aux_vars:
            if var in aux_ds and aux_ds[var].isnull().any():
                # Get the time dimension name
                time_dim = aux_ds[var].dims[0] if aux_ds[var].dims else None
                if time_dim is not None:
                    aux_ds[var] = aux_ds[var].interpolate_na(
                        dim=time_dim, method="linear", fill_value="extrapolate"
                    )
                    if verbose:
                        print(f"Interpolated NaN values in auxiliary variable '{var}'")

        if verbose:
            print(f"Loaded auxiliary dataset from {auxiliary_file}")

    args = [(f, output_dir, config, overwrite, aux_ds) for f in nc_files]

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
