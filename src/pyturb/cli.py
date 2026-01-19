"""Command line interface for pyturb."""

import logging
from importlib.metadata import version
from pathlib import Path

import typer
from typing_extensions import Annotated

from .merge import merge_netcdf
from .pfile import batch_convert_to_netcdf
from .processing import batch_compute_epsilon, bin_profiles

app = typer.Typer()

# Map string log levels to logging constants
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def _setup_logging(level: str) -> None:
    """Configure logging for the CLI."""
    log_level = LOG_LEVELS.get(level.lower(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        force=True,  # Override any existing configuration
    )


def version_callback(value: bool):
    if value:
        typer.echo(f"pyturb version {version('pyturb')}")
        raise typer.Exit()


def cli():
    app()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-v", callback=version_callback, help="Show version and exit."
        ),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            "-l",
            help="Logging level (debug, info, warning, error)",
            show_default=True,
        ),
    ] = "info",
):
    """pyturb: Tools for processing ocean microstructure data."""
    _setup_logging(log_level)


@app.command()
def p2nc(
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for NetCDF files",
            show_default="current directory",
        ),
    ] = None,
    compress: Annotated[
        bool, typer.Option(help="Compress NetCDF output", show_default=True)
    ] = False,
    compression_level: Annotated[
        int, typer.Option(help="Compression level (1-9)", show_default=True)
    ] = 4,
    n_workers: Annotated[
        int | None,
        typer.Option(
            "--n-workers",
            "-n",
            help="Number of parallel workers",
            show_default="all CPUs",
        ),
    ] = None,
    min_file_size: Annotated[
        int, typer.Option(help="Minimum file size in bytes", show_default=True)
    ] = 100_000,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite/--no-overwrite",
            "-w/-W",
            help="Overwrite existing files",
            show_default=True,
        ),
    ] = False,
    input_files: Annotated[
        list[Path],
        typer.Argument(help="Input P-files (supports shell globs)"),
    ] = None,
):
    """Convert P-files to NetCDF format.

    Examples:
        pyturb p2nc ./data/*.p -o ./output
        pyturb p2nc file1.p file2.p file3.p
    """
    if not input_files:
        typer.echo("Error: No input files specified.", err=True)
        raise typer.Exit(1)

    batch_convert_to_netcdf(
        files=input_files,
        output_dir=output_dir,
        compress=compress,
        compression_level=compression_level,
        n_workers=n_workers,
        min_file_size=min_file_size,
        overwrite=overwrite,
    )


@app.command()
def eps(
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for epsilon NetCDF files",
            show_default="current directory",
        ),
    ] = None,
    diss_len: Annotated[
        float,
        typer.Option(
            "--diss-len",
            "-d",
            help="Dissipation window length in seconds",
            show_default=True,
        ),
    ] = 4.0,
    fft_len: Annotated[
        float,
        typer.Option(
            "--fft-len", "-f", help="FFT window length in seconds", show_default=True
        ),
    ] = 1.0,
    min_speed: Annotated[
        float,
        typer.Option(
            "--min-speed", "-s", help="Minimum speed threshold (m/s)", show_default=True
        ),
    ] = 0.2,
    pressure_smoothing_period: Annotated[
        float,
        typer.Option(
            "--pressure-smoothing",
            help="Low-pass filter cutoff period for pressure (s)",
            show_default=True,
        ),
    ] = 0.25,
    temperature: Annotated[
        str,
        typer.Option(
            "--temperature",
            "-t",
            help="Temperature variable name for viscosity",
            show_default=True,
        ),
    ] = "JAC_T",
    speed: Annotated[
        str,
        typer.Option(
            "--speed",
            help="Speed variable name. If not found, estimates from pressure.",
            show_default=True,
        ),
    ] = "W",
    angle_of_attack: Annotated[
        float,
        typer.Option(
            "--aoa",
            help="Angle of attack in degrees (used when estimating speed from pressure)",
            show_default=True,
        ),
    ] = 3.0,
    use_pitch_correction: Annotated[
        bool,
        typer.Option(
            "--pitch-correction/--no-pitch-correction",
            help="Apply pitch correction when estimating speed from pressure",
            show_default=True,
        ),
    ] = False,
    auxiliary_file: Annotated[
        Path | None,
        typer.Option(
            "--aux",
            "-a",
            help="Auxiliary NetCDF file with lat, lon, T, S, density time series",
        ),
    ] = None,
    aux_lat: Annotated[
        str,
        typer.Option(
            "--aux-lat",
            help="Latitude variable name in auxiliary file",
            show_default=True,
        ),
    ] = "lat",
    aux_lon: Annotated[
        str,
        typer.Option(
            "--aux-lon",
            help="Longitude variable name in auxiliary file",
            show_default=True,
        ),
    ] = "lon",
    aux_temp: Annotated[
        str,
        typer.Option(
            "--aux-temp",
            help="Temperature variable name in auxiliary file",
            show_default=True,
        ),
    ] = "temperature",
    aux_sal: Annotated[
        str,
        typer.Option(
            "--aux-sal",
            help="Salinity variable name in auxiliary file",
            show_default=True,
        ),
    ] = "salinity",
    aux_dens: Annotated[
        str,
        typer.Option(
            "--aux-dens",
            help="Density variable name in auxiliary file",
            show_default=True,
        ),
    ] = "density",
    profile_direction: Annotated[
        str,
        typer.Option(
            "--direction",
            help="Profile direction to process: down, up, or both",
            show_default=True,
        ),
    ] = "down",
    min_profile_pressure: Annotated[
        float,
        typer.Option(
            "--min-profile-pressure",
            help="Minimum pressure (dbar) for profile detection",
            show_default=True,
        ),
    ] = 0.0,
    peaks_height: Annotated[
        float,
        typer.Option(
            "--peaks-height",
            help="Minimum peak height for profile detection (dbar)",
            show_default=True,
        ),
    ] = 25.0,
    peaks_distance: Annotated[
        int,
        typer.Option(
            "--peaks-distance",
            help="Minimum samples between peaks for profile detection",
            show_default=True,
        ),
    ] = 200,
    peaks_prominence: Annotated[
        float,
        typer.Option(
            "--peaks-prominence",
            help="Minimum peak prominence for profile detection (dbar)",
            show_default=True,
        ),
    ] = 25.0,
    despike_passes: Annotated[
        int,
        typer.Option(
            "--despike-passes",
            help="Max despike iterations (1 = fast, 10 = thorough)",
            show_default=True,
        ),
    ] = 6,
    n_workers: Annotated[
        int | None,
        typer.Option(
            "--n-workers",
            "-n",
            help="Number of parallel workers",
            show_default="all CPUs",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite/--no-overwrite",
            "-w/-W",
            help="Overwrite existing files",
            show_default=True,
        ),
    ] = False,
    input_files: Annotated[
        list[Path],
        typer.Argument(help="Input NetCDF files (supports shell globs)"),
    ] = None,
):
    """Compute epsilon TKE dissipation rate from converted NetCDF files.

    Automatically detects multiple profiles (dive cycles) within each file.
    Output files are named {input_stem}_p{NNN}.nc for each profile.

    Examples:
        pyturb eps ./converted/*.nc -o ./eps_output/
        pyturb eps ./converted/*.nc --direction both
        pyturb eps ./converted/*.nc --direction up --peaks-height 50
    """
    if not input_files:
        typer.echo("Error: No input files specified.", err=True)
        raise typer.Exit(1)

    # Build peaks_kwargs from individual options
    peaks_kwargs = {
        "height": peaks_height,
        "distance": peaks_distance,
        "width": peaks_distance,  # Use same as distance
        "prominence": peaks_prominence,
    }

    batch_compute_epsilon(
        files=input_files,
        output_dir=output_dir,
        diss_len_sec=diss_len,
        fft_len_sec=fft_len,
        min_speed=min_speed,
        pressure_smoothing_period=pressure_smoothing_period,
        temperature=temperature,
        speed=speed,
        angle_of_attack=angle_of_attack,
        use_pitch_correction=use_pitch_correction,
        profile_direction=profile_direction,
        min_profile_pressure=min_profile_pressure,
        peaks_kwargs=peaks_kwargs,
        auxiliary_file=auxiliary_file,
        aux_latitude=aux_lat,
        aux_longitude=aux_lon,
        aux_temperature=aux_temp,
        aux_salinity=aux_sal,
        aux_density=aux_dens,
        despike_max_passes=despike_passes,
        n_workers=n_workers,
        overwrite=overwrite,
        verbose=True,
    )


@app.command()
def bin(
    output_file: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output NetCDF file for binned data",
            show_default=True,
        ),
    ] = Path("binned_profiles.nc"),
    depth_min: Annotated[
        float,
        typer.Option(
            "--dmin",
            help="Minimum depth for binning (m)",
            show_default=True,
        ),
    ] = 0.0,
    depth_max: Annotated[
        float,
        typer.Option(
            "--dmax",
            help="Maximum depth for binning (m)",
            show_default=True,
        ),
    ] = 1000.0,
    bin_width: Annotated[
        float,
        typer.Option(
            "--bin-width",
            "-b",
            help="Depth bin width (m)",
            show_default=True,
        ),
    ] = 2.0,
    default_latitude: Annotated[
        float,
        typer.Option(
            "--lat",
            help="Default latitude for pressure-to-depth conversion if not in data",
            show_default=True,
        ),
    ] = 45.0,
    bin_by_pressure: Annotated[
        bool,
        typer.Option(
            "--pressure",
            "-p",
            help="Bin by pressure (dbar) instead of depth (m)",
            show_default=True,
        ),
    ] = False,
    variables: Annotated[
        str | None,
        typer.Option(
            "--vars",
            "-v",
            help="Comma-separated list of variables to bin (default: eps_1,eps_2,W,temperature,salinity,density,nu,lat,lon)",
        ),
    ] = None,
    n_workers: Annotated[
        int | None,
        typer.Option(
            "--n-workers",
            "-n",
            help="Number of parallel workers",
            show_default="all CPUs",
        ),
    ] = None,
    input_files: Annotated[
        list[Path],
        typer.Argument(help="Input epsilon NetCDF files (supports shell globs)"),
    ] = None,
):
    """Bin epsilon profiles by depth and concatenate into a single file.

    By default, bins by depth calculated from pressure using gsw.
    Use --pressure to bin by pressure instead.

    Examples:
        pyturb bin ./eps_output/*.nc -o binned.nc
        pyturb bin ./eps_output/*.nc -b 5.0 --dmax 500
        pyturb bin ./eps_output/*.nc --pressure --dmin 0 --dmax 500
    """
    if not input_files:
        typer.echo("Error: No input files specified.", err=True)
        raise typer.Exit(1)

    # Parse variables if provided
    var_list = None
    if variables is not None:
        var_list = [v.strip() for v in variables.split(",")]

    result = bin_profiles(
        files=input_files,
        output_file=output_file,
        depth_min=depth_min,
        depth_max=depth_max,
        bin_width=bin_width,
        variables=var_list,
        default_latitude=default_latitude,
        bin_by_pressure=bin_by_pressure,
        n_workers=n_workers,
        verbose=True,
    )

    if result is None:
        typer.echo("Error: No data was binned.", err=True)
        raise typer.Exit(1)


@app.command()
def merge(
    output_file: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output filename for merged NetCDF file",
        ),
    ],
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite/--no-overwrite",
            "-w/-W",
            help="Overwrite output file if it exists",
            show_default=True,
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show files that would be merged without merging",
            show_default=True,
        ),
    ] = False,
    input_files: Annotated[
        list[Path],
        typer.Argument(help="Input NetCDF files to merge (supports shell globs)"),
    ] = None,
):
    """Merge multiple p2nc NetCDF files into a single file.

    Concatenates files along t_fast and t_slow dimensions, converting
    timestamps to POSIX time (seconds since 1970-01-01).

    Examples:
        pyturb merge ./converted/*.nc -o combined.nc
        pyturb merge file1.nc file2.nc file3.nc -o merged.nc
        pyturb merge ./converted/*.nc -o combined.nc --dry-run
    """
    if not input_files:
        typer.echo("Error: No input files specified.", err=True)
        raise typer.Exit(1)

    # Sort files by name
    file_list = sorted(input_files)

    if dry_run:
        typer.echo(f"Would merge {len(file_list)} files into '{output_file}':")
        for f in file_list:
            if f.exists():
                size = f.stat().st_size / (1024 * 1024)
                typer.echo(f"  {f} ({size:.2f} MB)")
            else:
                typer.echo(f"  {f} (not found)")
        raise typer.Exit(0)

    try:
        merge_netcdf(
            files=file_list,
            output_file=output_file,
            verbose=True,
            overwrite=overwrite,
        )
    except FileExistsError as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo("Use -w/--overwrite to replace existing file.", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Successfully merged {len(file_list)} files into '{output_file}'")
