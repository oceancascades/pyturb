"""Command line interface for pyturb."""

import logging
from pathlib import Path
from typing import Callable, Literal, Optional

import typer
from typing_extensions import Annotated

from . import __version__
from .merge import merge_netcdf
from .pfile import batch_convert_to_netcdf, extract_pfile_segment
from .processing import batch_compute_epsilon, bin_profiles
from .profile import ProfileConfig
from .profile_index import batch_index_profiles

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
        typer.echo(f"pyturb version {__version__}")
        raise typer.Exit()


def _parse_input_list(
    spec: Optional[str],
    flag_name: str,
    fields: dict[str, Callable[[str], object]],
) -> Optional[dict]:
    """Parse a comma-separated positional value list into a dict.

    ``fields`` maps each output key to a 1-arg type constructor (e.g. ``int``,
    ``float``) that converts the corresponding comma-separated token. The
    order of ``fields`` dictates the expected token order.

    Returns ``None`` when ``spec`` is omitted so callers can keep defaults.
    Raises ``typer.BadParameter`` for wrong arity or unparseable values.
    """
    if not spec:
        return None
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != len(fields):
        raise typer.BadParameter(
            f"--{flag_name} needs {len(fields)} comma-separated values "
            f"({','.join(fields)}); got {len(parts)}"
        )
    try:
        return {key: cast(val) for (key, cast), val in zip(fields.items(), parts)}
    except ValueError as e:
        raise typer.BadParameter(f"Could not parse --{flag_name} values: {e}")


_DESPIKE_FIELDS: dict[str, Callable[[str], object]] = {
    "passes": int,
    "thresh": float,
    "smooth": float,
    "replace_sec": float,
}

_PEAKS_FIELDS: dict[str, Callable[[str], object]] = {
    "height": float,
    "distance": int,
    "width": int,
    "prominence": float,
}

_QC_THRESH_FIELDS: dict[str, Callable[[str], object]] = {
    "questionable": float,
    "bad": float,
}


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
    despike: Annotated[
        Optional[str],
        typer.Option(
            "--despike",
            help=(
                "Despike shear (sh1, sh2) and gradT (gradT1, gradT2) signals. "
                "Adds <probe>_clean and <probe>_despike_mask variables to the NetCDF. "
                "Specify using 4 comma-separated values: "
                "passes,thresh,smooth,replace_sec. "
                "passes = max iterations (1=fast, 10=thorough). "
                "thresh = spike-detection ratio of HP to LP envelope. "
                "smooth = envelope low-pass cutoff (Hz). "
                "replace_sec = replacement window around each spike (s). "
                "Defaults: 6,8.0,0.5,0.04. Example: --despike 10,7,0.5,0.05"
            ),
        ),
    ] = None,
    input_files: Annotated[
        list[Path] | None,
        typer.Argument(help="Input P-files (supports shell globs)"),
    ] = None,
):
    """Convert P-files to NetCDF format.

    Examples:
        pyturb p2nc ./data/*.p -o ./output
        pyturb p2nc file1.p file2.p file3.p
        pyturb p2nc ./data/*.p -o ./out --despike 6,8,0.5,0.04
    """
    if not input_files:
        typer.echo("Error: No input files specified.", err=True)
        raise typer.Exit(1)

    despike_opts = _parse_input_list(despike, "despike", _DESPIKE_FIELDS)

    batch_convert_to_netcdf(
        files=input_files,
        output_dir=output_dir,
        compress=compress,
        compression_level=compression_level,
        n_workers=n_workers,
        min_file_size=min_file_size,
        overwrite=overwrite,
        despike_kwargs=despike_opts,
    )


@app.command()
def cutp(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output p-file path",
        ),
    ],
    start: Annotated[
        int,
        typer.Option(
            "--start",
            "-s",
            help="First data record to copy (0-based, after config record)",
            show_default=True,
        ),
    ] = 0,
    n_records: Annotated[
        int,
        typer.Option(
            "--n-records",
            "-n",
            help="Number of data records to copy (~1 per second)",
            show_default=True,
        ),
    ] = 60,
    input_file: Annotated[
        Path,
        typer.Argument(help="Input p-file"),
    ] = None,
):
    """Extract a segment from a p-file.

    Copies the header/config record verbatim, then copies N contiguous data
    records.  The output is a valid p-file that can be processed normally.

    Each record is approximately 1 second of data (~60 records ≈ 1 minute).

    Examples:
        pyturb cutp deployment.p -o segment.p --start 300 --n-records 60
        pyturb cutp deployment.p -o segment.p -s 300 -n 120
    """
    if input_file is None:
        typer.echo("Error: No input file specified.", err=True)
        raise typer.Exit(1)

    try:
        result = extract_pfile_segment(
            input_file=input_file,
            output_file=output,
            start_record=start,
            n_records=n_records,
        )
        typer.echo(f"Wrote {n_records} records to {result}")
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


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
            "--min-speed",
            "-s",
            help="Speed below which a window's epsilon is QC-flagged questionable (m/s)",
            show_default=True,
        ),
    ] = 0.2,
    pressure_smoothing_period: Annotated[
        float,
        typer.Option(
            "--pressure-smoothing",
            help="Low-pass filter cutoff period for pressure (s)",
            show_default=True,
        ),
    ] = 0.5,
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
        str | None,
        typer.Option(
            "--aux-temp",
            help="Auxiliary temperature variable name (optional). If omitted, auxiliary temperature will NOT be applied.",
            show_default=True,
        ),
    ] = None,
    aux_sal: Annotated[
        str | None,
        typer.Option(
            "--aux-sal",
            help="Auxiliary salinity variable name (optional). If omitted, auxiliary salinity will NOT be applied.",
            show_default=True,
        ),
    ] = None,
    aux_dens: Annotated[
        str | None,
        typer.Option(
            "--aux-dens",
            help="Auxiliary density variable name (optional). If omitted, auxiliary density will NOT be applied.",
            show_default=True,
        ),
    ] = None,
    profile_direction: Annotated[
        Literal["down", "up", "both"],
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
    peaks: Annotated[
        Optional[str],
        typer.Option(
            "--peaks",
            help=(
                "Peak-detection parameters as 4 comma-separated values: "
                "height,distance,width,prominence. "
                "height = min peak height (dbar). "
                "distance = min samples between peaks. "
                "width = min peak width (samples). "
                "prominence = min peak prominence (dbar). "
                "Defaults: 25,200,200,25. Example: --peaks 50,300,300,25"
            ),
        ),
    ] = None,
    despike: Annotated[
        Optional[str],
        typer.Option(
            "--despike",
            help=(
                "Despike parameters as 4 comma-separated values: "
                "passes,thresh,smooth,replace_sec. "
                "passes = max iterations (1=fast, 10=thorough). "
                "thresh = spike-detection ratio of HP to LP envelope. "
                "smooth = envelope low-pass cutoff (Hz). "
                "replace_sec = replacement window around each spike (s). "
                "Defaults: 6,8.0,0.5,0.04."
                ""
                "Note that despike may be applied at the p2nc conversion, in which case the "
                "eps command will not despike unless the parameters are have changed."
            ),
        ),
    ] = None,
    accel_clean: Annotated[
        bool,
        typer.Option(
            "--accel-clean/--no-accel-clean",
            help="Apply Goodman coherent-noise removal using accelerometers",
            show_default=True,
        ),
    ] = False,
    emc_clean: Annotated[
        bool,
        typer.Option(
            "--emc-clean/--no-emc-clean",
            help="Apply Goodman coherent-noise removal using EM current meter driving current",
            show_default=True,
        ),
    ] = True,
    thermo: Annotated[
        bool,
        typer.Option(
            "--thermo/--no-thermo",
            help=(
                "Compute Conservative Temperature, Absolute Salinity, and "
                "potential density (0 dbar) when temperature and salinity are "
                "available. Uses lat/lon from --aux if provided, otherwise a "
                "default position (45N, 0E)."
            ),
            show_default=True,
        ),
    ] = False,
    chi: Annotated[
        bool,
        typer.Option(
            "--chi/--no-chi",
            help=(
                "Compute the temperature variance dissipation rate (chi) from "
                "the microstructure temperature gradient probes, using the "
                "combined shear-probe epsilon."
            ),
            show_default=True,
        ),
    ] = True,
    match_conductivity: Annotated[
        bool,
        typer.Option(
            "--match-conductivity/--no-match-conductivity",
            help="Lag and low-pass match JAC_C to JAC_T before window-averaging",
            show_default=True,
        ),
    ] = True,
    ctd_bin_sec: Annotated[
        float,
        typer.Option(
            "--ctd-bin-sec",
            help=(
                "Bin width in seconds for high-resolution CTD output "
                "(pressure, temperature, salinity, conductivity, density) "
                "on a separate ctd_time axis. Set to 0 to disable."
            ),
            show_default=True,
        ),
    ] = 0.25,
    stationary_platform: Annotated[
        Optional[bool],
        typer.Option(
            "--stationary-platform/--moving-platform",
            help=(
                "Use one lat/lon per profile instead of interpolating a "
                "position onto every window/bin. Default: auto-detect from "
                "the p-file's vehicle field (vmp/rvmp/xmp are stationary)."
            ),
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
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite/--no-overwrite",
            "-w/-W",
            help="Overwrite existing files",
            show_default=True,
        ),
    ] = False,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing/--no-skip-existing",
            help=(
                "Skip a file entirely if any output already exists for its "
                "stem, without detecting profiles first. Faster than the "
                "default per-profile check. Ignored with --overwrite."
            ),
            show_default=True,
        ),
    ] = False,
    input_files: Annotated[
        list[Path] | None,
        typer.Argument(help="Input NetCDF files (supports shell globs)"),
    ] = None,
):
    """Compute the dissipation rate of turbulent kinetic energy.

    Detects multiple profiles within each input file.
    Output files are named {input_stem}_p{NNN}.nc for each profile.

    Examples:
        pyturb eps ./converted/*.nc -o ./eps_output/
        pyturb eps ./converted/*.nc --direction both
        pyturb eps ./converted/*.nc --direction up --peaks-height 50
    """
    if not input_files:
        typer.echo("Error: No input files specified.", err=True)
        raise typer.Exit(1)

    despike_opts = _parse_input_list(despike, "despike", _DESPIKE_FIELDS)
    cfg_kwargs: dict = dict(
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
        aux_latitude=aux_lat,
        aux_longitude=aux_lon,
        aux_temperature=aux_temp,
        aux_salinity=aux_sal,
        aux_density=aux_dens,
        accel_clean=accel_clean,
        emc_clean=emc_clean,
        compute_thermo=thermo,
        compute_chi=chi,
        match_conductivity=match_conductivity,
        ctd_bin_sec=ctd_bin_sec,
        stationary_platform=stationary_platform,
    )
    # Only override the despike defaults — and force re-despike — when the
    # user explicitly passed --despike. Otherwise embedded <probe>_clean
    # vars from p2nc flow through untouched.
    if despike_opts is not None:
        cfg_kwargs["despike_max_passes"] = despike_opts["passes"]
        cfg_kwargs["despike_thresh"] = despike_opts["thresh"]
        cfg_kwargs["despike_smooth"] = despike_opts["smooth"]
        cfg_kwargs["despike_replace_sec"] = despike_opts["replace_sec"]
        cfg_kwargs["force_despike"] = True
    peaks_kwargs = _parse_input_list(peaks, "peaks", _PEAKS_FIELDS)
    if peaks_kwargs is not None:
        cfg_kwargs["peaks_kwargs"] = peaks_kwargs
    config = ProfileConfig(**cfg_kwargs)

    batch_compute_epsilon(
        files=input_files,
        config=config,
        output_dir=output_dir,
        auxiliary_file=auxiliary_file,
        n_workers=n_workers,
        overwrite=overwrite,
        skip_existing=skip_existing,
    )


@app.command()
def profiles(
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for profile index (and hires) files",
            show_default="current directory",
        ),
    ] = None,
    direction: Annotated[
        Literal["down", "up", "both"],
        typer.Option(
            "--direction",
            help="Profile direction to detect: down, up, or both",
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
    peaks: Annotated[
        Optional[str],
        typer.Option(
            "--peaks",
            help=(
                "Peak-detection parameters as 4 comma-separated values: "
                "height,distance,width,prominence. "
                "height = min peak height (dbar). "
                "distance = min samples between peaks. "
                "width = min peak width (samples). "
                "prominence = min peak prominence (dbar). "
                "Defaults: 25,200,200,25. Example: --peaks 50,300,300,25"
            ),
        ),
    ] = None,
    pressure_smoothing_period: Annotated[
        float,
        typer.Option(
            "--pressure-smoothing",
            help="Low-pass filter cutoff period for pressure (s)",
            show_default=True,
        ),
    ] = 0.5,
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
    materialize: Annotated[
        bool,
        typer.Option(
            "--materialize/--no-materialize",
            help=(
                "Also write each detected profile's full-resolution data "
                "(all fast and slow channel variables) to its own file, "
                "{stem}_p{NNNN}_hires.nc, in the same pass."
            ),
            show_default=True,
        ),
    ] = False,
    compress: Annotated[
        bool,
        typer.Option(
            help="Compress materialized profile NetCDF output (--materialize only)",
            show_default=True,
        ),
    ] = False,
    compression_level: Annotated[
        int,
        typer.Option(
            help="Compression level (1-9) for materialized output", show_default=True
        ),
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
        list[Path] | None,
        typer.Argument(help="Input NetCDF files (supports shell globs)"),
    ] = None,
):
    """Detect profiles in converted NetCDF files and write a boundary index.

    Output files are named {stem}_profiles.nc and record
    each profile's start/end indices, times, and direction, plus the
    detection config used.

    Use --materialize to also write each detected profile's full-resolution
    data to its own file ({stem}_p{NNNN}_hires.nc), or extract a single
    profile later, on demand, from a saved index with
    pyturb.profile_index.extract_profile().

    Examples:
        pyturb profiles ./converted/*.nc -o ./profiles/
        pyturb profiles ./converted/*.nc --direction both --materialize
    """
    if not input_files:
        typer.echo("Error: No input files specified.", err=True)
        raise typer.Exit(1)

    peaks_kwargs = _parse_input_list(peaks, "peaks", _PEAKS_FIELDS)
    cfg_kwargs: dict = dict(
        pressure_smoothing_period=pressure_smoothing_period,
        speed=speed,
        angle_of_attack=angle_of_attack,
        use_pitch_correction=use_pitch_correction,
        profile_direction=direction,
        min_profile_pressure=min_profile_pressure,
    )
    if peaks_kwargs is not None:
        cfg_kwargs["peaks_kwargs"] = peaks_kwargs
    config = ProfileConfig(**cfg_kwargs)

    results = batch_index_profiles(
        files=input_files,
        config=config,
        output_dir=output_dir,
        n_workers=n_workers,
        overwrite=overwrite,
        materialize=materialize,
        compress=compress,
        compression_level=compression_level,
    )

    if not results:
        typer.echo("Error: No data was indexed.", err=True)
        raise typer.Exit(1)


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
    ctd_bin_width: Annotated[
        Optional[float],
        typer.Option(
            "--ctd-bin-width",
            help=(
                "Also bin higher-resolution CTD variables onto a finer grid of this width (m)."
                "Variables appear on a separate ctd_depth coordinate."
            ),
        ),
    ] = None,
    default_latitude: Annotated[
        float,
        typer.Option(
            "--lat",
            help="Default latitude for pressure-to-depth conversion if not in data",
            show_default=True,
        ),
    ] = 45.0,
    variables: Annotated[
        str | None,
        typer.Option(
            "--vars",
            "-v",
            help=(
                "Comma-separated list of variables to bin (default: "
                "eps_1,eps_2,chi_1,chi_2,W,temperature,salinity,density,z,"
                "absolute_salinity,conservative_temperature,potential_density,"
                "N2,nu,kappa_T,lat,lon)"
            ),
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
    qc_thresh: Annotated[
        Optional[str],
        typer.Option(
            "--qc-thresh",
            help=(
                "Epsilon rejection thresholds (W/kg) as two comma-separated "
                "values: questionable,bad. A qc=2 (questionable) window is "
                "dropped before binning when its eps exceeds the first value; "
                "a qc=4 (bad) window is dropped when its eps exceeds the "
                "second. Low-epsilon flagged values are usually noise-floor "
                "artifacts and are kept by default. Defaults: 1e-7,1e-9. "
                "Pass 'inf,inf' to disable masking. Example: "
                "--qc-thresh 5e-8,5e-10"
            ),
        ),
    ] = None,
    input_files: Annotated[
        list[Path] | None,
        typer.Argument(help="Input epsilon NetCDF files (supports shell globs)"),
    ] = None,
):
    """Bin epsilon profiles by depth and concatenate into a single file.

    Depth is calculated from pressure using gsw.

    Examples:
        pyturb bin ./eps_output/*.nc -o binned.nc
        pyturb bin ./eps_output/*.nc -b 5.0 --dmax 500
        pyturb bin ./eps_output/*.nc --qc-thresh 5e-10
    """
    if not input_files:
        typer.echo("Error: No input files specified.", err=True)
        raise typer.Exit(1)

    # Parse variables if provided
    var_list = None
    if variables is not None:
        var_list = [v.strip() for v in variables.split(",")]

    qc_opts = _parse_input_list(qc_thresh, "qc-thresh", _QC_THRESH_FIELDS) or {}

    result = bin_profiles(
        files=input_files,
        output_file=output_file,
        depth_min=depth_min,
        depth_max=depth_max,
        bin_width=bin_width,
        variables=var_list,
        default_latitude=default_latitude,
        n_workers=n_workers,
        questionable_thresh=qc_opts.get("questionable", 1e-7),
        bad_thresh=qc_opts.get("bad", 1e-9),
        ctd_bin_width=ctd_bin_width,
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
