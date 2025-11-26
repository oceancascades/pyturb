"""Command line interface for pyturb."""

from importlib.metadata import version
from pathlib import Path

import typer
from typing_extensions import Annotated

from .pfile import batch_convert_to_netcdf
from .processing import batch_compute_epsilon

app = typer.Typer()


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
):
    """pyturb: Tools for processing ocean microstructure data."""


@app.command()
def p2nc(
    input_dir: Annotated[
        Path, typer.Argument(help="Input directory containing P-files")
    ],
    output_dir: Annotated[
        Path, typer.Argument(help="Output directory for NetCDF files")
    ],
    pattern: Annotated[
        str, typer.Option(help="Glob pattern for P-files", show_default=True)
    ] = "*.p",
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
):
    """Convert P-files to NetCDF format."""
    input_path = input_dir / pattern

    batch_convert_to_netcdf(
        pattern=input_path,
        output_dir=output_dir,
        compress=compress,
        compression_level=compression_level,
        n_workers=n_workers,
        min_file_size=min_file_size,
        overwrite=overwrite,
        verbose=True,
    )


@app.command()
def epsilon(
    input_dir: Annotated[
        Path, typer.Argument(help="Input directory containing converted NetCDF files")
    ],
    output_dir: Annotated[
        Path, typer.Argument(help="Output directory for epsilon NetCDF files")
    ],
    pattern: Annotated[
        str, typer.Option(help="Glob pattern for NetCDF files", show_default=True)
    ] = "*.nc",
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
    smoothing_period: Annotated[
        float,
        typer.Option(
            "--smoothing",
            help="Low-pass filter cutoff period for speed/pressure (s)",
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
):
    """Compute epsilon TKE dissipation rate from converted NetCDF files.

    This command processes raw p2nc output by smoothing speed/pressure,
    scaling shear and gradT probes by fall speed, and computing epsilon
    using the Nasmyth spectrum fit.
    """
    input_path = input_dir / pattern

    batch_compute_epsilon(
        pattern=input_path,
        output_dir=output_dir,
        diss_len_sec=diss_len,
        fft_len_sec=fft_len,
        min_speed=min_speed,
        smoothing_period=smoothing_period,
        temperature=temperature,
        n_workers=n_workers,
        overwrite=overwrite,
        verbose=True,
    )
