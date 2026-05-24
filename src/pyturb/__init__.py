"""
PyTurb: Python tools for processing oceanic turbulence microstructure data.

This package provides tools for reading and processing data from RSI
microstructure instruments (P-files) and analyzing turbulence quantities.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyturb")
except PackageNotFoundError:
    __version__ = "unknown"

from . import auxiliary, io, merge, pfile, processing, profile, shear, viscosity

__all__ = [
    "__version__",
    "auxiliary",
    "io",
    "merge",
    "pfile",
    "processing",
    "profile",
    "shear",
    "viscosity",
]
