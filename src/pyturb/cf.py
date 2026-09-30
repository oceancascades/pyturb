"""CF variable attributes from the bundled registry (``assets/variables.yml``)."""

import re
from functools import cache
from importlib.resources import files
from typing import Optional

import numpy as np
import xarray as xr
import yaml

__all__ = ["apply_cf", "missing_cf", "normalize_units"]

_SUFFIXES = {"_hires": "", "_clean": " (despiked)", "_smooth": " (smoothed)"}
_ANCILLARY_SUFFIXES = ("_qc", "_n", "_fm")

_UNIT_ALIASES = {
    "C": "degree_C",
    "degC": "degree_C",
    "deg": "degree",
    "dBar": "dbar",
    "counts": "1",
    "µT": "uT",
    "μT": "uT",
}


@cache
def _registry() -> tuple[dict, list]:
    entries = yaml.safe_load(
        files("pyturb").joinpath("assets/variables.yml").read_text()
    )
    exact = {k: v for k, v in entries.items() if "{" not in k}
    patterns = [(_compile(k), v) for k, v in entries.items() if "{" in k]
    return exact, patterns


def _compile(key: str) -> re.Pattern:
    seen: set[str] = set()

    def group(m: re.Match) -> str:
        name = m.group(1)
        if name in seen:
            return f"(?P={name})"
        seen.add(name)
        return f"(?P<{name}>\\d+)" if name == "n" else f"(?P<{name}>\\w+?)"

    return re.compile(re.sub(r"\\\{(\w+)\\\}", group, re.escape(key)))


def lookup(name: str) -> Optional[dict]:
    """Registry attributes for variable ``name``, or None if it has no entry."""
    exact, patterns = _registry()
    if name in exact:
        return dict(exact[name])
    for pattern, attrs in patterns:
        if m := pattern.fullmatch(name):
            return {k: str(v).format(**m.groupdict()) for k, v in attrs.items()}
    for suffix, note in _SUFFIXES.items():
        if name.endswith(suffix) and (base := lookup(name[: -len(suffix)])):
            base["long_name"] += note
            return base
    return None


def normalize_units(units: str) -> str:
    """Map an ODAS/setup-file unit string (e.g. ``mS/cm``, ``m^2 s^-3``) to UDUNITS.

    Strings it doesn't recognize are returned with only brackets and ``^`` removed.
    """
    u = units.strip().removeprefix("[").removesuffix("]").strip().replace("^", "")
    num, *dens = (part.strip() for part in u.split("/"))
    parts = [] if num == "1" and dens else [_UNIT_ALIASES.get(num, num)]
    for den in dens:
        if not (m := re.fullmatch(r"([A-Za-z]+)(\d*)", den)):
            return u
        parts.append(f"{m[1]}-{m[2] or 1}")
    return " ".join(parts)


def apply_cf(ds: xr.Dataset) -> xr.Dataset:
    """Set registry ``long_name``/``units``/``standard_name`` on every variable.

    Variables are matched by exact name, then pattern, then by stripping a
    ``_hires``/``_clean``/``_smooth`` suffix. Other attributes (``comment``, ``cal_*``,
    provenance) are kept, ``_qc`` flag variables are left untouched, and
    unmatched variables keep whatever attributes they already have. Also sets
    ``ancillary_variables`` to any ``<var>_qc``/``_n``/``_fm`` companions.
    Modifies ``ds`` in place and returns it.
    """
    for name in list(ds.variables):
        if name.endswith("_qc"):
            continue
        var = ds.variables[name]
        if attrs := lookup(name):
            var.attrs.update(attrs)
        ancillary = [f"{name}{s}" for s in _ANCILLARY_SUFFIXES if f"{name}{s}" in ds]
        if ancillary:
            var.attrs["ancillary_variables"] = " ".join(ancillary)
    return ds


def missing_cf(ds: xr.Dataset) -> list[str]:
    """Names of variables lacking ``long_name``, or ``units`` where CF expects one.

    ``units`` isn't required on flag variables or non-numeric (string) ones.
    """
    missing = []
    for name in ds.variables:
        attrs = ds.variables[name].attrs
        needs_units = np.issubdtype(ds.variables[name].dtype, np.number) and (
            "flag_values" not in attrs
        )
        if "long_name" not in attrs or (needs_units and "units" not in attrs):
            missing.append(str(name))
    return missing
