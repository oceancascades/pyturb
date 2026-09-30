"""Global attributes for pyturb outputs: provenance, lineage and user-supplied attributes."""

import logging
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import yaml

from . import __version__

_log = logging.getLogger(__name__)

__all__ = ["load_global_attrs", "stamp_globals", "validate_global_attrs"]

CONVENTIONS = "CF-1.8"
INSTRUMENT_ATTRS = ("instrument_vehicle", "instrument_model", "instrument_sn")
USER_ATTRS_KEY = "pyturb_user_attrs"

_PROTECTED = {"Conventions", "history", "date_created", "n_fft", "n_diss"}
_PROTECTED_PREFIXES = (
    "pyturb_",
    "source_",
    "instrument_",
    "profile_",
    "fs_",
    "auxiliary_",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_history(attrs: dict, step: str, detail: str) -> str:
    """``attrs``' history with one line appended for this pyturb step."""
    line = f"{utc_now()} pyturb {__version__} {step}: {detail}"
    prior = str(attrs.get("history", "")).strip()
    return f"{prior}\n{line}" if prior else line


def _source_pfile(attrs: dict) -> Optional[str]:
    if "source_pfile" in attrs:
        return str(attrs["source_pfile"])
    # Files converted before source_pfile existed stored the p-file's full path.
    legacy = str(attrs.get("source_file", ""))
    return Path(legacy).name if legacy.endswith(".p") else None


def stamp_globals(
    source_attrs: dict,
    *,
    step: str,
    title: str,
    detail: str,
    user_attrs: Optional[dict] = None,
    **extra: Any,
) -> dict:
    """Build the full set of global attributes for a pyturb output.

    Carries instrument identity and ``source_pfile`` from ``source_attrs``,
    adds ``extra``, stamps version/date, appends a ``history`` line, and
    applies ``user_attrs`` last (recording their keys in ``pyturb_user_attrs``).
    Nothing else from ``source_attrs`` is carried.
    """
    attrs: dict[str, Any] = {"Conventions": CONVENTIONS, "title": title}
    attrs.update({k: source_attrs[k] for k in INSTRUMENT_ATTRS if k in source_attrs})
    if pfile := _source_pfile(source_attrs):
        attrs["source_pfile"] = pfile
    attrs.update(extra)
    attrs["pyturb_version"] = __version__
    attrs["date_created"] = utc_now()
    attrs["history"] = append_history(source_attrs, step, detail)
    if user_attrs:
        attrs.update(user_attrs)
        attrs[USER_ATTRS_KEY] = " ".join(user_attrs)
    return attrs


def validate_global_attrs(attrs: dict) -> dict:
    """Check user-supplied global attributes and return them.

    Values must be strings, numbers, or lists of numbers, and keys pyturb
    sets itself (``history``, ``instrument_*``, ``pyturb_*``, ...) are refused.
    Raises ValueError naming the offending key.
    """
    for key, value in attrs.items():
        if key in _PROTECTED or key.startswith(_PROTECTED_PREFIXES):
            raise ValueError(
                f"Global attribute '{key}' is set by pyturb and can't be overridden"
            )
        is_number = isinstance(value, Number) and not isinstance(value, bool)
        is_number_list = isinstance(value, list) and all(
            isinstance(v, Number) and not isinstance(v, bool) for v in value
        )
        if not (isinstance(value, str) or is_number or is_number_list):
            raise ValueError(
                f"Global attribute '{key}' must be a string, number, or list of "
                f"numbers, not {type(value).__name__} (quote dates and booleans)"
            )
    return attrs


def load_global_attrs(path: Union[str, Path]) -> dict:
    """Load user global attributes (title, institution, creator_name, ...) from YAML.

    The file is a flat mapping of attribute name to value; see
    :func:`validate_global_attrs` for what's allowed.
    """
    attrs = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(attrs, dict):
        raise ValueError(f"{path}: expected a mapping of attribute names to values")
    return validate_global_attrs(attrs)


def _same(a: Any, b: Any) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b))


def split_shared(attrs_list: list[dict], keys: list[str]) -> tuple[dict, list[str]]:
    """Split ``keys`` into those identical in every dict and those that aren't.

    Returns ``(shared, differing)``: a key that differs, or is missing from
    only some dicts, is in ``differing``; one missing from all is in neither.
    """
    shared, differing = {}, []
    for key in keys:
        values = [a.get(key) for a in attrs_list]
        if all(v is None for v in values):
            continue
        if all(v is not None and _same(v, values[0]) for v in values):
            shared[key] = values[0]
        else:
            differing.append(key)
    return shared, differing


def common_attrs(attrs_list: list[dict], keys: list[str], what: str) -> dict:
    """The ``keys`` whose value is identical in every dict of ``attrs_list``.

    A key that differs, or is missing from only some dicts, is omitted with a
    warning; one missing from all of them is omitted silently.
    """
    shared, differing = split_shared(attrs_list, keys)
    for key in differing:
        distinct = list(dict.fromkeys(str(a.get(key))[:60] for a in attrs_list))
        _log.warning(
            f"{what} '{key}' differs between profiles ({len(distinct)} distinct "
            f"values, e.g. {distinct[:3]}); omitted from the binned file"
        )
    return shared
