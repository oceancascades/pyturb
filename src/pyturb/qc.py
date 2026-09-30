"""Quality-control flags (IODE convention: 1 good, 2 not evaluated, 3 questionable,
4 bad, 9 missing) for per-window estimates and their depth-binned averages."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import xarray as xr
import yaml

if TYPE_CHECKING:
    from .profile import ProfileConfig

FLAG_VALUES = np.array([1, 2, 3, 4, 9], dtype="i1")
FLAG_MEANINGS = "good not_evaluated questionable bad missing"
MISSING = np.int8(9)
EXCLUDED = np.int8(-1)
AGREEMENT_FACTOR = 10.0

# Platform noise floors (W/kg): a ship-tethered VMP is far noisier than a
# MicroRider on a glider or mooring.
VMP_EPS_FLOOR = 1e-10
DEFAULT_EPS_FLOOR = 1e-12

OVERRIDE_PROBES = ("sh1", "sh2", "T1", "T2")

# Broad, globally-safe bounds on seawater temperature. A T1/T2 value outside
# this range usually indicates a calibration extrapolated beyond its fitted
# range (e.g. applied to an anomalous profile) rather than real data.
# calibrate-fp07's apply_probe_calibration applies the best available fit
# as-is and does not mask this -- flagged via QC here instead, at the eps
# step, per policy: don't destroy data, flag it and let the
# consumer decide.
MIN_VALID_TEMP_C = -3.0
MAX_VALID_TEMP_C = 40.0


def flag_attrs(long_name: str, comment: str) -> dict:
    """CF attributes for a QC flag variable."""
    return {
        "long_name": long_name,
        "flag_values": FLAG_VALUES,
        "flag_meanings": FLAG_MEANINGS,
        "valid_min": np.int8(1),
        "valid_max": np.int8(9),
        "comment": comment,
    }


def temperature_range_mask(T: np.ndarray) -> np.ndarray:
    """True where T is missing or outside the valid seawater temperature range."""
    return ~np.isfinite(T) | (T < MIN_VALID_TEMP_C) | (T > MAX_VALID_TEMP_C)


def compose_range_qc(
    range_frac: np.ndarray,
    config: "ProfileConfig",
    fit_confident: Optional[bool] = None,
) -> np.ndarray:
    """QC flag from the fraction of a temperature probe's raw samples that
    fell outside the valid range within a window, set to 4 (bad) if the
    calibration fit did not meet the fit quality criteria.

      * fit_confident is False                 -> 4 (bad), regardless of
        range_frac -- a fit that isn't confident (see fp07_calibration.
        fit_is_confident) can produce values that look physically plausible
        while still being substantially wrong; no per-sample range check
        can catch that, so it's flagged from the fit's own quality instead.
        fit_confident is None (never run through calibrate-fp07, or an
        older file predating this attr) applies no such floor.
      * range_frac > despike_frac_bad          -> 4 (bad)
      * range_frac > despike_frac_questionable -> 3 (questionable)
      * range_frac is NaN (no raw samples)     -> 9 (missing)
      * otherwise                              -> 1 (good)
    """
    qc = np.ones(range_frac.shape, dtype="i1")
    qc[range_frac > config.despike_frac_questionable] = 3
    qc[range_frac > config.despike_frac_bad] = 4
    qc[np.isnan(range_frac)] = 9
    if fit_confident is False:
        qc[qc != 9] = 4
    return qc


def compose_qc(
    eps: np.ndarray,
    fm: np.ndarray,
    speed_bad: np.ndarray,
    despike_frac: np.ndarray,
    config: "ProfileConfig",
) -> np.ndarray:
    """Combine speed, FM, and despike-fraction contributions into a per-window flag.

    Precedence (max wins across the three contributions, then NaN-eps overrides as 9):
      * FM <= fm_good                                           -> 1 (good)
      * FM NaN                                                  -> 4 (bad)
      * fm_good < FM <= fm_bad                                  -> 3 (questionable)
      * FM > fm_bad                                             -> 4 (bad)
      * speed below min_speed                                   -> 3 (questionable)
      * despike_frac > despike_frac_questionable                -> 3 (questionable)
      * despike_frac > despike_frac_bad                         -> 4 (bad)
      * eps NaN                                                 -> 9 (missing, overrides)
    """
    qc_speed = np.ones(eps.size, dtype="i1")
    qc_speed[speed_bad] = 3

    qc_fm = np.full(eps.size, 4, dtype="i1")
    qc_fm[fm <= config.fm_good] = 1
    qc_fm[(fm > config.fm_good) & (fm <= config.fm_bad)] = 3

    # Despike fraction: 0 (or absent → zeros from caller) leaves qc_dsp at 1.
    qc_dsp = np.ones(eps.size, dtype="i1")
    qc_dsp[despike_frac > config.despike_frac_questionable] = 3
    qc_dsp[despike_frac > config.despike_frac_bad] = 4

    qc = np.maximum(np.maximum(qc_speed, qc_fm), qc_dsp)
    qc[np.isnan(eps)] = 9
    return qc


def combine_probe_pair(
    eps1: np.ndarray, eps2: np.ndarray, qc1: np.ndarray, qc2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Combine two probe estimates into (eps, eps_qc).

    Only estimates that are finite and flagged good/questionable (qc <= 3)
    are used.

    eps:
      * both used & within factor of ``AGREEMENT_FACTOR`` → mean
      * both used, disagreement larger → element-wise minimum
      * exactly one used → that value
      * neither used → NaN

    eps_qc:
      * both used and agree → max(qc1, qc2)
      * both used but disagree, or only one used → 3 (questionable)
      * neither used → 9 if both are missing, else 4 (bad)
    """
    e1_ok = np.isfinite(eps1) & (qc1 <= 3)
    e2_ok = np.isfinite(eps2) & (qc2 <= 3)
    both = e1_ok & e2_ok

    hi = np.fmax(eps1, eps2)
    lo = np.fmin(eps1, eps2)
    within = both & (hi <= AGREEMENT_FACTOR * lo)

    eps = np.where(
        within,
        0.5 * (eps1 + eps2),
        np.where(both, lo, np.where(e1_ok, eps1, np.where(e2_ok, eps2, np.nan))),
    )

    both_missing = (qc1 == MISSING) & (qc2 == MISSING)
    eps_qc = np.where(both_missing, MISSING, 4).astype("i1")
    eps_qc[e1_ok | e2_ok] = 3
    eps_qc[within] = np.maximum(qc1[within], qc2[within])
    return eps.astype("f4"), eps_qc


def attach_combined(ds: xr.Dataset, name: str, long_name: str, units: str) -> bool:
    """Attach per-window ``name``/``name_qc`` combined from ``name_1``/``name_2``.

    A missing probe is treated as all-missing. Returns False if neither exists.
    """
    if f"{name}_1" not in ds and f"{name}_2" not in ds:
        return False
    n = ds.sizes["time"]
    missing = (np.full(n, np.nan), np.full(n, MISSING, dtype="i1"))
    (v1, q1), (v2, q2) = (
        (ds[v].values, ds[f"{v}_qc"].values.astype("i1")) if v in ds else missing
        for v in (f"{name}_1", f"{name}_2")
    )
    val, qc = combine_probe_pair(v1, v2, q1, q2)
    ds[name] = ("time", val)
    ds[name].attrs = {
        "long_name": long_name,
        "units": units,
        "comment": (
            f"Mean of {name}_1 and {name}_2 where both are flagged good or "
            f"questionable and agree within a factor of "
            f"{AGREEMENT_FACTOR:g}; the minimum where they differ by more; "
            "the value from the one probe flagged good or questionable "
            "otherwise."
        ),
    }
    ds[f"{name}_qc"] = ("time", qc)
    ds[f"{name}_qc"].attrs = flag_attrs(
        f"QC flag for {name}",
        f"Max of {name}_1_qc and {name}_2_qc where both probes are used and "
        f"agree within a factor of {AGREEMENT_FACTOR:g}; 3 (questionable) "
        "where they differ by more or only one probe is used; 9 (missing) "
        "if both are missing, else 4 (bad).",
    )
    return True


def mask_low_quality_eps(
    ds: xr.Dataset,
    names: tuple[str, ...],
    questionable_thresh: float,
    bad_thresh: float,
) -> xr.Dataset:
    """NaN out epsilon (and sentinel its QC) using separate questionable / bad
    rejection thresholds.

    A QC-flagged questionable (qc=3) window is excluded when its epsilon
    exceeds ``questionable_thresh``; a QC-flagged bad (qc=4) window is
    excluded when its epsilon exceeds ``bad_thresh``. Below the respective
    threshold the value is kept (low-epsilon flagged windows are usually
    noise-floor artifacts rather than instrument problems). The QC sentinel
    ``EXCLUDED`` marks "excluded from binning"; it is mapped back to 9
    (missing) after binning.
    """
    for eps_name in names:
        qc_name = f"{eps_name}_qc"
        if eps_name not in ds or qc_name not in ds:
            continue
        eps = ds[eps_name].values.copy()
        qc = ds[qc_name].values.astype("i1", copy=True)
        excluded = ((qc == 3) & (eps > questionable_thresh)) | (
            (qc == 4) & (eps > bad_thresh)
        )
        if not excluded.any():
            continue
        eps[excluded] = np.nan
        qc[excluded] = EXCLUDED
        ds[eps_name] = (ds[eps_name].dims, eps)
        ds[qc_name] = (ds[qc_name].dims, qc)
    return ds


def select_best_windows(
    ds: xr.Dataset, value_vars: list[str], codes: np.ndarray, n_bins: int
) -> xr.Dataset:
    """Per bin, keep only windows flagged good/questionable if any exist,
    else fall back to the remaining (bad) windows. Dropped windows get NaN
    and the QC sentinel; ``<var>_n`` marks the windows kept. A bin with no
    finite value keeps flag 4 if any of its windows were flagged bad.
    """
    in_bin = codes >= 0
    bin_idx = np.where(in_bin, codes, 0)
    for v in value_vars:
        val = ds[v].values.astype("f8", copy=True)
        qc = ds[f"{v}_qc"].values.astype("i1", copy=True)
        finite = np.isfinite(val)
        usable = finite & (qc >= 1) & (qc <= 3)
        has_usable = np.bincount(codes[usable & in_bin], minlength=n_bins) > 0
        used = finite & in_bin & (usable | ~has_usable[bin_idx])
        has_used = np.bincount(codes[used], minlength=n_bins) > 0
        bad_flag = in_bin & ~has_used[bin_idx] & (qc == 4)
        val[~used] = np.nan
        qc[~(used | bad_flag)] = EXCLUDED
        ds[v] = (ds[v].dims, val)
        ds[f"{v}_qc"] = (ds[v].dims, qc)
        ds[f"{v}_n"] = (ds[v].dims, used.astype("i4"))
    return ds


def restore_qc_missing(ds: xr.Dataset, qc_vars: list[str]) -> xr.Dataset:
    """Map sentinel ``EXCLUDED`` and empty-bin NaN values in QC vars back to 9.

    After groupby_bins.max(), QC vars may contain:
      * EXCLUDED: all contributing windows were excluded → missing
      * NaN: the bin had no contributing windows at all → missing
      * one of {1, 2, 3, 4, 9}: a real flag from at least one window
    """
    for v in qc_vars:
        if v not in ds:
            continue
        arr = ds[v].values
        out = np.where(np.isnan(arr) | (arr == EXCLUDED), MISSING, arr)
        ds[v] = (ds[v].dims, out.astype("i1"))
    return ds


def load_overrides(path: Union[str, Path]) -> list[dict]:
    """Load manual QC override rules from a YAML file.

    The file holds a list of rules, each with:

      * ``instrument_sn`` (required): matched against the ``instrument_sn`` attr.
      * ``probes`` (required): any of ``sh1``, ``sh2``, ``T1``, ``T2``. A shear
        probe flags ``eps_N``; a thermistor flags ``TN`` and ``chi_N``.
      * ``start`` / ``end`` (optional, ISO 8601 UTC): inclusive time range;
        omit either for an open-ended range.
      * ``flag`` (optional, 3 or 4; default 4).
      * ``reason`` (required): recorded in the flagged variables' comments.

    Overrides only make flags worse. Returns the rules with times as strings.
    """
    rules = yaml.safe_load(Path(path).read_text()) or []
    if not isinstance(rules, list):
        raise ValueError(f"{path}: expected a list of override rules")
    out = []
    for i, r in enumerate(rules):
        missing = {"instrument_sn", "probes", "reason"} - set(r)
        if missing:
            raise ValueError(f"{path}: rule {i} is missing {sorted(missing)}")
        probes = [r["probes"]] if isinstance(r["probes"], str) else list(r["probes"])
        unknown = set(probes) - set(OVERRIDE_PROBES)
        if unknown:
            raise ValueError(f"{path}: rule {i} has unknown probes {sorted(unknown)}")
        flag = int(r.get("flag", 4))
        if flag not in (3, 4):
            raise ValueError(f"{path}: rule {i} flag must be 3 or 4, got {flag}")
        rule = {
            "instrument_sn": str(r["instrument_sn"]).strip(),
            "probes": probes,
            "flag": flag,
            "reason": str(r["reason"]),
        }
        for key in ("start", "end"):
            if r.get(key) is not None:
                rule[key] = str(np.datetime64(r[key], "s"))
        out.append(rule)
    return out


def apply_overrides(ds: xr.Dataset, rules: list[dict], qc_vars: list[str]) -> None:
    """Raise ``qc_vars`` (per-window, on ``time``) to each matching rule's flag, in place."""
    sn = str(ds.attrs.get("instrument_sn", "")).strip()
    rules = [r for r in rules if r["instrument_sn"] == sn]
    if not rules:
        return
    times = xr.decode_cf(ds[["time"]])["time"].values.astype("datetime64[s]")
    for v in qc_vars:
        if v not in ds:
            continue
        n = re.search(r"\d", v).group()
        probe = f"sh{n}" if v.startswith("eps_") else f"T{n}"
        for r in rules:
            if probe not in r["probes"]:
                continue
            hit = np.ones(times.shape, bool)
            if "start" in r:
                hit &= times >= np.datetime64(r["start"])
            if "end" in r:
                hit &= times <= np.datetime64(r["end"])
            if not hit.any():
                continue
            flags = ds[v].values
            flags[hit] = np.maximum(flags[hit], r["flag"])
            ds[v].attrs["comment"] = (
                ds[v].attrs.get("comment", "")
                + f" Manually flagged {r['flag']}: {r['reason']}"
            ).strip()
