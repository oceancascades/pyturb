"""In-situ recalibration of FP07 thermistor probes against a reference thermometer.

Python port of ODAS's ``cal_FP07_in_situ.m``, adapted to work from p2nc-
converted data (no raw p-file patching). Requires p2nc's retained raw counts
(``<probe>_counts``, ``<probe>_d<probe>``) and per-variable ``cal_<key>``
calibration attrs (see ``_pfile/to_xarray.py``) -- reconvert with the current
p2nc if a file predates these.

Fitting regresses the resistance-ratio log_R (computed directly from raw
counts via the electronics constants, unaffected by any prior clipping)
against the reference. Applying rebuilds gradT1/gradT2 from scratch with the
new coefficients via ``deconvolve``/``make_gradT``, exact and lossless.
"""

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal as sig
import xarray as xr
import yaml
from numpy.typing import NDArray

from ._pfile import deconvolve, make_gradT
from .conductivity import _lag_filter, _matching_filter
from .profile import ProfileConfig

_log = logging.getLogger(__name__)

__all__ = [
    "ProbeCalibrationFit",
    "fit_probe_calibration",
    "apply_probe_calibration",
    "write_report",
    "read_report",
]


def _channel_params(ds: xr.Dataset, probe: str) -> dict:
    """A channel's calibration parameters, from the ``cal_<key>`` attrs p2nc
    attaches to each variable (see ``_pfile/to_xarray.py``)."""
    if probe in ds:
        cal_attrs = {
            k[len("cal_") :]: v
            for k, v in ds[probe].attrs.items()
            if k.startswith("cal_")
        }
        if cal_attrs:
            return cal_attrs
    raise ValueError(
        f"No 'cal_*' attrs found for channel '{probe}'; reconvert with the "
        "current p2nc to enable in-situ calibration."
    )


def _electronics(params: dict) -> tuple[float, float, float, float, float, int]:
    """(a, b, g, e_b, eta, adc_bits) from a channel's electronics parameters."""
    a = float(params["a"])
    b = float(params["b"])
    g = float(params["g"])
    e_b = float(params["e_b"])
    adc_fs = float(params["adc_fs"])
    adc_bits = int(params["adc_bits"])
    eta = (b / 2) * (2**adc_bits) * g * e_b / adc_fs
    return a, b, g, e_b, eta, adc_bits


def _coefficients(params: dict) -> tuple[float, float, Optional[float]]:
    """(T_0, beta_1, beta_2) from a channel's thermal calibration parameters.

    beta_2 is None when the channel is a first-order (beta_1-only) fit.
    """
    t_0 = float(params["t_0"])
    beta_1 = float(params["beta_1"]) if "beta_1" in params else float(params["beta"])
    beta_2 = float(params["beta_2"]) if "beta_2" in params else None
    return t_0, beta_1, beta_2


def log_r_from_counts(
    counts: NDArray,
    a: float,
    b: float,
    g: float,
    e_b: float,
    adc_fs: float,
    adc_bits: int,
) -> NDArray:
    """log(R_T/R_0) directly from raw ADC counts (electronics only).

    Matches ``_therm``'s conversion up to (but not including) its Z clip --
    unlike a value inverted from an already-converted temperature, this is
    never affected by that clip.
    """
    counts = np.asarray(counts, dtype=float)
    Z = ((counts - a) / b) * (adc_fs / 2**adc_bits) * 2 / (g * e_b)
    R = (1 - Z) / (1 + Z)
    return np.log(R)


def steinhart_hart_forward(
    log_R: NDArray, T_0: float, beta_1: float, beta_2: Optional[float] = None
) -> NDArray:
    """Absolute temperature (K) from log(R_T/R_0) via the Steinhart-Hart fit."""
    T_inv = 1 / T_0 + log_R / beta_1
    if beta_2 is not None:
        T_inv = T_inv + log_R**2 / beta_2
    return 1.0 / T_inv


def find_lag(
    T: NDArray,
    T_ref: NDArray,
    fs: float,
    max_lag_s: float = 10.0,
    smooth_hz: float = 4.0,
) -> tuple[float, float]:
    """Lag (s) of T relative to T_ref via cross-correlation, and the peak coefficient.

    Negative lag means T_ref trails behind T (the usual case: the reference
    sensor is physically behind the fast probe on the instrument frame).
    """
    b, a = sig.butter(2, smooth_hz / (fs / 2))
    x = sig.lfilter(b, a, sig.detrend(np.diff(T)))
    y = sig.lfilter(b, a, sig.detrend(np.diff(T_ref)))
    max_lag = int(round(max_lag_s * fs))
    correlation = sig.correlate(x, y, mode="full")
    norm = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if norm > 0:
        correlation = correlation / norm
    lags = sig.correlation_lags(len(x), len(y), mode="full")
    window = np.abs(lags) <= max_lag
    lags_w, corr_w = lags[window], correlation[window]
    idx = int(np.argmax(np.abs(corr_w)))
    return float(lags_w[idx]) / fs, float(corr_w[idx])


def _shift(x: NDArray, lag_samples: float) -> NDArray:
    """Shift x by a signed (possibly fractional) number of samples.

    Positive delays x; negative advances it (via _lag_filter on the
    reversed array, since _lag_filter only delays).
    """
    if lag_samples > 0:
        return _lag_filter(x, lag_samples)
    if lag_samples < 0:
        return _lag_filter(x[::-1], -lag_samples)[::-1]
    return x


def steinhart_hart_regress(
    log_R: NDArray, T_ref_degC: NDArray, order: int
) -> tuple[float, float, Optional[float], float]:
    """Fit (T_0, beta_1, beta_2, residual_std) by regressing log_R against 1/T_ref.

    Mirrors ``cal_FP07_in_situ.m``'s regression exactly: fit a degree-``order``
    polynomial of log_R against 1/T_ref, then invert each coefficient.
    """
    T_ref_inv = 1.0 / (np.asarray(T_ref_degC, dtype=float) + 273.15)
    p_desc = np.polyfit(log_R, T_ref_inv, order)  # highest degree first
    coeffs = (1.0 / p_desc)[::-1]  # ascending: [T_0, beta_1, beta_2, ...]
    T_0 = float(coeffs[0])
    beta_1 = float(coeffs[1])
    beta_2 = float(coeffs[2]) if order >= 2 else None

    predicted_inv = 1 / T_0 + log_R / beta_1
    if beta_2 is not None:
        predicted_inv = predicted_inv + log_R**2 / beta_2
    residual_std = float(np.std(1 / predicted_inv - 1 / T_ref_inv))
    return T_0, beta_1, beta_2, residual_std


@dataclass
class ProbeCalibrationFit:
    """Result of fitting one probe's in-situ calibration against a reference."""

    probe: str
    sn: str
    instrument_sn: str
    fit_file: str
    profile_index: int
    reference: str
    order: int
    old_T_0: float
    old_beta_1: float
    old_beta_2: Optional[float]
    new_T_0: float
    new_beta_1: float
    new_beta_2: Optional[float]
    lag_s: float
    lag_corr: float
    n_points: int
    temperature_range_c: float
    rms_diff_old_c: float
    rms_diff_new_c: float
    max_abs_diff_old_c: float
    max_abs_diff_new_c: float
    mean_bias_old_c: float
    mean_bias_new_c: float
    residual_std_c: float
    fit_date: str


def fit_probe_calibration(
    profile_ds: xr.Dataset,
    probe: str,
    config: ProfileConfig,
    fit_file: str = "",
    profile_index: int = -1,
    ref: str = "JAC_T",
    order: int = 2,
    min_range_c: float = 8.0,
    f_tc: float = 0.73,
    reference_speed: float = 0.62,
) -> ProbeCalibrationFit:
    """Fit ``probe``'s (e.g. ``"T1"``) in-situ calibration against ``ref``.

    ``profile_ds`` must be a single profile segment with smoothed speed (see
    :func:`pyturb.profile.prepare_profile`/``split_into_profiles``) and the
    probe's raw counts (``<probe>_counts``).
    """
    counts_name = f"{probe}_counts"
    if counts_name not in profile_ds:
        raise ValueError(
            f"{probe}: '{counts_name}' not found; reconvert with the current "
            "p2nc to enable in-situ calibration."
        )

    params = _channel_params(profile_ds, probe)
    sn = str(params.get("sn", "unknown"))
    instrument_sn = str(profile_ds.attrs.get("instrument_sn", "unknown"))
    a, b, g, e_b, _, adc_bits = _electronics(params)
    adc_fs = float(params["adc_fs"])
    old_T_0, old_beta_1, old_beta_2 = _coefficients(params)

    log_R = log_r_from_counts(
        profile_ds[counts_name].values, a, b, g, e_b, adc_fs, adc_bits
    )
    # Unclipped reconstruction of the old calibration, used for lag-finding
    # and low-pass matching (robust even where the stored T1/T2 saturated).
    T_unclipped = (
        steinhart_hart_forward(log_R, old_T_0, old_beta_1, old_beta_2) - 273.15
    )

    T_stored = np.asarray(profile_ds[probe].values, dtype=float)
    T_ref = np.asarray(profile_ds[ref].values, dtype=float)
    W = np.asarray(profile_ds[config.speed_smooth].values, dtype=float)
    fs = float(profile_ds.fs_slow)

    W_mean = float(np.nanmean(np.abs(W)))
    fc = f_tc * np.sqrt(W_mean / reference_speed)
    T_filtered = _matching_filter(T_unclipped, fs, fc)

    lag_s, lag_corr = find_lag(T_filtered, T_ref, fs)
    lag_samples = -lag_s * fs  # shift T forward by |lag| to align with T_ref
    if lag_samples > 0:
        T_ref_aligned = T_ref
    else:
        T_ref_aligned = _lag_filter(T_ref, -lag_samples)
    log_R_aligned = _shift(log_R, lag_samples)
    T_stored_aligned = _shift(T_stored, lag_samples)

    t_range = float(np.nanmax(T_ref_aligned) - np.nanmin(T_ref_aligned))
    fit_order = order
    if t_range < min_range_c and order > 1:
        _log.warning(
            f"{probe}: temperature range {t_range:.1f}C < {min_range_c:.1f}C; "
            "falling back to a first-order fit."
        )
        fit_order = 1

    new_T_0, new_beta_1, new_beta_2, residual_std = steinhart_hart_regress(
        log_R_aligned, T_ref_aligned, fit_order
    )

    T_new = steinhart_hart_forward(log_R, new_T_0, new_beta_1, new_beta_2) - 273.15
    T_new_filtered = _matching_filter(T_new, fs, fc)
    T_new_aligned = _shift(T_new_filtered, lag_samples)

    # "old" reflects the actually-stored (possibly clipped) T1/T2, since
    # apply_probe_calibration only corrects gradT1/gradT2, not T1/T2 itself.
    diff_old = T_stored_aligned - T_ref_aligned
    diff_new = T_new_aligned - T_ref_aligned

    return ProbeCalibrationFit(
        probe=probe,
        sn=sn,
        instrument_sn=instrument_sn,
        fit_file=fit_file,
        profile_index=profile_index,
        reference=ref,
        order=fit_order,
        old_T_0=old_T_0,
        old_beta_1=old_beta_1,
        old_beta_2=old_beta_2,
        new_T_0=new_T_0,
        new_beta_1=new_beta_1,
        new_beta_2=new_beta_2,
        lag_s=lag_s,
        lag_corr=lag_corr,
        n_points=int(log_R.size),
        temperature_range_c=t_range,
        rms_diff_old_c=float(np.sqrt(np.nanmean(diff_old**2))),
        rms_diff_new_c=float(np.sqrt(np.nanmean(diff_new**2))),
        max_abs_diff_old_c=float(np.nanmax(np.abs(diff_old))),
        max_abs_diff_new_c=float(np.nanmax(np.abs(diff_new))),
        mean_bias_old_c=float(np.nanmean(diff_old)),
        mean_bias_new_c=float(np.nanmean(diff_new)),
        residual_std_c=residual_std,
        fit_date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _rebuild_gradT_from_raw(
    ds: xr.Dataset, fit: ProbeCalibrationFit, params: dict
) -> NDArray:
    """Recompute gradT from raw counts with the new coefficients (exact, no clip)."""
    dT_name = f"{fit.probe}_d{fit.probe}"
    counts_name = f"{fit.probe}_counts"
    if dT_name not in ds or counts_name not in ds:
        raise ValueError(
            f"{fit.probe}: '{dT_name}'/'{counts_name}' not found; reconvert with "
            "the current p2nc to enable in-situ calibration."
        )
    dT_params = _channel_params(ds, dT_name)
    if "diff_gain" not in dT_params:
        raise ValueError(f"{dT_name}: no 'diff_gain' in its calibration parameters.")

    merged = dict(params)
    merged.update(dT_params)
    merged["t_0"] = fit.new_T_0
    merged["beta_1"] = fit.new_beta_1
    if fit.new_beta_2 is not None:
        merged["beta_2"] = fit.new_beta_2
    else:
        merged.pop("beta_2", None)

    fs_fast = float(ds.fs_fast)
    X_dX = np.asarray(ds[dT_name].values, dtype=float)
    X = np.asarray(ds[counts_name].values, dtype=float)
    T_deconvolved = deconvolve(X_dX, fs_fast, float(dT_params["diff_gain"]), X)
    return make_gradT(X_dX, merged, fs_fast, "high_pass", T_deconvolved=T_deconvolved)


def apply_probe_calibration(ds: xr.Dataset, fit: ProbeCalibrationFit) -> xr.Dataset:
    """Apply a fitted calibration to a converted file's gradT signal, in place.

    Recomputes ``gradT1``/``gradT2`` from raw counts with the new
    coefficients (exact, no data loss), shifts it by the fitted lag, and
    stamps provenance attributes. No-ops unless both the probe SN and the
    instrument SN match ``fit.sn``/``fit.instrument_sn`` -- a probe SN alone
    isn't a safe match key (e.g. generic/placeholder SNs are occasionally
    reused across probes/instruments).
    """
    grad_name = f"grad{fit.probe}"
    if fit.probe not in ds or grad_name not in ds:
        _log.warning(f"{fit.probe}/{grad_name} not found; skipping.")
        return ds

    instrument_sn = str(ds.attrs.get("instrument_sn", "unknown"))
    if instrument_sn != fit.instrument_sn:
        _log.warning(
            f"Instrument SN mismatch (file has '{instrument_sn}', fit is for "
            f"'{fit.instrument_sn}'); skipping {fit.probe}."
        )
        return ds

    params = _channel_params(ds, fit.probe)
    sn = str(params.get("sn", "unknown"))
    if sn != fit.sn:
        _log.warning(
            f"{fit.probe} SN mismatch (file has '{sn}', fit is for '{fit.sn}'); skipping."
        )
        return ds

    grad = _rebuild_gradT_from_raw(ds, fit, params)
    fs_fast = float(ds.fs_fast)
    grad = _shift(grad, -fit.lag_s * fs_fast)

    old_T_0, old_beta_1, old_beta_2 = _coefficients(params)
    ds = ds.copy()
    ds[grad_name] = (ds[grad_name].dims, grad.astype(ds[grad_name].values.dtype))
    ds[grad_name].attrs[f"{fit.probe}_fp07_recalibrated"] = np.int8(1)
    ds[grad_name].attrs[f"{fit.probe}_fp07_old_T_0"] = old_T_0
    ds[grad_name].attrs[f"{fit.probe}_fp07_old_beta_1"] = old_beta_1
    ds[grad_name].attrs[f"{fit.probe}_fp07_new_T_0"] = fit.new_T_0
    ds[grad_name].attrs[f"{fit.probe}_fp07_new_beta_1"] = fit.new_beta_1
    ds[grad_name].attrs[f"{fit.probe}_fp07_new_beta_2"] = (
        fit.new_beta_2 if fit.new_beta_2 is not None else np.nan
    )
    ds[grad_name].attrs[f"{fit.probe}_fp07_lag_s"] = fit.lag_s
    ds[grad_name].attrs[f"{fit.probe}_fp07_cal_date"] = fit.fit_date
    ds[grad_name].attrs[f"{fit.probe}_fp07_cal_source"] = (
        f"{fit.fit_file}:p{fit.profile_index}"
    )
    return ds


def write_report(fits: list[ProbeCalibrationFit], path: Path) -> None:
    """Write a calibration report (parameters + fit-quality comparison) as YAML."""
    data = {fit.probe: asdict(fit) for fit in fits}
    Path(path).write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    )


def read_report(path: Path) -> list[ProbeCalibrationFit]:
    """Read a calibration report written by :func:`write_report`."""
    data = yaml.safe_load(Path(path).read_text())
    return [ProbeCalibrationFit(**entry) for entry in data.values()]
