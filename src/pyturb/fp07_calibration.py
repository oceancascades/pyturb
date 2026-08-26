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
    "MIN_LAG_CORR",
    "MIN_PLAUSIBLE_T0_K",
    "MAX_PLAUSIBLE_T0_K",
    "MIN_PLAUSIBLE_BETA1",
    "MAX_PLAUSIBLE_BETA1",
    "fit_is_plausible",
    "fit_is_confident",
    "fit_probe_calibration",
    "fit_probe_calibration_multi",
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


# A confident lag estimate: observed good fits land at lag_corr >= 0.92;
# observed bad ones (the cross-correlation search finding no real peak, or
# a probe whose signal is too noisy relative to the reference even at its
# best -- see the session investigation into VMP412 T1 SN T1592, whose best
# achievable fit still only reached lag_corr~-0.07 and correlated just
# 0.35-0.39 with the reference) land well below this.
MIN_LAG_CORR = 0.7

# steinhart_hart_regress fits 1/T_ref as a polynomial in log_R, then inverts
# the coefficients to get (T_0, beta_1, beta_2). T_0 is, by construction,
# whatever that fit predicts AT log_R=0 -- if the fitted data doesn't sit
# near 0, T_0 is an extrapolation, not a measurement, and a low-order
# polynomial can fit a narrow, offset window of real data beautifully (a
# confident lag, a tiny residual) while still swinging to nonsense once
# projected back to log_R=0. A probe whose resistance baseline has drifted
# (e.g. a failing connection, well short of full ADC railing) produces
# exactly this: fits that look perfect on their own segment but generalize
# catastrophically. T_0 and beta_1 are real physical quantities (a
# reference temperature; a positive thermistor material constant) that
# should land close to the factory defaults (T_0=289.3K, beta_1=3143.55)
# even for a genuinely different but still-working probe -- never hundreds
# of Kelvin off, never negative.
MIN_PLAUSIBLE_T0_K = 250.0
MAX_PLAUSIBLE_T0_K = 350.0
MIN_PLAUSIBLE_BETA1 = 1500.0
MAX_PLAUSIBLE_BETA1 = 6000.0


def fit_is_plausible(fit: ProbeCalibrationFit) -> bool:
    """False if the fitted T_0/beta_1 are outside a physically plausible
    range -- see the module comment above MIN_PLAUSIBLE_T0_K."""
    return (
        MIN_PLAUSIBLE_T0_K <= fit.new_T_0 <= MAX_PLAUSIBLE_T0_K
        and MIN_PLAUSIBLE_BETA1 <= fit.new_beta_1 <= MAX_PLAUSIBLE_BETA1
    )


def fit_is_confident(fit: ProbeCalibrationFit) -> bool:
    """True if fit is both physically plausible and has a confident lag
    estimate -- the bar 'calibrate-fp07 auto' uses to accept a fit
    immediately, and (via apply_probe_calibration's provenance attrs) what
    pyturb.profile's T1_qc/T2_qc/chi_N_qc floor to "bad" when not met.
    Plausible-but-unconfident fits are real but empirically still
    inaccurate (see MIN_LAG_CORR's docstring) -- there is no "trustworthy
    but not immediately accepted" tier.
    """
    return fit_is_plausible(fit) and abs(fit.lag_corr) >= MIN_LAG_CORR


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
    min_pressure_dbar: float = 1.0,
) -> ProbeCalibrationFit:
    """Fit ``probe``'s (e.g. ``"T1"``) in-situ calibration against ``ref``.

    ``profile_ds`` must be a single profile segment with smoothed speed (see
    :func:`pyturb.profile.prepare_profile`/``split_into_profiles``) and the
    probe's raw counts (``<probe>_counts``).

    The regression itself only uses samples with pressure (``config.
    pressure_smooth``) above ``min_pressure_dbar`` -- a profile segment
    starts at the surface, before the vehicle reaches depth, and near-
    surface/out-of-water samples shouldn't be allowed to pull the fit
    (lag-finding still uses the full segment, which needs continuous data
    for its cross-correlation/filtering to be meaningful).
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

    if config.pressure_smooth in profile_ds:
        deep_mask = (
            np.asarray(profile_ds[config.pressure_smooth].values, dtype=float)
            > min_pressure_dbar
        )
    else:
        deep_mask = np.ones_like(log_R_aligned, dtype=bool)
    log_R_fit = log_R_aligned[deep_mask]
    T_ref_fit = T_ref_aligned[deep_mask]

    t_range = float(np.nanmax(T_ref_fit) - np.nanmin(T_ref_fit))
    fit_order = order
    if t_range < min_range_c and order > 1:
        _log.warning(
            f"{probe}: temperature range {t_range:.1f}C < {min_range_c:.1f}C; "
            "falling back to a first-order fit."
        )
        fit_order = 1

    new_T_0, new_beta_1, new_beta_2, residual_std = steinhart_hart_regress(
        log_R_fit, T_ref_fit, fit_order
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


def fit_probe_calibration_multi(
    profile_list: list[tuple[int, xr.Dataset]],
    probe: str,
    config: ProfileConfig,
    fit_file: str = "",
    ref: str = "JAC_T",
    order: int = 2,
    min_range_c: float = 8.0,
    f_tc: float = 0.73,
    reference_speed: float = 0.62,
    min_pressure_dbar: float = 1.0,
) -> ProbeCalibrationFit:
    """Fit ``probe``'s in-situ calibration aggregated across every profile in
    ``profile_list``, instead of a single best one.

    Mirrors the mousebrains/``odas_tpw`` package's approach: median lag
    across every profile in a file, then a single Steinhart-Hart regression
    on the concatenated data from all of them, rather than pyturb's older
    single-best-profile fit. The median is far less sensitive to any one
    profile's noisy lag estimate than trusting a single profile's search
    outright, and concatenating data from profiles spanning different
    depths/conditions gives the regression a wider, better-conditioned
    ``log_R`` range to fit -- directly countering the narrow/offset-window
    fragility a single profile's fit can have (see the session
    investigation into VMP412 T1 SN T2146/T1592). Verified at least as
    accurate as the single-profile fit on every well-behaved probe tested,
    so this is now the only fitting strategy ``calibrate-fp07 auto`` uses.

    The regression itself only uses samples with pressure (``config.
    pressure_smooth``) above ``min_pressure_dbar`` from each profile -- a
    profile segment starts at the surface, before the vehicle reaches
    depth, and near-surface/out-of-water samples shouldn't be allowed to
    pull the fit (lag-finding still uses each full segment, which needs
    continuous data for its cross-correlation/filtering to be meaningful).

    ``profile_list`` is e.g. the output of
    :func:`pyturb.profile.split_into_profiles`. ``profile_index`` on the
    returned fit is ``-1``, a sentinel meaning "aggregate of multiple
    profiles" (see :func:`apply_probe_calibration`'s provenance attrs).
    """
    counts_name = f"{probe}_counts"

    params: Optional[dict] = None
    sn = instrument_sn = "unknown"
    old_T_0 = old_beta_1 = old_beta_2 = None
    a = b = g = e_b = adc_fs = adc_bits = None

    # (log_R, T_ref, T_stored, fs, fc, P) per usable profile.
    per_profile: list[tuple[NDArray, NDArray, NDArray, float, float, NDArray]] = []
    lags: list[float] = []
    corrs: list[float] = []

    for _, profile_ds in profile_list:
        if counts_name not in profile_ds or ref not in profile_ds:
            continue
        if params is None:
            params = _channel_params(profile_ds, probe)
            sn = str(params.get("sn", "unknown"))
            instrument_sn = str(profile_ds.attrs.get("instrument_sn", "unknown"))
            a, b, g, e_b, _, adc_bits = _electronics(params)
            adc_fs = float(params["adc_fs"])
            old_T_0, old_beta_1, old_beta_2 = _coefficients(params)

        log_R = log_r_from_counts(
            profile_ds[counts_name].values, a, b, g, e_b, adc_fs, adc_bits
        )
        T_unclipped = (
            steinhart_hart_forward(log_R, old_T_0, old_beta_1, old_beta_2) - 273.15
        )
        T_stored = np.asarray(profile_ds[probe].values, dtype=float)
        T_ref = np.asarray(profile_ds[ref].values, dtype=float)
        W = np.asarray(profile_ds[config.speed_smooth].values, dtype=float)
        fs = float(profile_ds.fs_slow)
        P = (
            np.asarray(profile_ds[config.pressure_smooth].values, dtype=float)
            if config.pressure_smooth in profile_ds
            else np.full(log_R.shape, np.inf)
        )

        W_mean = float(np.nanmean(np.abs(W)))
        fc = f_tc * np.sqrt(W_mean / reference_speed)
        T_filtered = _matching_filter(T_unclipped, fs, fc)

        lag_s, lag_corr = find_lag(T_filtered, T_ref, fs)
        # A flatlined/non-finite segment gives a non-finite lag; drop it so
        # it cannot poison the median.
        if not np.isfinite(lag_s):
            continue
        lags.append(lag_s)
        corrs.append(lag_corr)
        per_profile.append((log_R, T_ref, T_stored, fs, fc, P))

    if params is None or not lags:
        raise ValueError(
            f"{probe}: no usable profiles (need '{counts_name}' and '{ref}') "
            "to fit an aggregate calibration."
        )

    median_lag_s = float(np.median(lags))
    median_corr = float(np.median(corrs))

    log_R_parts, T_ref_parts, T_stored_parts = [], [], []
    log_R_fit_parts, T_ref_fit_parts = [], []
    for log_R, T_ref, T_stored, fs, _fc, P in per_profile:
        lag_samples = -median_lag_s * fs
        T_ref_aligned = T_ref if lag_samples > 0 else _lag_filter(T_ref, -lag_samples)
        log_R_aligned_p = _shift(log_R, lag_samples)
        log_R_parts.append(log_R_aligned_p)
        T_ref_parts.append(T_ref_aligned)
        T_stored_parts.append(_shift(T_stored, lag_samples))

        deep_mask = P > min_pressure_dbar
        log_R_fit_parts.append(log_R_aligned_p[deep_mask])
        T_ref_fit_parts.append(T_ref_aligned[deep_mask])

    log_R_aligned = np.concatenate(log_R_parts)
    T_ref_aligned = np.concatenate(T_ref_parts)
    T_stored_aligned = np.concatenate(T_stored_parts)
    log_R_fit = np.concatenate(log_R_fit_parts)
    T_ref_fit = np.concatenate(T_ref_fit_parts)

    t_range = float(np.nanmax(T_ref_fit) - np.nanmin(T_ref_fit))
    fit_order = order
    if t_range < min_range_c and order > 1:
        _log.warning(
            f"{probe}: aggregate temperature range {t_range:.1f}C < "
            f"{min_range_c:.1f}C; falling back to a first-order fit."
        )
        fit_order = 1

    new_T_0, new_beta_1, new_beta_2, residual_std = steinhart_hart_regress(
        log_R_fit, T_ref_fit, fit_order
    )

    T_new_parts = []
    for log_R, _T_ref, _T_stored, fs, fc, _P in per_profile:
        T_new = steinhart_hart_forward(log_R, new_T_0, new_beta_1, new_beta_2) - 273.15
        T_new_filtered = _matching_filter(T_new, fs, fc)
        lag_samples = -median_lag_s * fs
        T_new_parts.append(_shift(T_new_filtered, lag_samples))
    T_new_aligned = np.concatenate(T_new_parts)

    diff_old = T_stored_aligned - T_ref_aligned
    diff_new = T_new_aligned - T_ref_aligned

    return ProbeCalibrationFit(
        probe=probe,
        sn=sn,
        instrument_sn=instrument_sn,
        fit_file=fit_file,
        profile_index=-1,
        reference=ref,
        order=fit_order,
        old_T_0=old_T_0,
        old_beta_1=old_beta_1,
        old_beta_2=old_beta_2,
        new_T_0=new_T_0,
        new_beta_1=new_beta_1,
        new_beta_2=new_beta_2,
        lag_s=median_lag_s,
        lag_corr=median_corr,
        n_points=int(log_R_aligned.size),
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


def _rebuild_T_from_raw(
    ds: xr.Dataset, fit: ProbeCalibrationFit, params: dict
) -> NDArray:
    """Recompute the standalone T1/T2 temperature from raw counts with the
    new coefficients (no clip -- see log_r_from_counts)."""
    counts_name = f"{fit.probe}_counts"
    if counts_name not in ds:
        raise ValueError(
            f"{fit.probe}: '{counts_name}' not found; reconvert with the "
            "current p2nc to enable in-situ calibration."
        )
    a, b, g, e_b, _, adc_bits = _electronics(params)
    adc_fs = float(params["adc_fs"])
    log_R = log_r_from_counts(ds[counts_name].values, a, b, g, e_b, adc_fs, adc_bits)
    return (
        steinhart_hart_forward(log_R, fit.new_T_0, fit.new_beta_1, fit.new_beta_2)
        - 273.15
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
    """Apply a fitted calibration to a converted file, in place.

    Recomputes both the standalone temperature (``T1``/``T2``) and the
    gradient (``gradT1``/``gradT2``) from raw counts with the new
    coefficients (exact, no data loss) and shifts each by the fitted lag.
    Applies the best available calibration as-is, with no NaN-masking or
    other data destruction here -- a fit extrapolated to raw counts far
    outside its fitted range (e.g. an anomalous profile) can still produce
    physically implausible values, but flagging that is the ``eps`` step's
    job (it checks the raw T1/T2 range and the chi fit quality per window
    and marks QC there), not this conversion step's. Stamps provenance
    attributes. No-ops unless both the probe SN and the instrument SN match
    ``fit.sn``/``fit.instrument_sn`` -- a probe SN alone isn't a safe match
    key (e.g. generic/placeholder SNs are occasionally reused across
    probes/instruments).
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

    fs_slow = float(ds.fs_slow)
    fs_fast = float(ds.fs_fast)

    T = _rebuild_T_from_raw(ds, fit, params)
    T = _shift(T, -fit.lag_s * fs_slow)

    grad = _rebuild_gradT_from_raw(ds, fit, params)
    grad = _shift(grad, -fit.lag_s * fs_fast)

    old_T_0, old_beta_1, old_beta_2 = _coefficients(params)
    # Save attrs before the bare (dims, data) assignments below, which would
    # otherwise silently wipe them -- including the cal_* attrs that a later
    # fit's SN-mismatch check (_channel_params, above) depends on. Losing
    # them mid-run raised an uncaught ValueError on the next fit for the
    # same probe, killing 'calibrate-fp07 auto' partway through a batch with
    # no indication beyond the traceback -- and since callers loop through
    # every fit against every file relying on the mismatch check to no-op,
    # the very first successful match for a probe silently broke every
    # subsequent fit attempt for that same probe.
    probe_attrs = dict(ds[fit.probe].attrs)
    grad_attrs = dict(ds[grad_name].attrs)
    ds = ds.copy()
    ds[fit.probe] = (ds[fit.probe].dims, T.astype(ds[fit.probe].values.dtype))
    ds[grad_name] = (ds[grad_name].dims, grad.astype(ds[grad_name].values.dtype))
    ds[fit.probe].attrs.update(probe_attrs)
    ds[grad_name].attrs.update(grad_attrs)
    for name in (fit.probe, grad_name):
        ds[name].attrs[f"{fit.probe}_fp07_recalibrated"] = np.int8(1)
        ds[name].attrs[f"{fit.probe}_fp07_old_T_0"] = old_T_0
        ds[name].attrs[f"{fit.probe}_fp07_old_beta_1"] = old_beta_1
        ds[name].attrs[f"{fit.probe}_fp07_new_T_0"] = fit.new_T_0
        ds[name].attrs[f"{fit.probe}_fp07_new_beta_1"] = fit.new_beta_1
        ds[name].attrs[f"{fit.probe}_fp07_new_beta_2"] = (
            fit.new_beta_2 if fit.new_beta_2 is not None else np.nan
        )
        ds[name].attrs[f"{fit.probe}_fp07_lag_s"] = fit.lag_s
        ds[name].attrs[f"{fit.probe}_fp07_lag_corr"] = fit.lag_corr
        ds[name].attrs[f"{fit.probe}_fp07_cal_date"] = fit.fit_date
        ds[name].attrs[f"{fit.probe}_fp07_cal_source"] = (
            f"{fit.fit_file}:p{fit.profile_index}"
        )
        # Whether the *fit itself* was trustworthy (see fit_is_confident) --
        # not whether the resulting values happen to look physically sane.
        # A fit from a probe whose signal is too noisy relative to the
        # reference (e.g. VMP412 T1 SN T1592) can still produce values
        # within a normal-looking temperature range while being
        # substantially wrong; pyturb.profile's T1_qc/T2_qc/chi_N_qc read
        # this to flag that case, since no per-sample check can catch it.
        ds[name].attrs[f"{fit.probe}_fp07_confident"] = np.int8(
            1 if fit_is_confident(fit) else 0
        )
    return ds


def write_report(fits: list[ProbeCalibrationFit], path: Path) -> None:
    """Write a calibration report (parameters + fit-quality comparison) as YAML.

    A plain list, keyed by nothing -- multiple fits can share the same
    ``probe`` (e.g. the same channel calibrated separately per instrument/SN
    by 'calibrate-fp07 auto'), so a dict keyed by probe would silently drop
    all but one.
    """
    data = [asdict(fit) for fit in fits]
    Path(path).write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    )


def read_report(path: Path) -> list[ProbeCalibrationFit]:
    """Read a calibration report written by :func:`write_report`."""
    data = yaml.safe_load(Path(path).read_text())
    return [ProbeCalibrationFit(**entry) for entry in data]
