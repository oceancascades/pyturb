"""Fit and apply the lag and low-pass filter matching conductivity to a co-located thermometer."""

import logging
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np
import scipy.signal as sig
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .signal import despike

_log = logging.getLogger(__name__)

__all__ = [
    "CTResponseFit",
    "fit_ct_response",
    "match_conductivity_to_temperature",
]

_SEGMENT_SEC = 8.0
_MIN_SEGMENTS = 20
_FIT_BAND_HZ = (0.1, 2.5)
_MIN_COHERENCE = 0.5
_TAU_BOUNDS = (0.05, 0.4)
_LAG_BOUNDS = (0.0, 0.15)


@dataclass
class CTResponseFit:
    """Thermometer response relative to conductivity, from :func:`fit_ct_response`."""

    lag: float  # s
    tau: float  # s
    n_segments: int
    speed: float  # dbar/s, mean over the segments used


def fit_ct_response(
    records: Iterable[tuple[ArrayLike, ArrayLike, ArrayLike]],
    fs: float,
    direction: Literal["down", "up", "both"] = "down",
    min_speed: float = 0.5,
    min_temperature_change: float = 0.3,
    min_pressure: float = 3.0,
) -> Optional[CTResponseFit]:
    """Fit the lag and time constant of a thermometer relative to conductivity.

    Conductivity follows temperature closely, so their cross-spectrum gives the
    thermometer's response: it is fitted to a single pole of time constant
    ``tau`` plus a pure delay ``lag``. Delaying conductivity by ``lag`` and
    low-pass filtering it with ``tau`` then matches it to the thermometer.

    Parameters
    ----------
    records : iterable of (T, C, P)
        Temperature, conductivity and pressure (dbar) arrays sampled at ``fs``.
        Spectra are pooled over all records, e.g. every file of one instrument.
    fs : float
        Sampling rate, in Hz.
    direction : {"down", "up", "both"}, default "down"
        Profiling direction of the segments to use.
    min_speed : float, default 0.5
        Minimum profiling speed of a segment, in dbar/s.
    min_temperature_change : float, default 0.3
        Minimum temperature change across a segment.
    min_pressure : float, default 3.0
        Segments reaching shallower than this (dbar) are skipped.

    Returns
    -------
    CTResponseFit or None
        None if too few segments qualify or the fit is implausible.
    """
    n = round(_SEGMENT_SEC * fs)
    window = np.hanning(n)
    TT = CC = TC = 0.0
    speeds = []
    for T, C, P in records:
        T, C, P = (np.asarray(x, dtype=float) for x in (T, C, P))
        if len(C) > n and np.isfinite(C).all():
            C = despike(C, fs=fs)[0]
        for i in range(0, len(P) - n, n // 4):
            s = slice(i, i + n)
            speed = (P[i + n - 1] - P[i]) / _SEGMENT_SEC
            if direction == "up":
                speed = -speed
            elif direction == "both":
                speed = abs(speed)
            if not (
                speed > min_speed
                and P[s].min() > min_pressure
                and abs(T[i + n - 1] - T[i]) > min_temperature_change
                and np.isfinite(C[s]).all()
            ):
                continue
            # First difference to whiten the red spectra before windowing
            dT, dC = np.diff(T[s], prepend=T[i]), np.diff(C[s], prepend=C[i])
            FT = np.fft.rfft(window * (dT - dT.mean()))
            FC = np.fft.rfft(window * (dC - dC.mean()))
            TT, CC, TC = TT + np.abs(FT) ** 2, CC + np.abs(FC) ** 2, TC + FT.conj() * FC
            speeds.append(speed)

    if len(speeds) < _MIN_SEGMENTS:
        return None

    f = np.fft.rfftfreq(n, 1 / fs)
    coherence = np.abs(TC) ** 2 / (TT * CC)
    band = (f > _FIT_BAND_HZ[0]) & (f < _FIT_BAND_HZ[1]) & (coherence > _MIN_COHERENCE)
    if band.sum() < 6:
        return None
    H, w, weight = (TC / TT)[band], 2j * np.pi * f[band], coherence[band]

    def cost(p):
        gain, tau, lag = p
        return np.sum(
            weight * np.abs(gain * (1 + w * tau) * np.exp(w * lag) / H - 1) ** 2
        )

    _, tau, lag = minimize(cost, [np.abs(H[0]), 0.15, 0.05], method="Nelder-Mead").x
    if not (
        _TAU_BOUNDS[0] < tau < _TAU_BOUNDS[1] and _LAG_BOUNDS[0] < lag < _LAG_BOUNDS[1]
    ):
        return None
    return CTResponseFit(float(lag), float(tau), len(speeds), float(np.mean(speeds)))


def _lag_filter(C: NDArray, lag_samples: float) -> NDArray:
    """Delay C by a (possibly fractional) number of samples."""
    n = int(np.ceil(lag_samples))
    b = np.zeros(n + 1)
    b[-2] = n - lag_samples
    b[-1] = 1.0 - b[-2]

    x = np.arange(len(b) + 1)
    coeffs = np.polyfit(x, C[: len(b) + 1], 1)
    previous_inputs = np.polyval(coeffs, -x[1:])  # most-recent-first

    zi = sig.lfiltic(b, [1.0], [], x=previous_inputs)
    return sig.lfilter(b, [1.0], C, zi=zi)[0]


def _matching_filter(C: NDArray, fs: float, f_tc: float) -> NDArray:
    """Single-pole low-pass filter matching C's response to the thermometer's."""
    b, a = sig.butter(1, f_tc / (fs / 2))
    delay = 1.0 / (2 * np.pi * f_tc)

    n_x = int(round(delay * fs)) + 2
    x = np.arange(n_x)
    coeffs = np.polyfit(x, C[:n_x], 1)
    initial_input = np.polyval(coeffs, -1)
    initial_output = np.polyval(coeffs, -x[-1])

    zi = sig.lfiltic(b, a, [initial_output], x=[initial_input])
    return sig.lfilter(b, a, C, zi=zi)[0]


def match_conductivity_to_temperature(
    C: ArrayLike,
    fs: float,
    speed: float,
    lag: float = 0.0234,
    f_tc: float = 0.73,
    reference_speed: float = 0.62,
) -> NDArray:
    """Lag- and low-pass-match a conductivity signal to a co-located thermometer.

    Parameters
    ----------
    C : array_like
        Conductivity signal.
    fs : float
        Sampling rate of C, in Hz.
    speed : float
        Mean profiling speed (m/s).
    lag : float, default 0.0234
        Lag of C relative to temperature at ``reference_speed``, in seconds.
    f_tc : float, default 0.73
        Matching low-pass filter cutoff at ``reference_speed``, in Hz.
    reference_speed : float, default 0.62
        Speed at which ``lag`` and ``f_tc`` were characterized, in m/s.

    Returns
    -------
    ndarray
        Matched conductivity, same length as ``C``.
    """
    C = np.asarray(C, dtype=float)

    if not np.isfinite(speed) or speed <= 0:
        _log.warning(f"Invalid speed ({speed}); skipping conductivity matching.")
        return C

    scaled_lag = lag * reference_speed / speed
    scaled_f_tc = f_tc * np.sqrt(speed / reference_speed)

    # As speed -> 0, scaled_lag and the filter delay both -> inf.
    delay_samples = fs / (2 * np.pi * scaled_f_tc) if scaled_f_tc > 0 else np.inf
    if scaled_lag * fs >= len(C) or delay_samples >= len(C):
        _log.warning(
            f"speed ({speed:.4g} m/s) too small relative to reference_speed "
            f"({reference_speed:.4g} m/s) for a {len(C)}-sample signal; "
            "skipping conductivity matching."
        )
        return C

    try:
        matched = C
        if scaled_lag > 0:
            matched = _lag_filter(matched, scaled_lag * fs)
        return _matching_filter(matched, fs, scaled_f_tc)
    except Exception as e:
        _log.warning(
            f"Conductivity matching failed ({e}); using unmatched conductivity."
        )
        return C
