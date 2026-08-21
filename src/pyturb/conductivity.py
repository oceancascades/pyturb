"""Lag and low-pass filter conductivity to match a co-located thermometer's response."""

import logging

import numpy as np
import scipy.signal as sig
from numpy.typing import ArrayLike, NDArray

_log = logging.getLogger(__name__)

__all__ = ["match_conductivity_to_temperature"]


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
