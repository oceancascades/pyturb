"""
Deconvolve pre-emphasized signals using Mudge & Lueck (1994) algorithm.
"""

from typing import Optional

import numpy as np
from scipy import interpolate, signal


def deconvolve(
    X_dX: np.ndarray, fs: float, diff_gain: float, X: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Deconvolve pre-emphasized signal (x + gain*dx/dt) to high-resolution data.

    Parameters
    ----------
    X_dX : ndarray
        Pre-emphasized signal (e.g., T1_dT1, P_dP)
    fs : float
        Sampling rate in Hz
    diff_gain : float
        Differentiator time constant (seconds)
    X : ndarray, optional
        Low-resolution signal for improved initial conditions

    Returns
    -------
    ndarray
        Deconvolved high-resolution signal
    """
    # Interpolate X to match X_dX length if needed
    if X is not None and len(X) > 1:
        X = _interp1_if_required(X, X_dX, fs)

    # Calculate filter parameters
    f_c = 1 / (2 * np.pi * diff_gain)  # Cut-off frequency
    sos = signal.butter(1, f_c / (fs / 2), output="sos")

    # Calculate initial conditions for the filter
    if X is not None and len(X) > 1:
        # Use both signals to determine initial conditions
        # Check if X_dX is inverted relative to X
        with np.errstate(all="ignore"):
            p = np.polyfit(X[: min(100, len(X))], X_dX[: min(100, len(X_dX))], 1)
        if p[0] < -0.5:
            X_dX = -X_dX

        # Initial condition based on first data point
        zi = _sosfilt_zi_from_values(sos, X[0], X_dX[0])
    else:
        # For T/C without low-res signal, use first portion of record
        # Base initial conditions on linear fit to beginning of signal
        timeV_len = min(int(2 * diff_gain * fs) + 1, len(X_dX))
        timeV = np.arange(timeV_len) / fs
        p = np.polyfit(timeV, X_dX[:timeV_len], 1)
        previous_output = p[1] - diff_gain * p[0]
        zi = _sosfilt_zi_from_values(sos, previous_output, X_dX[0])

    # Deconvolve using low-pass filter
    X_hires, _ = signal.sosfilt(sos, X_dX, zi=zi)

    # For signals with low-res version, regress to remove offset
    if X is not None and len(X) > 1:
        # Fit high-res to low-res
        with np.errstate(all="ignore"):
            p = np.polyfit(X_hires, X, 1)

        # Calculate adjusted initial conditions
        p2 = np.array([2 - p[0], -p[1]])
        initial_output = np.polyval(p2, X[0])
        zi = _sosfilt_zi_from_values(sos, initial_output, X_dX[0])

        # Re-deconvolve with better initial conditions
        X_hires, _ = signal.sosfilt(sos, X_dX, zi=zi)

        # Apply calibration to match low-res signal
        X_hires = np.polyval(p, X_hires)

    return X_hires


def _sosfilt_zi_from_values(sos: np.ndarray, y0: float, x0: float) -> np.ndarray:
    """
    Compute sosfilt initial conditions from initial input/output values.

    Parameters
    ----------
    sos : ndarray
        Second-order sections filter coefficients.
    y0 : float
        Initial output value.
    x0 : float
        Initial input value.

    Returns
    -------
    ndarray
        Initial conditions for sosfilt, shape (n_sections, 2).
    """
    # For a first-order Butterworth in SOS form, we have one section
    # SOS format: [b0, b1, b2, 1, a1, a2] where a0=1
    # For 1st order: b2=0, a2=0
    n_sections = sos.shape[0]
    zi = np.zeros((n_sections, 2))

    # For a 1st-order filter, only z[0] is used
    # The state represents: z[0] = b1*x[n] - a1*y[n]
    for i in range(n_sections):
        b0, b1, b2, a0, a1, a2 = sos[i]
        # Initial state: z1 = b1*x0 - a1*y0
        zi[i, 0] = b1 * x0 - a1 * y0
        zi[i, 1] = b2 * x0 - a2 * y0

    return zi


def _interp1_if_required(X: np.ndarray, X_dX: np.ndarray, fs: float) -> np.ndarray:
    """
    Interpolate X to match X_dX length if they differ.

    Parameters
    ----------
    X : ndarray
        Low-resolution signal
    X_dX : ndarray
        High-resolution pre-emphasized signal
    fs : float
        Sampling rate of X_dX

    Returns
    -------
    ndarray
        X interpolated to match X_dX length, or original X if no interpolation needed
    """
    if len(X) == len(X_dX) or len(X) <= 1:
        return X

    # Calculate slow sampling rate
    f_slow = fs * len(X) / len(X_dX)

    # Create time vectors
    t_slow = np.arange(len(X)) / f_slow
    t_fast = np.arange(len(X_dX)) / fs

    # Interpolate using cubic spline
    interp_func = interpolate.PchipInterpolator(t_slow, X, extrapolate=True)
    newX = interp_func(t_fast)

    return newX
