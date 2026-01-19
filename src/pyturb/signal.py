# Methods for signal processing, including despiking, power spectra and windowed means

import numpy as np
import scipy.signal as sig
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray
from typing import Optional


def _despike_once(
    signal: NDArray,
    thresh: float = 8.0,
    smooth: float = 0.5,
    fs: float = 512.0,
    n: Optional[int] = None,
) -> tuple[NDArray, NDArray]:
    if n is None:
        n = int(0.04 * fs)  # 40 ms of data around each spike (matches MATLAB ODAS)

    n_half = n // 2
    length = len(signal)
    pad_len = min(length, 2 * int(fs // smooth))
    pad_left = signal[:pad_len][::-1]
    pad_right = signal[-pad_len:][::-1]
    padded = np.concatenate([pad_left, signal, pad_right])

    sos_hp = sig.butter(1, 0.5 / (fs / 2), btype="high", output="sos")
    hp = np.abs(sig.sosfiltfilt(sos_hp, padded))

    sos_lp = sig.butter(1, smooth / (fs / 2), output="sos")
    lp = sig.sosfiltfilt(sos_lp, hp)

    # Only consider the original (unpadded) region
    region = slice(pad_len, pad_len + length)
    hp_region = hp[region]
    lp_region = lp[region]

    # Spike detection
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = hp_region / lp_region
    spikes = np.where(ratio > thresh)[0]

    if spikes.size == 0:
        return signal.copy(), np.array([], dtype=int)

    # Mark good points
    good = np.ones(len(padded), dtype=bool)
    for s in spikes + pad_len:
        idx = slice(max(0, s - n_half), min(len(good), s + 2 * n_half + 1))
        good[idx] = False

    # Find contiguous bad regions
    bad = ~good
    diff = np.diff(bad.astype(int))
    starts = np.where(diff == 1)[0] + 1
    stops = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if bad[0]:
        starts = np.insert(starts, 0, 0)
    if bad[-1]:
        stops = np.append(stops, len(bad))

    # Replace bad regions
    for start, stop in zip(starts, stops):
        # Only replace if region overlaps with the original (unpadded) signal
        if start >= pad_len + length or stop <= pad_len:
            continue
        # Clamp to region
        start_clamped = max(start, pad_len)
        stop_clamped = min(stop, pad_len + length)
        # Use valid points before and after for interpolation
        before = padded[
            max(pad_len, start_clamped - int(fs // (4 * smooth))) : start_clamped
        ]
        after = padded[
            stop_clamped : min(stop_clamped + int(fs // (4 * smooth)), pad_len + length)
        ]
        before = before[
            good[max(pad_len, start_clamped - int(fs // (4 * smooth))) : start_clamped]
        ]
        after = after[
            good[
                stop_clamped : min(
                    stop_clamped + int(fs // (4 * smooth)), pad_len + length
                )
            ]
        ]
        start_val = np.mean(before) if before.size > 0 else padded[start_clamped]
        stop_val = np.mean(after) if after.size > 0 else padded[stop_clamped - 1]
        padded[start_clamped:stop_clamped] = (start_val + stop_val) / 2

    # Return only the original (unpadded) region
    return padded[region], spikes


def despike(
    signal: ArrayLike,
    thresh: float = 8.0,
    smooth: float = 0.5,
    fs: float = 512.0,
    n: Optional[int] = None,
    max_passes: int = 6,
) -> tuple[NDArray, NDArray, int, float]:
    """
    Remove spikes from a signal using iterative filtering and replacement.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D signal.
    thresh : float
        Threshold for spike detection.
    smooth : float
        Low-pass filter cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    n : int
        Window size for spike replacement.
    max_passes : int, optional
        Maximum number of despike iterations. Default 6. Use 1 for ~4x faster
        processing with slightly less aggressive spike removal.

    Returns
    -------
    cleaned : np.ndarray
        Despiked signal.
    spikes : np.ndarray
        Indices of detected spikes.
    pass_count : int
        Number of despiking passes performed.
    despike_fraction : float
        Fraction of points replaced.
    """
    if signal.ndim != 1:
        raise ValueError("Input signal must be 1D.")
    if len(signal) < 5:
        raise ValueError("Signal too short for despiking.")
    cleaned = np.asarray(signal)
    all_spikes = np.array([], dtype=int)
    pass_count = 0

    for _ in range(max_passes):
        cleaned, spikes = _despike_once(cleaned, thresh, smooth, fs, n)
        if spikes.size == 0:
            break
        all_spikes = np.union1d(all_spikes, spikes)
        pass_count += 1

    despike_fraction = np.sum(cleaned != signal) / len(signal)
    return cleaned, all_spikes, pass_count, despike_fraction


def window_mean(y: ArrayLike, n_fft: int, n_diss: int):
    """Compute mean over dissipation windows.

    Assumes y has been pre-trimmed to fit an exact number of windows.
    """
    y = np.asarray(y)
    fft_overlap = n_fft // 2
    diss_step = n_diss - fft_overlap
    y_windowed = sliding_window_view(y, n_diss, writeable=True)[::diss_step, :]
    return y_windowed.mean(axis=1)


def window_psd(y: ArrayLike, fs: float, n_fft: int, n_diss: int, window: str = "hann"):
    """Compute windowed power spectral density averaged over dissipation windows.

    Assumes y has been pre-trimmed to fit an exact number of windows.
    """
    y = np.asarray(y)
    fft_overlap = n_fft // 2

    if y.ndim != 1:
        raise ValueError("y must be 1D array")
    if n_fft % 2 != 0:
        raise ValueError("n_fft must be even")
    if n_diss % n_fft != 0:
        raise ValueError("n_diss must be multiple of n_fft")

    win = sig.windows.get_window(window, n_fft)
    y_windowed = (
        sliding_window_view(y, n_fft, writeable=True)[:: n_fft - fft_overlap, :]
        * win
    )
    fft = np.fft.fft(y_windowed, axis=1)[:, : n_fft // 2 + 1]
    PSD = 2 * np.real(fft * fft.conj()) / (np.sum(win**2) * fs)
    freq = np.fft.fftfreq(n_fft, d=1 / fs)[: n_fft // 2 + 1]

    # Fix the Nyquist frequency and zero frequency, which should not be doubled
    freq[-1] *= -1
    PSD[:, [0, -1]] *= 0.5

    # Average PSDs over dissipation windows
    ffts_per_diss = (n_diss - fft_overlap) // (n_fft - fft_overlap)
    PSD = PSD.reshape(-1, ffts_per_diss, PSD.shape[1]).mean(axis=1)

    return freq, PSD
