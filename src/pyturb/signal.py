# Methods for signal processing, including despiking, power spectra and windowed means

import logging
from typing import Optional

import numpy as np
import scipy.signal as sig
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

_log = logging.getLogger(__name__)


def despike_mask_name(var: str) -> str:
    """Standard name for the per-sample despike mask companion of ``var``."""
    return f"{var}_despike_mask"


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


def block_mean(y: ArrayLike, n: int) -> np.ndarray:
    """Mean over consecutive, non-overlapping blocks of n samples.

    Trailing samples that don't fill a complete block are dropped.
    """
    y = np.asarray(y)
    n_blocks = len(y) // n
    if n_blocks == 0:
        return np.array([], dtype=float)
    return y[: n_blocks * n].reshape(n_blocks, n).mean(axis=1)


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
        sliding_window_view(y, n_fft, writeable=True)[:: n_fft - fft_overlap, :] * win
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


def clean_spec(
    y: np.ndarray,
    y_c: np.ndarray,
    n_fft: int,
    fs: float,
    n_diss: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Remove coherent contamination from spectra via the Goodman method.

    Given a time series ``y`` and one or more reference (coherent / correlated)
    signals ``y_c``, this returns the auto-spectra of ``y`` with the portion
    coherent with ``y_c`` subtracted in the cross-spectral domain. The classic
    use case is removing accelerometer- or EM-current-coherent vibration from
    shear-probe spectra, but the algorithm is general: any reference signals
    expected to share linear, coherent power with ``y`` can be used.

    Operates entirely in the spectral domain. Computes ensemble-averaged
    cross-spectral matrices over the FFT segments within each dissipation
    window and applies the Lueck/Goodman correction
    ``clean(YY) = YY - YC @ inv(CC) @ YC^H`` per (window, frequency) bin.
    A bias correction (RSI Technical Note 61) is then applied to compensate
    for the reduction in effective degrees of freedom.

    Parameters
    ----------
    y : ndarray, shape (N,) or (N, n_y)
        Time series to be cleaned. Each column is an independent channel.
    y_c : ndarray, shape (N,) or (N, n_c)
        Reference time series carrying the coherent component to remove.
        Each column is one reference (e.g. one accelerometer axis or one
        EM-current component).
    n_fft : int
        FFT segment length (must be even).
    fs : float
        Sampling rate (Hz).
    n_diss : int
        Dissipation-window length in samples (must be a multiple of n_fft).
    window : str, optional
        Window function name (default ``"hann"``).

    Returns
    -------
    freq : ndarray, shape (n_fft // 2 + 1,)
        Frequency vector (Hz).
    clean_psd : ndarray, shape (n_windows, n_y, n_fft // 2 + 1)
        Cleaned auto-spectra of ``y`` averaged over dissipation windows. If
        ``y`` has a single channel the channel axis is squeezed out, leaving
        shape ``(n_windows, n_fft // 2 + 1)``.
    """
    y = np.asarray(y, dtype=np.float64)
    y_c = np.asarray(y_c, dtype=np.float64)

    if y.ndim == 1:
        y = y[:, np.newaxis]
    if y_c.ndim == 1:
        y_c = y_c[:, np.newaxis]
    if y.shape[0] != y_c.shape[0]:
        raise ValueError("y and y_c must have the same number of rows")

    n_y = y.shape[1]
    n_c = y_c.shape[1]
    fft_overlap = n_fft // 2
    n_freq = n_fft // 2 + 1
    step = n_fft - fft_overlap

    # Build window and normalisation factor
    win = sig.windows.get_window(window, n_fft).astype(np.float64)
    norm = np.sum(win**2) * fs  # power-spectrum normalisation

    # ------------------------------------------------------------------
    # Segment all channels using sliding_window_view
    # sliding_window_view with axis=0 on (N, n_ch) gives (n_seg, n_ch, n_fft)
    # We transpose to (n_seg, n_fft, n_ch) for consistent downstream use.
    # ------------------------------------------------------------------
    y_segs = np.array(
        sliding_window_view(y, n_fft, axis=0)[::step], dtype=np.float64
    ).transpose(0, 2, 1)
    yc_segs = np.array(
        sliding_window_view(y_c, n_fft, axis=0)[::step], dtype=np.float64
    ).transpose(0, 2, 1)

    # Apply window: (n_seg, n_fft, n_ch) * (n_fft,)
    y_segs *= win[np.newaxis, :, np.newaxis]
    yc_segs *= win[np.newaxis, :, np.newaxis]

    # Linear detrend each segment (matches MATLAB 'linear' method)
    x = np.linspace(0.0, 1.0, n_fft, dtype=np.float64)
    xm = x - x.mean()
    xm_ss = np.dot(xm, xm)  # sum of squares
    for segs in (y_segs, yc_segs):
        # segs: (n_seg, n_fft, n_ch)
        mean_y = segs.mean(axis=1, keepdims=True)
        slope = np.einsum("stc,t->sc", segs, xm) / xm_ss
        segs -= mean_y + slope[:, np.newaxis, :] * xm[np.newaxis, :, np.newaxis]

    # One-sided FFT: (n_seg, n_freq, n_ch)
    Y = np.fft.rfft(y_segs, axis=1)
    C = np.fft.rfft(yc_segs, axis=1)

    # ------------------------------------------------------------------
    # Group FFT segments into dissipation windows and build ensemble-
    # averaged cross-spectral matrices.
    # ------------------------------------------------------------------
    ffts_per_diss = (n_diss - fft_overlap) // step
    n_seg_total = Y.shape[0]
    n_windows = n_seg_total // ffts_per_diss

    # Trim to exact number of complete dissipation windows
    n_seg_used = n_windows * ffts_per_diss
    Y = Y[:n_seg_used].reshape(n_windows, ffts_per_diss, n_freq, n_y)
    C = C[:n_seg_used].reshape(n_windows, ffts_per_diss, n_freq, n_c)

    # Scale factor for one-sided spectrum (×2), then halve DC and Nyquist
    scale = 2.0 / norm

    # Cross-spectral matrices averaged over segments within each window
    # Axes: w=window, s=segment, f=freq, i/j=channel
    # YY: (n_windows, n_freq, n_y, n_y)
    YY = scale * np.einsum("wsfi,wsfj->wfij", Y, Y.conj()) / ffts_per_diss
    CC = scale * np.einsum("wsfi,wsfj->wfij", C, C.conj()) / ffts_per_diss
    YC = scale * np.einsum("wsfi,wsfj->wfij", Y, C.conj()) / ffts_per_diss

    # Fix DC and Nyquist (should not be doubled)
    for M in (YY, CC, YC):
        M[:, 0, :, :] *= 0.5
        M[:, -1, :, :] *= 0.5

    # ------------------------------------------------------------------
    # Goodman cleaning: clean_YY = YY - YC @ inv(CC) @ YC^H
    # Solve via np.linalg.solve for numerical stability.
    # CC @ X = YC^H  =>  X = inv(CC) @ YC^H
    # Then  clean_YY = YY - YC @ X
    # ------------------------------------------------------------------
    # YC^H: (w, f, n_c, n_y) — conjugate transpose of last two axes
    YC_H = np.conj(np.swapaxes(YC, -2, -1))

    # Solve CC @ X = YC_H for X: (w, f, n_c, n_y)
    # In low-signal or perfectly coherent synthetic cases, CC can be singular
    # at some (window, frequency) bins. Fall back to pseudo-inverse so the
    # cleaner remains numerically robust across platforms/LAPACK builds.
    try:
        X = np.linalg.solve(CC, YC_H)
    except np.linalg.LinAlgError:
        _log.debug("Goodman solve encountered singular CC; using pinv fallback")
        CC_pinv = np.linalg.pinv(CC)
        X = np.einsum("wfij,wfjk->wfik", CC_pinv, YC_H)

    # Correction: YC @ X  -> (w, f, n_y, n_y)
    correction = np.einsum("wfij,wfjk->wfik", YC, X)

    clean_YY = YY - correction

    # Take real part of the diagonal (auto-spectra)
    # clean_YY[..., i, i] for each channel of y
    clean_psd = np.real(
        np.diagonal(clean_YY, axis1=-2, axis2=-1)
    ).copy()  # (n_windows, n_freq, n_y)

    # ------------------------------------------------------------------
    # Bias correction (RSI Technical Note 61)
    # R = 1 / (1 - 1.02 * n_references / n_fft_segments)
    # ------------------------------------------------------------------
    n_segments = ffts_per_diss
    R = 1.0 / (1.0 - 1.02 * n_c / n_segments)
    clean_psd *= R

    # Ensure non-negative (numerical noise can cause tiny negatives)
    np.maximum(clean_psd, 0.0, out=clean_psd)

    # Frequency vector
    freq = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # Transpose to (n_windows, n_y, n_freq) for consistency with
    # how callers store spectra per channel.
    clean_psd = np.moveaxis(clean_psd, -1, 1)

    if n_y == 1:
        clean_psd = clean_psd[:, 0, :]  # squeeze single-channel axis

    return freq, clean_psd


def despike_variables(
    ds: xr.Dataset,
    variables: tuple[str, ...],
    fs: float,
    suffix: str = "_clean",
    max_passes: int = 6,
    thresh: float = 8.0,
    smooth: float = 0.5,
    replace_sec: float = 0.04,
) -> xr.Dataset:
    """Despike specified variables on the ``t_fast`` dimension.

    For each present ``var`` this adds:
      - ``var + suffix`` (default ``"_clean"``) — the despiked signal.
      - ``<var>_despike_mask`` — boolean per-sample mask of modified samples.

    If both companions are already present (e.g., the input file was
    pre-cleaned during ``p2nc``), despiking is skipped for that variable so
    the existing cleaned signal and mask flow through unchanged.

    ``replace_sec`` is the spike replacement window in seconds; converted to
    samples here as ``int(replace_sec * fs)`` for the underlying ``despike``.
    """
    ds = ds.copy()
    n_samples = int(replace_sec * fs)

    for var in variables:
        if var not in ds:
            continue
        cleaned_name = var + suffix
        mask_name = despike_mask_name(var)
        if cleaned_name in ds and mask_name in ds:
            _log.debug("Despike skipped for %s; pre-cleaned signal present", var)
            continue
        original = ds[var].values
        cleaned, _, _, _ = despike(
            original,
            thresh=thresh,
            smooth=smooth,
            fs=fs,
            n=n_samples,
            max_passes=max_passes,
        )
        ds[cleaned_name] = ("t_fast", cleaned)
        ds[mask_name] = ("t_fast", cleaned != original)

    return ds
