# Functions for estimating turbulent dissipation rate from shear microstructure

import logging

import numpy as np
import scipy.signal as sig
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)


def nasmyth_spectrum(k: np.ndarray, eps: float, nu: float = 1e-6) -> np.ndarray:
    """
    Nasmyth 1D shear spectrum phi(k) (cycles/m domain).
    k   : wavenumber (cpm, cycles per metre)
    eps : dissipation (W/kg)
    nu  : kinematic viscosity (m^2/s)
    """
    eta = (nu**3 / eps) ** 0.25
    x = k * eta
    return (eps**0.75 / nu**0.25) * 8.05 * x**0.33333333 / (1 + (20.6 * x) ** 3.715)


def integrated_nasmyth(k_max: float, eps: float, nu: float = 1e-6) -> float:
    r"""
    Integrated Nasmyth spectrum [W / kg]
    k_max   : wavenumber (cpm, cycles per metre)
    eps : dissipation (W/kg)
    nu  : kinematic viscosity (m^2/s)

    The equation from RSI Technical Note 28 (2016):
    15/2 \int_0^x spectrum = tanh(48 x^(4/3)) - 2.9 x^(4/3) exp(-22.3 x^(4/3))
    where x = k_max * (nu^3 / eps)^(1/4) is the non-dimensional wavenumber maximum.
    """
    x = k_max * (nu**3 / eps) ** 0.25
    x43 = x**1.3333333
    return np.tanh(48 * x43) - 2.9 * x43 * np.exp(-22.3 * x43)


def single_pole_correction(
    k: np.ndarray, k0: float = 48.0, k_max: float = 150.0
) -> np.ndarray:
    """
    Single pole Macoun & Lueck style correction: 1 + (k/48)^2 up to k_max cpm, then 1.

    Note that this is the inverse of the single pole transfer function:
    H^2 = 1 / (1 + (k/k0)^2)
    so this function is equivalent to multiplying by H^-2.
    """
    corr = np.ones_like(k)
    mask = k <= k_max
    corr[mask] = 1.0 + (k[mask] / k0) ** 2
    return corr


def poly_deriv(coeff: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Derivative of a polynomial with coefficients in descending powers.
    """
    c = coeff.copy()
    n = len(c) - 1
    for _ in range(order):
        c = np.array([c[i] * (n - i) for i in range(len(c) - 1)], dtype=float)
        n -= 1
        if n < 0:
            return np.array([0.0])
    return c


def inertial_subrange_fit(
    k: np.ndarray,
    phi: np.ndarray,
    eps_init: float,
    nu: float,
    k_limit: float,
    x_isr: float = 0.02,
) -> tuple[float, float, np.ndarray]:
    """
    Fit epsilon by aligning spectrum to Nasmyth in inertial subrange.
    Returns (epsilon, K_max_used, index_mask)
    """

    eps = eps_init

    for _ in range(3):
        K_isr_max = min(k_limit, x_isr * (eps / nu**3) ** 0.25)
        fit_mask = (k > 0) & (k <= K_isr_max)
        if fit_mask.sum() < 8:
            break

        model = nasmyth_spectrum(k[fit_mask], eps, nu)
        err = np.mean(np.log10(phi[fit_mask] / model))
        eps *= 10 ** (1.5 * err)  # scale factor (3/2 slope relation)

    # Remove flyers (>0.5 dex) up to 20%
    K_isr_max = min(k_limit, x_isr * (eps / nu**3) ** 0.25)
    fit_mask = (k > 0) & (k <= K_isr_max)

    if fit_mask.sum() >= 8:
        model = nasmyth_spectrum(k[fit_mask], eps, nu)
        err_vec = np.log10(phi[fit_mask] / model)
        bad = np.where(np.abs(err_vec) > 0.5)[0]

        if bad.size:
            # allow removing up to 20%
            keep = np.ones(fit_mask.sum(), dtype=bool)
            bad_sorted = bad[np.argsort(-np.abs(err_vec[bad]))]
            bad_sorted = bad_sorted[: int(0.2 * fit_mask.sum())]
            keep[bad_sorted] = False
            kept_idx = np.where(fit_mask)[0][keep]
            fit_mask[:] = False
            fit_mask[kept_idx] = True
            # Refit (2 iterations)
            for _ in range(2):
                model = nasmyth_spectrum(k[fit_mask], eps, nu)
                err = np.mean(np.log10(phi[fit_mask] / model))
                eps *= 10 ** (1.5 * err)

    k_max = k[fit_mask][-1] if fit_mask.any() else k_limit

    return eps, k_max, fit_mask


def apply_unresolved_variance(e_var: float, k_max: float, nu: float = 1e-6) -> float:
    """
    Iteratively adjust epsilon for unresolved high-wavenumber variance
    following Lueck-style model.
    """
    eps = e_var
    for _ in range(12):
        variance_resolved = np.clip(integrated_nasmyth(k_max, eps, nu), 0.05, 0.999)
        eps_new = e_var / variance_resolved
        if eps_new / eps < 1.02:
            eps = eps_new
            break
        eps = eps_new
    return eps


def polynomial_spectral_min_search(k: np.ndarray, phi: np.ndarray, fit_order: int = 3):
    if fit_order < 3 or fit_order > 8:
        raise ValueError("fit_order must be between 3 and 8")

    # Exclude zero wavenumber if present
    i = 1 if np.isclose(k[0], 0) else 0

    k_log = np.log10(k[i:])
    s_log = np.log10(phi[i:])

    if k_log.size > fit_order + 2:
        coeff = np.polyfit(k_log, s_log, fit_order)
        d1 = poly_deriv(coeff, 1)
        d2 = poly_deriv(coeff, 2)
        roots = np.roots(d1)

        roots = roots[np.isreal(roots)].real  # Real roots only

        roots = [r for r in roots if np.polyval(d2, r) > 0 and r >= np.log10(10)]
        if roots:
            return roots[0]
        else:
            raise RuntimeError("No valid spectral minimum found")
    else:
        raise RuntimeError("Not enough points for polynomial fit")


def estimate_epsilon(
    f: np.ndarray,
    P_f: np.ndarray,
    W: float,
    nu: float = 1e-6,
    f_AA: float = 98.0,
    e_isr_threshold: float = 1.5e-5,
    fit_order: int = 3,
) -> tuple[float, float]:
    """
    Estimate epsilon from one shear spectrum.

    Inputs
    f    : frequency vector (Hz), length N
    P_f  : shear auto-spectrum (shear^2 / Hz), length N
    W    : mean speed (m/s)
    nu   : kinematic viscosity (m^2/s)
    f_AA : anti-alias cutoff (Hz)
    e_isr_threshold : threshold to switch to inertial-subrange fitting
    fit_order : polynomial order (3–8) for spectral-min search

    Returns
    epsilon, K_max_used
    """

    # Converto to wavenumber domain
    k = f / W  # cpm
    phi = P_f * W * single_pole_correction(k)

    # Make a first guess of epsilon from low-wavenumber variance by integrating
    # data to 10 cpm and applying a non-linear correction.
    a_const = 1.0774e9  # from RSI technical note 28, 2016, page 14.
    mask_10 = k <= 10
    if mask_10.sum() < 3:
        mask_10[:3] = True
    e_10 = 7.5 * nu * np.trapezoid(phi[mask_10], k[mask_10])
    e_1 = e_10 * np.sqrt(1 + a_const * e_10)

    k_AA = f_AA / W
    x_95 = 0.1205  # non-dimensional k for 95% variance capture
    x_isr = 0.02  # inertial subrange nondimensional limit

    # If the first-guess epsilon is above the threshold fit to the inertial subrange directly.
    if e_1 >= e_isr_threshold:
        # Direct inertial-subrange method - return fit result directly
        # (no additional variance corrections; ISR fit is self-consistent)
        eps_fit, k_max, _ = inertial_subrange_fit(k, phi, e_1, nu, min(150.0, k_AA))
        return eps_fit, k_max

    # If the first guess is low we undertake a more thorough fitting procedure
    # If there are enough points in the inertial subrange with refine the initial guess
    isr_count = np.sum(k * (nu**3 / e_1) ** 0.25 <= x_isr)
    if isr_count >= 20:
        e_1, _, _ = inertial_subrange_fit(k, phi, e_1, nu, min(150.0, k_AA))

    # Calculate the wavenumber at which 95% of variance is captured
    k_95 = x_95 * (e_1 / nu**3) ** 0.25
    valid_mask = k <= min(k_AA, k_95)
    if valid_mask.sum() < 3:
        valid_mask[:3] = True
    k_valid = k[valid_mask]
    phi_valid = phi[valid_mask]

    # Polynomial spectral minimum search
    try:
        pr1 = polynomial_spectral_min_search(k_valid, phi_valid, fit_order)
    except RuntimeError:
        pr1 = np.log10(k_95)

    # Final upper limit selection
    log_k_AA = np.log10(k_AA) if k_AA > 0 else np.log10(k_valid[-1])
    k_limit_log = min(pr1, np.log10(k_95), log_k_AA)
    # Constrain to [log10(7), log10(150)]
    k_limit_log = np.clip(k_limit_log, np.log10(7.0), np.log10(150.0))
    k_limit = 10**k_limit_log

    range_mask = k <= k_limit
    if range_mask.sum() < 3:
        range_mask[:3] = True
    # Ensure at least reaches 7 cpm
    if k[range_mask][-1] < 7 and range_mask.sum() < k.size:
        next_idx = np.where(~range_mask)[0][0]
        range_mask[next_idx] = True

    k_range = k[range_mask]
    phi_range = phi[range_mask]

    e_3 = 7.5 * nu * np.trapezoid(phi_range, k_range)

    # Unresolved variance
    eps_adj = apply_unresolved_variance(e_3, k_range[-1], nu)

    # Missing low-wavenumber variance (half-bin extrapolation) then re-check
    if k[1] > 0:
        phi0 = nasmyth_spectrum(k[1:3], eps_adj, nu)[0]
        low_k_correction = 0.25 * 7.5 * nu * k[1] * phi0
        eps_low = eps_adj + low_k_correction
        if eps_low / eps_adj > 1.1:
            # Significant low-k correction: add to original variance and re-iterate
            e_3_corrected = e_3 + low_k_correction
            eps_adj = apply_unresolved_variance(e_3_corrected, k_range[-1], nu)
        else:
            eps_adj = eps_low

    return eps_adj, k_range[-1]


def clean_shear_spec(
    shear: np.ndarray,
    accel: np.ndarray,
    n_fft: int,
    fs: float,
    n_diss: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Remove acceleration-coherent contamination from shear spectra (Goodman method).

    Uses the Goodman coherent-noise removal algorithm to subtract vibration
    signals (measured by accelerometers) that are coherent with the shear
    probe signals.  Operates entirely in the spectral domain.

    Parameters
    ----------
    shear : ndarray, shape (N,) or (N, n_probes)
        Shear probe time series.  Each column is one probe.
    accel : ndarray, shape (N, n_accel)
        Accelerometer time series.  Each column is one component (e.g. Ax, Ay, Az).
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
    freq : ndarray, shape (n_fft//2 + 1,)
        Frequency vector (Hz).
    clean_psd : ndarray, shape (n_windows, n_probes, n_fft//2 + 1)
        Cleaned shear auto-spectra averaged over dissipation windows.
        If there is only one shear probe the probe axis is squeezed out.
    """
    shear = np.asarray(shear, dtype=np.float64)
    accel = np.asarray(accel, dtype=np.float64)

    if shear.ndim == 1:
        shear = shear[:, np.newaxis]
    if accel.ndim == 1:
        accel = accel[:, np.newaxis]
    if shear.shape[0] != accel.shape[0]:
        raise ValueError("shear and accel must have the same number of rows")

    n_probes = shear.shape[1]
    n_accel = accel.shape[1]
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
    shear_segs = np.array(
        sliding_window_view(shear, n_fft, axis=0)[::step], dtype=np.float64
    ).transpose(0, 2, 1)
    accel_segs = np.array(
        sliding_window_view(accel, n_fft, axis=0)[::step], dtype=np.float64
    ).transpose(0, 2, 1)

    # Apply window: (n_seg, n_fft, n_ch) * (n_fft,)
    shear_segs *= win[np.newaxis, :, np.newaxis]
    accel_segs *= win[np.newaxis, :, np.newaxis]

    # Linear detrend each segment (matches MATLAB 'linear' method)
    x = np.linspace(0.0, 1.0, n_fft, dtype=np.float64)
    xm = x - x.mean()
    xm_ss = np.dot(xm, xm)  # sum of squares
    for segs in (shear_segs, accel_segs):
        # segs: (n_seg, n_fft, n_ch)
        mean_y = segs.mean(axis=1, keepdims=True)
        slope = np.einsum("stc,t->sc", segs, xm) / xm_ss
        segs -= mean_y + slope[:, np.newaxis, :] * xm[np.newaxis, :, np.newaxis]

    # One-sided FFT: (n_seg, n_freq, n_ch)
    U = np.fft.rfft(shear_segs, axis=1)
    A = np.fft.rfft(accel_segs, axis=1)

    # ------------------------------------------------------------------
    # Group FFT segments into dissipation windows and build ensemble-
    # averaged cross-spectral matrices.
    # ------------------------------------------------------------------
    ffts_per_diss = (n_diss - fft_overlap) // step
    n_seg_total = U.shape[0]
    n_windows = n_seg_total // ffts_per_diss

    # Trim to exact number of complete dissipation windows
    n_seg_used = n_windows * ffts_per_diss
    U = U[:n_seg_used].reshape(n_windows, ffts_per_diss, n_freq, n_probes)
    A = A[:n_seg_used].reshape(n_windows, ffts_per_diss, n_freq, n_accel)

    # Scale factor for one-sided spectrum (×2), then halve DC and Nyquist
    scale = 2.0 / norm

    # Cross-spectral matrices averaged over segments within each window
    # Axes: w=window, s=segment, f=freq, i/j=channel
    # UU: (n_windows, n_freq, n_probes, n_probes)
    UU = scale * np.einsum("wsfi,wsfj->wfij", U, U.conj()) / ffts_per_diss
    AA = scale * np.einsum("wsfi,wsfj->wfij", A, A.conj()) / ffts_per_diss
    UA = scale * np.einsum("wsfi,wsfj->wfij", U, A.conj()) / ffts_per_diss

    # Fix DC and Nyquist (should not be doubled)
    for M in (UU, AA, UA):
        M[:, 0, :, :] *= 0.5
        M[:, -1, :, :] *= 0.5

    # ------------------------------------------------------------------
    # Goodman cleaning: clean_UU = UU - UA @ inv(AA) @ UA^H
    # Solve via np.linalg.solve for numerical stability.
    # AA @ X = UA^H  =>  X = inv(AA) @ UA^H
    # Then  clean_UU = UU - UA @ X
    # ------------------------------------------------------------------
    # UA^H: (w, f, n_accel, n_probes) — conjugate transpose of last two axes
    UA_H = np.conj(np.swapaxes(UA, -2, -1))

    # Solve AA @ X = UA_H for X: (w, f, n_accel, n_probes)
    X = np.linalg.solve(AA, UA_H)

    # Correction: UA @ X  -> (w, f, n_probes, n_probes)
    correction = np.einsum("wfij,wfjk->wfik", UA, X)

    clean_UU = UU - correction

    # Take real part of the diagonal (auto-spectra)
    # clean_UU[..., i, i] for each probe
    clean_psd = np.real(
        np.diagonal(clean_UU, axis1=-2, axis2=-1)
    ).copy()  # (n_windows, n_freq, n_probes)

    # ------------------------------------------------------------------
    # Bias correction (RSI Technical Note 61)
    # R = 1 / (1 - 1.02 * n_vibration_signals / n_fft_segments)
    # ------------------------------------------------------------------
    n_segments = ffts_per_diss
    R = 1.0 / (1.0 - 1.02 * n_accel / n_segments)
    clean_psd *= R

    # Ensure non-negative (numerical noise can cause tiny negatives)
    np.maximum(clean_psd, 0.0, out=clean_psd)

    # Frequency vector
    freq = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # Transpose to (n_windows, n_probes, n_freq) for consistency with
    # how process_profile stores spectra per probe.
    clean_psd = np.moveaxis(clean_psd, -1, 1)

    if n_probes == 1:
        clean_psd = clean_psd[:, 0, :]  # squeeze probe axis

    return freq, clean_psd
