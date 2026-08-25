# Methods for computing the dissipation rate of temperature variance (chi)

from typing import Optional

import numpy as np

from .shear import polynomial_spectral_min_search

Q_KRAICHNAN = 5.26

# TEOS-10 fixed seawater heat capacity cp0 (J kg-1 K-1).
_CP0 = 3991.86795711963

# Solves (1 + x) exp(-x) = 0.05: 95% of Kraichnan gradient variance resolved.
_X95 = 4.74386


def thermal_diffusivity(
    S: np.ndarray, T: np.ndarray, rho: np.ndarray, P: np.ndarray | float = 0.0
) -> np.ndarray:
    """Molecular thermal diffusivity of seawater, kappa_T = k / (rho * cp) [m2/s].

    Thermal conductivity k from Caldwell (1974) as given in Sharqawy et al.
    (2010); cp is the fixed TEOS-10 value cp0.

    S : practical salinity, T : degC, rho : kg/m3, P : dbar.
    """
    P_MPa = np.asarray(P, dtype=float) / 100.0
    k = 0.5715 * (1 + 0.003 * T - 1.025e-5 * T**2 + 6.53e-4 * P_MPa - 0.00029 * S)
    return k / (rho * _CP0)


def batchelor_wavenumber(
    eps: float, nu: float = 1e-6, kappa_T: float = 1.4e-7
) -> float:
    """Batchelor wavenumber (cyclic, cpm): kB = (1 / 2pi) * (eps / (nu kappa_T^2))^(1/4).

    Bogucki, Domaradzki & Yeung (1997), J. Fluid Mech. 343, 111-130.
    """
    return (1 / (2 * np.pi)) * (eps / (nu * kappa_T**2)) ** 0.25


def kraichnan_spectrum(
    k: np.ndarray,
    chi: float,
    eps: float,
    nu: float = 1e-6,
    kappa_T: float = 1.4e-7,
    q_K: float = Q_KRAICHNAN,
) -> np.ndarray:
    """Kraichnan 1-D temperature gradient spectrum psi(k) (cpm domain) [K2 m-2 cpm-1].

    psi(k) = q_K * chi / (kappa_T * kB^2) * k * exp(-sqrt(6 q_K) * k / kB)

    k, kB both cyclic (cpm) -- Bogucki, Domaradzki & Yeung (1997), eq. 11.

    k       : wavenumber (cpm)
    chi     : temperature variance dissipation (K2/s)
    eps     : TKE dissipation (W/kg)
    nu      : kinematic viscosity (m2/s)
    kappa_T : molecular thermal diffusivity (m2/s)
    q_K     : Kraichnan constant

    Satisfies the integral constraint int_0^inf psi dk = chi / (6 kappa_T).
    """
    k_B = batchelor_wavenumber(eps, nu, kappa_T)
    return q_K * chi / (kappa_T * k_B**2) * k * np.exp(-np.sqrt(6 * q_K) * k / k_B)


def resolved_kraichnan_fraction(
    k_max: float,
    eps: float,
    nu: float = 1e-6,
    kappa_T: float = 1.4e-7,
    q_K: float = Q_KRAICHNAN,
) -> float:
    """Fraction of total Kraichnan gradient variance resolved in [0, k_max] (cpm).

    Closed form of the normalized integral: 1 - (1 + x) exp(-x) with
    x = sqrt(6 q_K) * k_max / kB (k_max, kB both cyclic, cpm).
    """
    k_B = batchelor_wavenumber(eps, nu, kappa_T)
    x = np.sqrt(6 * q_K) * k_max / k_B
    return 1.0 - (1.0 + x) * np.exp(-x)


def single_pole_correction(
    f: np.ndarray, W: float, tau0: float = 0.010, speed_exp: float = -0.5
) -> np.ndarray:
    """FP07 single-pole response correction H^-2 = 1 + (2 pi f tau)^2.

    tau = tau0 * W^speed_exp (Lueck style speed dependence).
    """
    tau = tau0 * W**speed_exp
    return 1.0 + (2 * np.pi * f * tau) ** 2


def _mad_vs_kraichnan(
    k: np.ndarray,
    phi: np.ndarray,
    chi: float,
    eps: float,
    nu: float,
    kappa_T: float,
    fit_mask: np.ndarray,
) -> float:
    """Mean absolute deviation in log10 space between spectrum and Kraichnan model.

    Computed over ``fit_mask`` bins after skipping the lowest one. NaN if
    fewer than 2 bins remain or nothing valid.
    """
    fit_idx = np.where(fit_mask)[0]
    if fit_idx.size < 2:
        return float("nan")
    idx = fit_idx[1:]
    model = kraichnan_spectrum(k[idx], chi, eps, nu, kappa_T)
    spec = phi[idx]
    valid = (spec > 0) & (model > 0) & np.isfinite(spec) & np.isfinite(model)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(np.log10(spec[valid] / model[valid]))))


def _noise_crossing_k(
    k: np.ndarray, phi: np.ndarray, phi_noise: np.ndarray, smooth_win: int = 5
) -> float:
    """Smallest k (ascending) where a smoothed ``phi`` drops to/below
    ``phi_noise``; NaN if there's no crossing in range.

    Smoothing (a centered rolling median) keeps a single noisy bin from
    triggering a false early crossing -- the noise floor should cap k_max
    where the spectrum's trend meets the noise, not where one point does.
    """
    valid = (k > 0) & np.isfinite(phi) & np.isfinite(phi_noise) & (phi_noise > 0)
    if valid.sum() < smooth_win:
        return float("nan")
    order = np.argsort(k[valid])
    k_v = k[valid][order]
    phi_v = phi[valid][order]
    noise_v = phi_noise[valid][order]

    if smooth_win >= 3 and smooth_win % 2 == 1 and phi_v.size >= smooth_win:
        pad = smooth_win // 2
        padded = np.pad(phi_v, pad, mode="edge")
        phi_v = np.array(
            [np.median(padded[i : i + smooth_win]) for i in range(phi_v.size)]
        )

    ratio = phi_v / noise_v
    below = ratio <= 1.0
    if not below.any():
        return float("nan")
    idx = int(np.argmax(below))
    if idx == 0 or ratio[idx - 1] <= ratio[idx]:
        return float(k_v[idx])

    # log-log interpolation between the bracketing points for a smoother estimate
    logr1, logr2 = np.log(ratio[idx - 1]), np.log(ratio[idx])
    frac = logr1 / (logr1 - logr2)
    log_k1, log_k2 = np.log(k_v[idx - 1]), np.log(k_v[idx])
    return float(np.exp(log_k1 + frac * (log_k2 - log_k1)))


def estimate_chi(
    f: np.ndarray,
    P_f: np.ndarray,
    W: float,
    eps: float,
    nu: float = 1e-6,
    kappa_T: float = 1.4e-7,
    f_AA: float = 98.0,
    fit_order: int = 3,
    phi_noise: Optional[np.ndarray] = None,
) -> tuple[float, float, float]:
    """Estimate chi from one temperature gradient spectrum with epsilon known.

    Integrates the observed spectrum over the resolved wavenumber band and
    corrects for unresolved variance using the Kraichnan spectrum shape set
    by eps.

    Inputs
    ------
    f    : frequency vector (Hz)
    P_f  : temperature gradient auto-spectrum ((K/m)^2 / Hz), already
        corrected for the FP07 single-pole frequency response (see
        :func:`single_pole_correction`) -- as saved to ``S_gradT1``/
        ``S_gradT2`` by :mod:`pyturb.profile`.
    W    : mean speed (m/s)
    eps  : TKE dissipation rate (W/kg), e.g. from the shear probes
    nu   : kinematic viscosity (m^2/s)
    kappa_T : molecular thermal diffusivity (m^2/s)
    f_AA : anti-alias cutoff (Hz)
    fit_order : polynomial order for the spectral-minimum search
    phi_noise : predicted electronic noise floor phi(k) [K2 m-2 cpm-1], same
        length/domain as ``f`` (e.g. from
        :func:`pyturb.noise.thermistor_noise_phi`). If given, k_max is also
        capped at the wavenumber where the (smoothed) observed spectrum
        first drops to/below this curve -- the spectral-minimum search
        alone can land past that point, integrating pure noise into chi.
        Optional; skipped if None.

    Returns
    -------
    chi : float
        Temperature variance dissipation rate (K^2/s). NaN if inputs unusable.
    k_max_used : float
        Upper wavenumber of the integration band (cpm).
    mad : float
        Mean absolute deviation in log10 of the observed spectrum from the
        Kraichnan model over the integration band.
    """
    finite_inputs = (
        np.isfinite(W)
        and np.isfinite(eps)
        and np.isfinite(nu)
        and np.isfinite(kappa_T)
        and eps > 0
        and np.all(np.isfinite(P_f))
    )
    if not finite_inputs:
        return float("nan"), float("nan"), float("nan")

    k = f / W
    phi = P_f * W

    k_B = batchelor_wavenumber(eps, nu, kappa_T)
    k_95 = _X95 * k_B / np.sqrt(6 * Q_KRAICHNAN)
    k_AA = f_AA / W

    valid_mask = k <= min(k_AA, k_95)
    if valid_mask.sum() < 3:
        valid_mask[:3] = True

    try:
        pr1 = polynomial_spectral_min_search(k[valid_mask], phi[valid_mask], fit_order)
    except RuntimeError:
        pr1 = np.log10(k_95)

    log_limits = [pr1, np.log10(k_95), np.log10(k_AA)]
    if phi_noise is not None:
        k_noise = _noise_crossing_k(k, phi, phi_noise)
        if np.isfinite(k_noise):
            log_limits.append(np.log10(k_noise))

    k_limit = 10 ** min(log_limits)

    range_mask = k <= k_limit
    if range_mask.sum() < 3:
        range_mask[:3] = True

    k_range = k[range_mask]
    chi_resolved = 6 * kappa_T * np.trapezoid(phi[range_mask], k_range)

    frac = np.clip(
        resolved_kraichnan_fraction(k_range[-1], eps, nu, kappa_T), 0.05, 1.0
    )
    chi = chi_resolved / frac

    mad = _mad_vs_kraichnan(k, phi, chi, eps, nu, kappa_T, range_mask & (k > 0))
    return chi, k_range[-1], mad
