# Methods for computing the dissipation rate of temperature variance (chi)

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


def kraichnan_spectrum(
    k: np.ndarray,
    chi: float,
    eps: float,
    nu: float = 1e-6,
    kappa_T: float = 1.4e-7,
    q_K: float = Q_KRAICHNAN,
) -> np.ndarray:
    """Kraichnan 1-D temperature gradient spectrum psi(k) (cpm domain) [K2 m-2 cpm-1].

    k       : wavenumber (cpm)
    chi     : temperature variance dissipation (K2/s)
    eps     : TKE dissipation (W/kg)
    nu      : kinematic viscosity (m2/s)
    kappa_T : molecular thermal diffusivity (m2/s)
    q_K     : Kraichnan constant

    Satisfies the integral constraint int_0^inf psi dk = chi / (6 kappa_T).
    """
    k_B = (eps / (nu * kappa_T**2)) ** 0.25  # Batchelor wavenumber (rad/m)
    k_rad = 2 * np.pi * k
    return (
        2
        * np.pi
        * q_K
        * chi
        * np.sqrt(nu / eps)
        * k_rad
        * np.exp(-np.sqrt(6 * q_K) * k_rad / k_B)
    )


def resolved_kraichnan_fraction(
    k_max: float,
    eps: float,
    nu: float = 1e-6,
    kappa_T: float = 1.4e-7,
    q_K: float = Q_KRAICHNAN,
) -> float:
    """Fraction of total Kraichnan gradient variance resolved in [0, k_max] (cpm).

    Closed form of the normalized integral: 1 - (1 + a k) exp(-a k) with
    a = sqrt(6 q_K) / k_B.
    """
    k_B = (eps / (nu * kappa_T**2)) ** 0.25
    x = np.sqrt(6 * q_K) * 2 * np.pi * k_max / k_B
    return 1.0 - (1.0 + x) * np.exp(-x)


def double_pole_correction(
    f: np.ndarray, W: float, tau0: float = 0.010, speed_exp: float = -0.5
) -> np.ndarray:
    """FP07 double-pole response correction H^-2 = (1 + (2 pi f tau)^2)^2.

    tau = tau0 * W^speed_exp (Vachon & Lueck style speed dependence).
    """
    tau = tau0 * W**speed_exp
    return (1.0 + (2 * np.pi * f * tau) ** 2) ** 2


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


def estimate_chi(
    f: np.ndarray,
    P_f: np.ndarray,
    W: float,
    eps: float,
    nu: float = 1e-6,
    kappa_T: float = 1.4e-7,
    f_AA: float = 98.0,
    tau0: float = 0.010,
    speed_exp: float = -0.5,
    fit_order: int = 3,
) -> tuple[float, float, float]:
    """Estimate chi from one temperature gradient spectrum with epsilon known.

    Integrates the response-corrected observed spectrum over the resolved
    wavenumber band and corrects for unresolved variance using the Kraichnan
    spectrum shape set by eps.

    Inputs
    ------
    f    : frequency vector (Hz)
    P_f  : temperature gradient auto-spectrum ((K/m)^2 / Hz)
    W    : mean speed (m/s)
    eps  : TKE dissipation rate (W/kg), e.g. from the shear probes
    nu   : kinematic viscosity (m^2/s)
    kappa_T : molecular thermal diffusivity (m^2/s)
    f_AA : anti-alias cutoff (Hz)
    tau0, speed_exp : FP07 response parameters (see double_pole_correction)
    fit_order : polynomial order for the spectral-minimum search

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
    phi = P_f * W * double_pole_correction(f, W, tau0, speed_exp)

    k_B = (eps / (nu * kappa_T**2)) ** 0.25
    k_95 = _X95 * k_B / (np.sqrt(6 * Q_KRAICHNAN) * 2 * np.pi)
    k_AA = f_AA / W

    valid_mask = k <= min(k_AA, k_95)
    if valid_mask.sum() < 3:
        valid_mask[:3] = True

    try:
        pr1 = polynomial_spectral_min_search(k[valid_mask], phi[valid_mask], fit_order)
    except RuntimeError:
        pr1 = np.log10(k_95)

    k_limit = 10 ** min(pr1, np.log10(k_95), np.log10(k_AA))

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
