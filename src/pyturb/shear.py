# Functions for estimating turbulent dissipation rate from shear microstructure


import numpy as np


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
    """
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
        # Direct inertial-subrange method
        eps_fit, k_max, fit_mask = inertial_subrange_fit(
            k, phi, e_1, nu, min(150.0, k_AA)
        )
        # Use variance within fit range for unresolved correction
        e_var = (
            7.5 * nu * np.trapezoid(phi[fit_mask], k[fit_mask])
            if fit_mask.any()
            else eps_fit
        )

        eps_final = apply_unresolved_variance(e_var, k_max, nu)
        # Low-end missing variance correction
        if k[1] > 0:
            phi0 = nasmyth_spectrum(k[1:3], eps_final, nu)[0]
            eps_add = 0.25 * 7.5 * nu * k[1] * phi0
            eps_new = eps_final + eps_add
            if eps_new / eps_final > 1.1:
                eps_final = apply_unresolved_variance(
                    7.5 * nu * np.trapezoid(phi[k <= k_max], k[k <= k_max]), k_max, nu
                )
            else:
                eps_final = eps_new
        return eps_final, k_max

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
        eps_low = eps_adj + 0.25 * 7.5 * nu * k[1] * phi0
        if eps_low / eps_adj > 1.1:
            eps_low = apply_unresolved_variance(eps_low, k_range[-1], nu)
        eps_adj = eps_low

    return eps_adj, k_range[-1]
