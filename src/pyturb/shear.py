import numpy as np


def nasmyth_shear_spectrum(k: np.ndarray, eps: float, nu: float) -> np.ndarray:
    """
    Nasmyth 1D shear spectrum phi(k) (cycles/m domain).
    k   : wavenumber (cpm, cycles per metre)
    eps : dissipation (W/kg)
    nu  : kinematic viscosity (m^2/s)
    """
    eta = (nu**3 / eps) ** 0.25
    x = k * eta
    return (eps**0.75 / nu**0.25) * 8.05 * x ** (1 / 3) / (1 + (20.6 * x) ** 3.715)


def wavenumber_correction(k: np.ndarray) -> np.ndarray:
    """
    Macoun & Lueck style correction: 1 + (k/48)^2 up to 150 cpm, then 1.
    """
    corr = np.ones_like(k)
    mask = k <= 150.0
    corr[mask] = 1.0 + (k[mask] / 48.0) ** 2
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
    K: np.ndarray,
    shear_spec: np.ndarray,
    eps_init: float,
    nu: float,
    K_limit: float,
    x_isr: float = 0.02,
) -> tuple[float, float, np.ndarray]:
    """
    Fit epsilon by aligning spectrum to Nasmyth in inertial subrange.
    Returns (epsilon, K_max_used, index_mask)
    """
    eps = eps_init
    for _ in range(3):
        K_isr_max = min(K_limit, x_isr * (eps / nu**3) ** 0.25)
        fit_mask = (K > 0) & (K <= K_isr_max)
        if fit_mask.sum() < 8:
            break
        model = nasmyth_shear_spectrum(K[fit_mask], eps, nu)
        err = np.mean(np.log10(shear_spec[fit_mask] / model))
        eps *= 10 ** (1.5 * err)  # scale factor (3/2 slope relation)
    # Remove flyers (>0.5 dex) up to 20%
    K_isr_max = min(K_limit, x_isr * (eps / nu**3) ** 0.25)
    fit_mask = (K > 0) & (K <= K_isr_max)
    if fit_mask.sum() >= 8:
        model = nasmyth_shear_spectrum(K[fit_mask], eps, nu)
        err_vec = np.log10(shear_spec[fit_mask] / model)
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
                model = nasmyth_shear_spectrum(K[fit_mask], eps, nu)
                err = np.mean(np.log10(shear_spec[fit_mask] / model))
                eps *= 10 ** (1.5 * err)
    K_max = K[fit_mask][-1] if fit_mask.any() else K_limit
    return eps, K_max, fit_mask


def apply_unresolved_variance(e_var: float, K_max: float, nu: float) -> float:
    """
    Iteratively adjust epsilon for unresolved high-wavenumber variance
    following Lueck-style model.
    """
    eps = e_var
    for _ in range(12):
        x = K_max * (nu**3 / eps) ** 0.25
        x43 = x ** (4 / 3)  # convenience form used in original model
        variance_resolved = np.tanh(48 * x43) - 2.9 * x43 * np.exp(-22.3 * x43)
        variance_resolved = float(np.clip(variance_resolved, 0.05, 0.999))
        eps_new = e_var / variance_resolved
        if eps_new / eps < 1.02:
            eps = eps_new
            break
        eps = eps_new
    return eps


def estimate_epsilon_single(
    f: np.ndarray,
    P_f: np.ndarray,
    W: float,
    nu: float,
    f_AA: float = 98.0,
    e_isr_threshold: float = 1.5e-5,
    fit_order: int = 3,
) -> tuple[float, float]:
    """
    Estimate dissipation (epsilon) from one shear auto-spectrum.

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
    # --- Convert to wavenumber spectrum with correction ---
    K = f / W  # cpm
    phi = P_f * W * wavenumber_correction(K)

    # Ensure ascending, remove potential negative or zero duplicates
    sort_idx = np.argsort(K)
    K = K[sort_idx]
    phi = phi[sort_idx]

    # --- Initial variance estimate to 10 cpm ---
    a_const = 1.0774e9
    mask_10 = K <= 10
    if mask_10.sum() < 3:
        mask_10[:3] = True
    e_10 = 7.5 * nu * np.trapz(phi[mask_10], K[mask_10])
    e_1 = e_10 * np.sqrt(1 + a_const * e_10)

    K_AA = f_AA / W
    x_95 = 0.1205  # non-dimensional K for 95% variance capture
    x_isr = 0.02  # inertial subrange nondimensional limit (doubled from 0.01)

    # Branch: high or low dissipation
    if e_1 >= e_isr_threshold:
        # Direct inertial-subrange method
        eps_fit, K_max, fit_mask = inertial_subrange_fit(
            K, phi, e_1, nu, min(150.0, K_AA)
        )
        # Use variance within fit range for unresolved correction
        e_var = (
            7.5 * nu * np.trapz(phi[fit_mask], K[fit_mask])
            if fit_mask.any()
            else eps_fit
        )
        eps_final = apply_unresolved_variance(e_var, K_max, nu)
        # Low-end missing variance correction
        if K[1] > 0:
            phi0 = nasmyth_shear_spectrum(K[1:3], eps_final, nu)[0]
            eps_add = 0.25 * 7.5 * nu * K[1] * phi0
            eps_new = eps_final + eps_add
            if eps_new / eps_final > 1.1:
                eps_final = apply_unresolved_variance(
                    7.5 * nu * np.trapz(phi[K <= K_max], K[K <= K_max]), K_max, nu
                )
            else:
                eps_final = eps_new
        return float(eps_final), float(K_max)

    # --- Variance path (e_1 below threshold) ---
    # Optional refinement if enough ISR points
    isr_count = np.sum(K * (nu**3 / e_1) ** 0.25 <= x_isr)
    if isr_count >= 20:
        e_1, _, _ = inertial_subrange_fit(K, phi, e_1, nu, min(150.0, K_AA))

    # Compute K_95
    K_95 = x_95 * (e_1 / nu**3) ** 0.25
    valid_mask = K <= min(K_AA, K_95)
    if valid_mask.sum() < 3:
        valid_mask[:3] = True
    K_valid = K[valid_mask]
    phi_valid = phi[valid_mask]

    # Polynomial spectral-min search (ignore K=0)
    k_nonzero = K_valid[1:]
    spec_nonzero = phi_valid[1:]
    # Guard against NaNs
    nz_mask = ~np.isnan(spec_nonzero)
    k_log = np.log10(k_nonzero[nz_mask])
    s_log = np.log10(spec_nonzero[nz_mask])

    fit_order = int(np.clip(fit_order, 3, 8))
    if k_log.size > fit_order + 2:
        coeff = np.polyfit(k_log, s_log, fit_order)
        d1 = poly_deriv(coeff, 1)
        d2 = poly_deriv(coeff, 2)
        roots = np.roots(d1)
        # Real roots only
        roots = roots[np.isreal(roots)].real

        # Minima: second derivative > 0 at root
        def second_val(xr):
            return np.polyval(d2, xr)

        roots = [r for r in roots if second_val(r) > 0 and r >= np.log10(10)]
        if roots:
            pr1 = roots[0]
        else:
            pr1 = np.log10(K_95)
    else:
        pr1 = np.log10(K_95)

    # Final upper limit selection
    log_K_AA = np.log10(K_AA) if K_AA > 0 else np.log10(K_valid[-1])
    K_limit_log = min(pr1, np.log10(K_95), log_K_AA)
    # Constrain to [log10(7), log10(150)]
    K_limit_log = np.clip(K_limit_log, np.log10(7.0), np.log10(150.0))
    K_limit = 10**K_limit_log

    Range_mask = K <= K_limit
    if Range_mask.sum() < 3:
        Range_mask[:3] = True
    # Ensure at least reaches 7 cpm
    if K[Range_mask][-1] < 7 and Range_mask.sum() < K.size:
        next_idx = np.where(~Range_mask)[0][0]
        Range_mask[next_idx] = True

    K_range = K[Range_mask]
    phi_range = phi[Range_mask]

    e_3 = 7.5 * nu * np.trapz(phi_range, K_range)

    # Unresolved variance (iterative)
    eps_adj = apply_unresolved_variance(e_3, K_range[-1], nu)

    # Missing low-wavenumber variance (half-bin extrapolation) then re-check
    if K[1] > 0:
        phi0 = nasmyth_shear_spectrum(K[1:3], eps_adj, nu)[0]
        eps_low = eps_adj + 0.25 * 7.5 * nu * K[1] * phi0
        if eps_low / eps_adj > 1.1:
            eps_low = apply_unresolved_variance(eps_low, K_range[-1], nu)
        eps_adj = eps_low

    return float(eps_adj), float(K_range[-1])
