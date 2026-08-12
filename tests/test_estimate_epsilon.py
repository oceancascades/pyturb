"""Tests for K_max and eps in estimate_epsilon using synthetic Nasmyth spectra
with a realistic rising shear-channel noise floor."""

import numpy as np
import pytest

from pyturb.noise import noise_shearchannel
from pyturb.shear import estimate_epsilon, nasmyth_spectrum

NU = 1e-6
X_95 = 0.1205
X_ISR = 0.02
F_AA = 98.0
DELTA_S = 4.096 / 2**16  # 16-bit, +/- VFS/2
SENS = 0.07  # V*s/m, typical SPM shear probe


def _k_grid(W: float, n: int = 1025, f_max: float = 256.0) -> np.ndarray:
    return np.linspace(0.0, f_max, n) / W


def _k95(eps: float, nu: float = NU) -> float:
    return X_95 * (eps / nu**3) ** 0.25


def _k_isr_upper(eps: float, nu: float = NU) -> float:
    return X_ISR * (eps / nu**3) ** 0.25


def _k_peak(eps: float, nu: float = NU) -> float:
    """Approximate wavenumber of the Nasmyth peak (x ≈ 0.025)."""
    return 0.025 * (eps / nu**3) ** 0.25


def _shear_noise_phi(k: np.ndarray, W: float, sens: float = SENS) -> np.ndarray:
    """Electronic shear-channel noise PSD [(s^-1)^2 / cpm] at wavenumbers k.

    Built from noise_shearchannel (counts^2/Hz), converted to physical shear units
    via the probe sensitivity, and converted from frequency to wavenumber density.
    """
    f = np.maximum(k * W, 1e-3)
    n_counts = noise_shearchannel(f)
    return n_counts * (DELTA_S / sens) ** 2 * W


class TestKmaxCleanNasmythVarianceBranch:
    """Pure Nasmyth, variance branch: K_max should hit k_95 cap; eps recovered."""

    @pytest.mark.parametrize("eps_true", [1e-9, 1e-8, 1e-7, 1e-6])
    def test_k_max_and_eps(self, eps_true):
        W = 0.6
        k = _k_grid(W)
        phi = nasmyth_spectrum(k, eps_true, NU)
        phi = phi + phi.max() * 1e-6  # tiny floor for numerical safety
        eps_est, k_max, _ = estimate_epsilon(
            k,
            phi,
            W=W,
            nu=NU,
            is_wavenumber=True,
            apply_single_pole_correction=False,
        )
        expected = min(_k95(eps_true), F_AA / W, 150.0)
        assert 0.85 * expected <= k_max <= 1.15 * expected
        assert abs(np.log10(eps_est / eps_true)) < 0.1


class TestKmaxCleanNasmythIsrBranch:
    """Pure Nasmyth, ISR branch: K_max ≈ x_isr * k_eta; eps recovered."""

    @pytest.mark.parametrize("eps_true", [3e-5, 1e-4, 1e-3])
    def test_k_max_and_eps(self, eps_true):
        W = 0.7
        k = _k_grid(W)
        phi = nasmyth_spectrum(k, eps_true, NU)
        eps_est, k_max, _ = estimate_epsilon(
            k,
            phi,
            W=W,
            nu=NU,
            is_wavenumber=True,
            apply_single_pole_correction=False,
        )
        expected = min(_k_isr_upper(eps_true), 150.0, F_AA / W)
        assert 0.85 * expected <= k_max <= 1.05 * expected
        assert abs(np.log10(eps_est / eps_true)) < 0.15


class TestKmaxRealisticNoise:
    """Nasmyth + realistic rising electronic noise at typical SPM sensitivity."""

    @pytest.mark.parametrize("eps_true", [1e-9, 1e-8, 1e-7, 1e-6])
    def test_eps_recovery_near_perfect_at_typical_sensitivity(self, eps_true):
        W = 0.6
        k = _k_grid(W)
        phi = nasmyth_spectrum(k, eps_true, NU) + _shear_noise_phi(k, W)
        eps_est, k_max, _ = estimate_epsilon(
            k,
            phi,
            W=W,
            nu=NU,
            is_wavenumber=True,
            apply_single_pole_correction=False,
        )
        # SPM noise floor sits well below signal for eps >= 1e-9 -> near-perfect eps
        assert abs(np.log10(eps_est / eps_true)) < 0.1
        expected = min(_k95(eps_true), F_AA / W, 150.0)
        assert 0.85 * expected <= k_max <= 1.15 * expected


class TestKmaxBowlTracking:
    """With strong rising noise creating a bowl below k_95, K_max should sit
    near the bowl minimum (between k_peak and the analytic bowl k)."""

    @pytest.mark.parametrize("eps_true", [3e-9, 1e-8, 3e-8])
    def test_k_max_brackets_bowl(self, eps_true):
        W = 0.6
        sens = 1e-3
        k = _k_grid(W)
        phi_tot = nasmyth_spectrum(k, eps_true, NU) + _shear_noise_phi(k, W, sens=sens)
        _, k_max, _ = estimate_epsilon(
            k,
            phi_tot,
            W=W,
            nu=NU,
            is_wavenumber=True,
            apply_single_pole_correction=False,
        )
        # Locate true bowl minimum within [k_peak, 150]
        m = (k >= _k_peak(eps_true)) & (k <= 150.0)
        k_bowl = float(k[m][np.argmin(phi_tot[m])])
        # K_max should land past the Nasmyth peak and not far past the bowl
        assert k_max > _k_peak(eps_true)
        assert k_max <= 1.2 * k_bowl


class TestKmaxMonotonicity:
    """K_max should grow with eps under a fixed rising noise floor."""

    def test_k_max_monotonic_with_eps(self):
        W = 0.6
        eps_values = [1e-9, 1e-8, 1e-7, 1e-6]
        k = _k_grid(W)
        noise = _shear_noise_phi(k, W, sens=1e-3)
        k_max_vals = []
        for eps_true in eps_values:
            phi = nasmyth_spectrum(k, eps_true, NU) + noise
            _, k_max, _ = estimate_epsilon(
                k,
                phi,
                W=W,
                nu=NU,
                is_wavenumber=True,
                apply_single_pole_correction=False,
            )
            k_max_vals.append(k_max)
        dk = k[2] - k[1]
        diffs = np.diff(k_max_vals)
        assert np.all(diffs >= -dk), f"k_max not monotonic: {k_max_vals}"


class TestNonFiniteInputReturnsNan:
    """A dead/uncalibrated probe (all-NaN spectrum) must not crash the fit.

    Regression test: an all-NaN P_f used to reach np.roots() inside
    polynomial_spectral_min_search() with a NaN-coefficient polynomial,
    raising numpy.linalg.LinAlgError deep inside estimate_epsilon and
    aborting the whole profile instead of yielding "no estimate".
    """

    def test_all_nan_spectrum_returns_nan_without_raising(self):
        W = 0.6
        k = _k_grid(W)
        phi = np.full_like(k, np.nan)
        eps, k_max, mad = estimate_epsilon(k, phi, W=W, nu=NU, is_wavenumber=True)
        assert np.isnan(eps)
        assert np.isnan(k_max)
        assert np.isnan(mad)

    def test_partially_nan_spectrum_returns_nan_without_raising(self):
        W = 0.6
        k = _k_grid(W)
        phi = nasmyth_spectrum(k, 1e-8, NU)
        phi[10] = np.nan
        eps, k_max, mad = estimate_epsilon(k, phi, W=W, nu=NU, is_wavenumber=True)
        assert np.isnan(eps)
        assert np.isnan(k_max)
        assert np.isnan(mad)

    def test_nan_speed_returns_nan_without_raising(self):
        W = 0.6
        k = _k_grid(W)
        phi = nasmyth_spectrum(k, 1e-8, NU)
        eps, k_max, mad = estimate_epsilon(k, phi, W=np.nan, nu=NU, is_wavenumber=True)
        assert np.isnan(eps)
        assert np.isnan(k_max)
        assert np.isnan(mad)
