"""Tests for chi (temperature variance dissipation) estimation."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pyturb._pfile import to_xarray
from pyturb.pfile import load_pfile_phys
from pyturb.processing import _write_epsilon_profile, bin_profiles
from pyturb.profile import ProfileConfig, _combine_eps_pair, process_profile
from pyturb.temperature import (
    double_pole_correction,
    estimate_chi,
    kraichnan_spectrum,
    resolved_kraichnan_fraction,
    thermal_diffusivity,
)

PFILE = Path(__file__).parent / "data" / "RIOTSHAKE_VMP142_0010_cut.p"

CHI, EPS, NU, KAPPA = 1e-8, 1e-9, 1.3e-6, 1.4e-7


class TestKraichnanSpectrum:
    def test_integral_identity(self):
        # int_0^inf psi dk = chi / (6 kappa_T)
        k_B = (EPS / (NU * KAPPA**2)) ** 0.25
        k = np.linspace(0, 5 * k_B / (2 * np.pi), 100000)
        integral = np.trapezoid(kraichnan_spectrum(k, CHI, EPS, NU, KAPPA), k)
        np.testing.assert_allclose(integral, CHI / (6 * KAPPA), rtol=1e-3)

    def test_resolved_fraction_matches_numeric_integral(self):
        k_B = (EPS / (NU * KAPPA**2)) ** 0.25
        k = np.linspace(0, 5 * k_B / (2 * np.pi), 100000)
        k_u = 0.05 * k_B / (2 * np.pi)
        sub = k <= k_u
        numeric = np.trapezoid(kraichnan_spectrum(k[sub], CHI, EPS, NU, KAPPA), k[sub])
        numeric /= CHI / (6 * KAPPA)
        closed = resolved_kraichnan_fraction(k_u, EPS, NU, KAPPA)
        np.testing.assert_allclose(closed, numeric, rtol=1e-2)

    def test_resolved_fraction_limits(self):
        k_B = (EPS / (NU * KAPPA**2)) ** 0.25
        assert resolved_kraichnan_fraction(100 * k_B, EPS, NU, KAPPA) > 0.999
        assert resolved_kraichnan_fraction(1e-4, EPS, NU, KAPPA) < 0.01


class TestThermalDiffusivity:
    def test_check_value(self):
        # Seawater at S=35, T=15, rho=1026: kappa_T ~ 1.44e-7 m2/s.
        np.testing.assert_allclose(
            thermal_diffusivity(35.0, 15.0, 1026.0), 1.44e-7, rtol=0.02
        )

    def test_increases_with_temperature(self):
        cold = thermal_diffusivity(35.0, 2.0, 1028.0)
        warm = thermal_diffusivity(35.0, 25.0, 1023.0)
        assert warm > cold


class TestDoublePoleCorrection:
    def test_unity_at_zero_frequency(self):
        corr = double_pole_correction(np.array([0.0, 10.0]), W=0.6)
        assert corr[0] == 1.0
        assert corr[1] > 1.0

    def test_slower_speed_larger_correction(self):
        f = np.array([20.0])
        assert double_pole_correction(f, W=0.3) > double_pole_correction(f, W=1.0)


class TestEstimateChi:
    def _synthetic_spectrum(self, eps, W=0.6):
        f = np.arange(1, 512) * 98.0 / 512
        psi = kraichnan_spectrum(f / W, CHI, eps, NU, KAPPA)
        P_f = psi / W / double_pole_correction(f, W)
        return f, P_f

    @pytest.mark.parametrize("eps", [1e-10, 1e-9, 1e-7])
    def test_round_trip(self, eps):
        f, P_f = self._synthetic_spectrum(eps)
        chi_hat, k_max, mad = estimate_chi(f, P_f, W=0.6, eps=eps, nu=NU, kappa_T=KAPPA)
        np.testing.assert_allclose(chi_hat, CHI, rtol=0.05)
        assert k_max > 0
        assert mad < 0.05

    def test_nan_on_bad_inputs(self):
        f, P_f = self._synthetic_spectrum(EPS)
        for kwargs in [
            dict(W=np.nan, eps=EPS),
            dict(W=0.6, eps=np.nan),
            dict(W=0.6, eps=-1e-9),
        ]:
            chi_hat, k_max, mad = estimate_chi(f, P_f, nu=NU, kappa_T=KAPPA, **kwargs)
            assert np.isnan(chi_hat)
            assert np.isnan(k_max)
            assert np.isnan(mad)

    def test_nan_spectrum(self):
        f, P_f = self._synthetic_spectrum(EPS)
        P_f[10] = np.nan
        chi_hat, _, _ = estimate_chi(f, P_f, W=0.6, eps=EPS, nu=NU, kappa_T=KAPPA)
        assert np.isnan(chi_hat)


class TestCombineEpsPair:
    def test_mean_within_factor_ten(self):
        e1 = np.array([1e-9])
        e2 = np.array([5e-9])
        qc = np.array([1], dtype="i1")
        eps, eps_qc = _combine_eps_pair(e1, e2, qc, qc)
        np.testing.assert_allclose(eps, 3e-9, rtol=1e-6)
        assert eps_qc[0] == 1

    def test_minimum_beyond_factor_ten(self):
        e1 = np.array([1e-9])
        e2 = np.array([5e-8])
        qc = np.array([1], dtype="i1")
        eps, _ = _combine_eps_pair(e1, e2, qc, qc)
        np.testing.assert_allclose(eps, 1e-9, rtol=1e-6)

    def test_single_probe_fallback(self):
        e1 = np.array([np.nan])
        e2 = np.array([2e-9])
        q1 = np.array([9], dtype="i1")
        q2 = np.array([1], dtype="i1")
        eps, eps_qc = _combine_eps_pair(e1, e2, q1, q2)
        np.testing.assert_allclose(eps, 2e-9, rtol=1e-6)
        assert eps_qc[0] == 1

    def test_both_missing(self):
        e = np.array([np.nan])
        q = np.array([9], dtype="i1")
        eps, eps_qc = _combine_eps_pair(e, e, q, q)
        assert np.isnan(eps[0])
        assert eps_qc[0] == 9


@pytest.fixture(scope="module")
def chi_eps_file(tmp_path_factory):
    """One real, processed profile with chi enabled, written to disk."""
    raw = to_xarray(load_pfile_phys(PFILE))
    config = ProfileConfig(
        shear_probes=("sh1", "sh2"),
        accel_channels=("Ax", "Ay"),
        diss_len_sec=4.0,
        fft_len_sec=1.0,
        compute_chi=True,
    )
    result = process_profile(raw.copy(deep=True), config)
    out_dir = tmp_path_factory.mktemp("eps_chi")
    out_file = out_dir / "profile_p0000.nc"
    _write_epsilon_profile(result, raw, out_file, PFILE.name, 0, config)
    return out_file


class TestChiPipeline:
    def test_chi_vars_written(self, chi_eps_file):
        written = xr.load_dataset(chi_eps_file, decode_times=False)
        for v in [
            "chi_1",
            "chi_2",
            "chi_k_max_1",
            "chi_k_max_2",
            "chi_1_fm",
            "chi_2_fm",
            "chi_1_qc",
            "chi_2_qc",
        ]:
            assert v in written, v
            assert written[v].dims == ("time",)

    def test_chi_values_plausible(self, chi_eps_file):
        written = xr.load_dataset(chi_eps_file, decode_times=False)
        chi = written["chi_1"].values
        finite = chi[np.isfinite(chi)]
        assert finite.size > 0
        assert np.all((finite > 1e-13) & (finite < 1e-3))

    def test_chi_qc_valid_flags(self, chi_eps_file):
        written = xr.load_dataset(chi_eps_file, decode_times=False)
        assert set(np.unique(written["chi_1_qc"].values)) <= {0, 1, 2, 4, 9}

    def test_on_by_default(self):
        assert ProfileConfig().compute_chi is True

    def test_disabled_with_compute_chi_false(self):
        raw = to_xarray(load_pfile_phys(PFILE))
        config = ProfileConfig(
            shear_probes=("sh1", "sh2"),
            accel_channels=("Ax", "Ay"),
            diss_len_sec=4.0,
            fft_len_sec=1.0,
            compute_chi=False,
        )
        result = process_profile(raw.copy(deep=True), config)
        assert "chi_1" not in result

    def test_binned_output_includes_combined_chi(self, chi_eps_file, tmp_path):
        binned = bin_profiles(
            [chi_eps_file],
            output_file=tmp_path / "binned.nc",
            depth_min=90.0,
            depth_max=130.0,
            bin_width=2.0,
        )
        for v in ["chi_1", "chi_2", "chi", "chi_qc", "chi_1_qc", "chi_2_qc"]:
            assert v in binned, v
        assert np.isfinite(binned["chi"].values).any()
