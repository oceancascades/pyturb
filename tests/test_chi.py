"""Tests for chi (temperature variance dissipation) estimation."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pyturb._pfile import to_xarray
from pyturb.noise import _channel_calibration_params
from pyturb.pfile import load_pfile_phys
from pyturb.processing import _write_epsilon_profile, bin_profiles
from pyturb.profile import ProfileConfig, _combine_eps_pair, process_profile
from pyturb.temperature import (
    _noise_crossing_k,
    batchelor_wavenumber,
    estimate_chi,
    kraichnan_spectrum,
    resolved_kraichnan_fraction,
    single_pole_correction,
    thermal_diffusivity,
)

PFILE = Path(__file__).parent / "data" / "RIOTSHAKE_VMP142_0010_cut.p"

CHI, EPS, NU, KAPPA = 1e-8, 1e-9, 1.3e-6, 1.4e-7


class TestKraichnanSpectrum:
    def test_integral_identity(self):
        # int_0^inf psi dk = chi / (6 kappa_T)
        k_B = batchelor_wavenumber(EPS, NU, KAPPA)
        k = np.linspace(0, 5 * k_B, 100000)
        integral = np.trapezoid(kraichnan_spectrum(k, CHI, EPS, NU, KAPPA), k)
        np.testing.assert_allclose(integral, CHI / (6 * KAPPA), rtol=1e-3)

    def test_resolved_fraction_matches_numeric_integral(self):
        k_B = batchelor_wavenumber(EPS, NU, KAPPA)
        k = np.linspace(0, 5 * k_B, 100000)
        k_u = 0.05 * k_B
        sub = k <= k_u
        numeric = np.trapezoid(kraichnan_spectrum(k[sub], CHI, EPS, NU, KAPPA), k[sub])
        numeric /= CHI / (6 * KAPPA)
        closed = resolved_kraichnan_fraction(k_u, EPS, NU, KAPPA)
        np.testing.assert_allclose(closed, numeric, rtol=1e-2)

    def test_resolved_fraction_limits(self):
        k_B = batchelor_wavenumber(EPS, NU, KAPPA)
        assert resolved_kraichnan_fraction(100 * k_B, EPS, NU, KAPPA) > 0.999
        assert resolved_kraichnan_fraction(1e-4, EPS, NU, KAPPA) < 0.01

    def test_matches_bogucki_1997_cyclic_form(self):
        # Cross-checked against an independent implementation (mousebrains'
        # odas_tpw.chi.batchelor.kraichnan_grad/batchelor_kB), which writes
        # the equation directly in the cyclic-wavenumber form given by
        # Bogucki, Domaradzki & Yeung (1997) eq. 11, with no explicit 2*pi
        # factors -- unlike this module's previous radian-first derivation
        # (algebraically equivalent, but not directly comparable to the
        # paper). Reference values below are that implementation's output.
        k = np.array([0.1, 1.0, 5.0, 20.0, 50.0, 100.0])
        k_B = batchelor_wavenumber(EPS, NU, KAPPA)
        np.testing.assert_allclose(k_B, 70.83864994288155, rtol=1e-12)

        expected = np.array(
            [
                7.428019e-06,
                6.916329e-05,
                2.518138e-04,
                3.065606e-04,
                7.099214e-05,
                2.692548e-06,
            ]
        )
        np.testing.assert_allclose(
            kraichnan_spectrum(k, CHI, EPS, NU, KAPPA), expected, rtol=1e-5
        )


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


class TestSinglePoleCorrection:
    def test_unity_at_zero_frequency(self):
        corr = single_pole_correction(np.array([0.0, 10.0]), W=0.6)
        assert corr[0] == 1.0
        assert corr[1] > 1.0

    def test_slower_speed_larger_correction(self):
        f = np.array([20.0])
        assert single_pole_correction(f, W=0.3) > single_pole_correction(f, W=1.0)


class TestEstimateChi:
    def _synthetic_spectrum(self, eps, W=0.6):
        # estimate_chi now expects an already response-corrected spectrum
        # (see pyturb.profile, which applies single_pole_correction once at
        # S_gradT1/S_gradT2 computation time), so this is the true spectrum
        # with no additional attenuation applied.
        f = np.arange(1, 512) * 98.0 / 512
        psi = kraichnan_spectrum(f / W, CHI, eps, NU, KAPPA)
        P_f = psi / W
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


class TestNoiseFloorCap:
    """estimate_chi's optional phi_noise cap -- added because the
    polynomial spectral-minimum search alone can land past where the
    spectrum is actually noise-dominated (seen investigating real and
    synthetic Kraichnan+noise spectra in this session); this caps k_max at
    the wavenumber where the (smoothed) signal first drops to/below a
    supplied noise floor, in addition to the existing k_95/k_AA/polyfit
    limits.
    """

    def test_noise_crossing_k_finds_interpolated_crossing(self):
        k = np.linspace(1, 100, 200)
        phi = 1e-4 / k  # monotonically decreasing "signal"
        noise = np.full_like(k, 1e-6)  # flat noise floor; true crossing k=100
        k_cross = _noise_crossing_k(k, phi, noise)
        assert 90 < k_cross <= 100 + 1e-6

    def test_noise_crossing_k_nan_when_no_crossing(self):
        k = np.linspace(1, 100, 200)
        phi = np.full_like(k, 1e-3)  # always well above the noise floor
        noise = np.full_like(k, 1e-6)
        assert np.isnan(_noise_crossing_k(k, phi, noise))

    def test_caps_k_max_when_noise_crosses_early(self):
        # A noise floor added to a clean Kraichnan spectrum can fool the
        # polynomial spectral-minimum search into extending k_max well
        # past where the signal is actually resolved. phi_noise must not
        # let k_max exceed roughly where signal meets noise, and the
        # resulting chi should be no worse (usually better) than uncapped.
        W = 0.6
        f = np.arange(1, 512) * 98.0 / 512
        k = f / W
        phi_signal = kraichnan_spectrum(k, CHI, EPS, NU, KAPPA)
        phi_noise = 1e-7 * (k / 10) ** 1.5  # steadily rising, RSI-electronics-like
        rng = np.random.default_rng(1)
        realization = rng.chisquare(13.3, size=k.shape) / 13.3
        P_f = (phi_signal + phi_noise) * realization / W

        chi_uncapped, k_max_uncapped, _ = estimate_chi(
            f, P_f, W=W, eps=EPS, nu=NU, kappa_T=KAPPA
        )
        chi_capped, k_max_capped, _ = estimate_chi(
            f, P_f, W=W, eps=EPS, nu=NU, kappa_T=KAPPA, phi_noise=phi_noise
        )
        assert k_max_capped <= k_max_uncapped
        assert (
            abs(np.log10(chi_capped / CHI)) <= abs(np.log10(chi_uncapped / CHI)) + 0.05
        )

    def test_no_effect_when_signal_always_above_noise(self):
        f, P_f = self._synthetic_spectrum(EPS)
        phi_noise = np.full_like(f, 1e-30)  # negligible everywhere
        chi_a, k_max_a, _ = estimate_chi(f, P_f, W=0.6, eps=EPS, nu=NU, kappa_T=KAPPA)
        chi_b, k_max_b, _ = estimate_chi(
            f, P_f, W=0.6, eps=EPS, nu=NU, kappa_T=KAPPA, phi_noise=phi_noise
        )
        np.testing.assert_allclose(k_max_a, k_max_b)
        np.testing.assert_allclose(chi_a, chi_b)

    def _synthetic_spectrum(self, eps, W=0.6):
        f = np.arange(1, 512) * 98.0 / 512
        psi = kraichnan_spectrum(f / W, CHI, eps, NU, KAPPA)
        P_f = psi / W
        return f, P_f


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

    def test_noise_floor_cap_is_exercised_and_sane(self, chi_eps_file):
        # This real PFILE does carry cal_* attrs, so process_profile's
        # noise-floor cap (see _attach_chi/thermistor_noise_phi) is
        # actually engaged for this fixture, not silently skipped -- and
        # k_max should never exceed the anti-alias cutoff it's capped by
        # regardless (f_AA=98 Hz default / W).
        raw = to_xarray(load_pfile_phys(PFILE))
        assert _channel_calibration_params(raw, "T1")
        assert _channel_calibration_params(raw, "T2")

        written = xr.load_dataset(chi_eps_file, decode_times=False)
        W = written["W"].values
        for probe_num in (1, 2):
            k_max = written[f"chi_k_max_{probe_num}"].values
            finite = np.isfinite(k_max) & np.isfinite(W) & (W > 0)
            assert finite.any()
            assert np.all(k_max[finite] <= 98.0 / W[finite] + 1e-6)

    def test_gradT_spectra_are_response_corrected_with_comment(self, chi_eps_file):
        # S_gradT1/S_gradT2 must be the response-corrected spectra (chi's
        # actual input), with metadata documenting the correction -- not
        # the raw, uncorrected Welch PSD.
        written = xr.load_dataset(chi_eps_file, decode_times=False)
        for v in ["S_gradT1", "S_gradT2"]:
            assert v in written, v
            comment = written[v].attrs.get("comment", "")
            assert "single-pole" in comment.lower()
            assert "fp07_tau0" in comment
            assert (
                written[v].attrs.get("long_name")
                == f"Power spectral density of {v[2:]}"
            )

    def test_shear_spectra_are_response_corrected_with_comment(self, chi_eps_file):
        # S_sh1/S_sh2 must be the response-corrected spectra (epsilon's
        # actual input), with metadata documenting the correction.
        written = xr.load_dataset(chi_eps_file, decode_times=False)
        for v in ["S_sh1", "S_sh2"]:
            assert v in written, v
            comment = written[v].attrs.get("comment", "")
            assert "single-pole" in comment.lower()
            assert "48" in comment
            assert (
                written[v].attrs.get("long_name")
                == f"Power spectral density of {v[2:]}"
            )

    def test_kappa_t_written(self, chi_eps_file):
        written = xr.load_dataset(chi_eps_file, decode_times=False)
        assert "kappa_T" in written
        assert written["kappa_T"].dims == ("time",)
        vals = written["kappa_T"].values
        assert np.all((vals > 1e-7) & (vals < 2e-7))

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
