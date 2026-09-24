"""Tests for FP07 in-situ recalibration."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pyturb._pfile import to_xarray
from pyturb.fp07_calibration import (
    REF_BANDWIDTH_MARGIN,
    ProbeCalibrationFit,
    apply_probe_calibration,
    find_lag,
    fit_is_confident,
    fit_is_plausible,
    fit_probe_calibration,
    fit_probe_calibration_multi,
    log_r_from_counts,
    read_report,
    steinhart_hart_forward,
    steinhart_hart_regress,
    write_report,
)
from pyturb.pfile import load_pfile_phys
from pyturb.profile import ProfileConfig, prepare_profile

PFILE = Path(__file__).parent / "data" / "RIOTSHAKE_VMP142_0010_cut.p"

T_0, BETA_1, BETA_2 = 289.301, 3143.55, 250000.0
# Real T1 electronics constants (RIOTSHAKE_VMP142_0010_cut.p).
A, B, G, E_B, ADC_FS, ADC_BITS = -11.5, 0.99954, 6.0, 0.68294, 4.096, 16


def _make_fit(**overrides) -> ProbeCalibrationFit:
    """A synthetic fit with sane, confident defaults; pass e.g.
    new_T_0=713.0 or lag_corr=0.2 to make one field implausible/unconfident
    for a specific test."""
    fields = dict(
        probe="T1",
        sn="SYN1",
        instrument_sn="142",
        fit_file="synthetic",
        profile_index=0,
        reference="JAC_T",
        order=1,
        old_T_0=T_0,
        old_beta_1=BETA_1,
        old_beta_2=None,
        new_T_0=289.0,
        new_beta_1=3100.0,
        new_beta_2=None,
        lag_s=0.0,
        lag_corr=0.95,
        n_points=100,
        temperature_range_c=10.0,
        rms_diff_old_c=0.0,
        rms_diff_new_c=0.0,
        max_abs_diff_old_c=0.0,
        max_abs_diff_new_c=0.0,
        mean_bias_old_c=0.0,
        mean_bias_new_c=0.0,
        residual_std_c=0.0,
        fit_date="2026-01-01T00:00:00+00:00",
    )
    fields.update(overrides)
    return ProbeCalibrationFit(**fields)


class TestFitIsPlausible:
    """T_0/beta_1 are real physical quantities (a reference temperature, a
    positive thermistor material constant) -- a fit that puts either far
    outside a sane range indicates a poorly-conditioned regression, even
    if its own-segment lag and residual look good."""

    def test_sane_coefficients_are_plausible(self):
        assert fit_is_plausible(_make_fit(new_T_0=288.0, new_beta_1=3050.0))

    def test_t0_far_above_range_is_implausible(self):
        assert not fit_is_plausible(_make_fit(new_T_0=713.7, new_beta_1=1484.5))

    def test_t0_negative_is_implausible(self):
        assert not fit_is_plausible(_make_fit(new_T_0=-913.0, new_beta_1=-9.5))

    def test_beta1_negative_is_implausible(self):
        assert not fit_is_plausible(_make_fit(new_T_0=291.8, new_beta_1=-1165.9))

    def test_beta1_far_above_range_is_implausible(self):
        assert not fit_is_plausible(_make_fit(new_T_0=289.0, new_beta_1=22950.2))


class TestFitIsConfident:
    """fit_is_confident is the bar 'calibrate-fp07 auto' uses to accept a
    fit immediately, and what apply_probe_calibration's *_fp07_confident
    attr reflects downstream: plausible coefficients alone don't mean the
    fit is trustworthy, since a weak lag can still land in a plausible
    range."""

    def test_plausible_and_confident_lag_is_confident(self):
        assert fit_is_confident(_make_fit(lag_corr=0.9))

    def test_plausible_but_weak_lag_is_not_confident(self):
        assert not fit_is_confident(_make_fit(lag_corr=0.2))

    def test_implausible_even_with_strong_lag_is_not_confident(self):
        assert not fit_is_confident(
            _make_fit(new_T_0=713.7, new_beta_1=1484.5, lag_corr=0.92)
        )


class TestLogRFromCounts:
    def test_matches_manual_formula(self):
        counts = np.array([-5000.0, 0.0, 3000.0])
        Z = ((counts - A) / B) * (ADC_FS / 2**ADC_BITS) * 2 / (G * E_B)
        R = (1 - Z) / (1 + Z)
        expected = np.log(R)
        actual = log_r_from_counts(counts, A, B, G, E_B, ADC_FS, ADC_BITS)
        np.testing.assert_allclose(actual, expected)

    def test_never_clips(self):
        # Within the real 16-bit ADC range, but well past the |Z|>0.6 bound
        # _therm's conversion clips at -- log_r_from_counts must not clip.
        counts = np.array([-30000.0, 30000.0])
        Z = ((counts - A) / B) * (ADC_FS / 2**ADC_BITS) * 2 / (G * E_B)
        assert np.all(np.abs(Z) > 0.6)  # confirms this would have clipped
        log_R = log_r_from_counts(counts, A, B, G, E_B, ADC_FS, ADC_BITS)
        assert np.all(np.isfinite(log_R))


class TestRegression:
    def test_recovers_known_coefficients_order2(self):
        log_R = np.linspace(-0.3, 0.3, 500)
        T_ref_C = steinhart_hart_forward(log_R, T_0, BETA_1, BETA_2) - 273.15
        fit_T0, fit_b1, fit_b2, resid = steinhart_hart_regress(log_R, T_ref_C, order=2)
        np.testing.assert_allclose(fit_T0, T_0, rtol=1e-6)
        np.testing.assert_allclose(fit_b1, BETA_1, rtol=1e-6)
        np.testing.assert_allclose(fit_b2, BETA_2, rtol=1e-4)
        assert resid < 1e-6

    def test_recovers_known_coefficients_order1(self):
        log_R = np.linspace(-0.3, 0.3, 500)
        T_ref_C = steinhart_hart_forward(log_R, T_0, BETA_1) - 273.15
        fit_T0, fit_b1, fit_b2, resid = steinhart_hart_regress(log_R, T_ref_C, order=1)
        np.testing.assert_allclose(fit_T0, T_0, rtol=1e-6)
        np.testing.assert_allclose(fit_b1, BETA_1, rtol=1e-6)
        assert fit_b2 is None


class TestFindLag:
    def test_recovers_known_shift(self):
        fs = 64.0
        rng = np.random.default_rng(0)
        base = np.cumsum(rng.standard_normal(2000))
        shift = 5
        T_ref = np.roll(base, shift)
        T_ref[:shift] = base[0]
        lag_s, corr = find_lag(base, T_ref, fs)
        np.testing.assert_allclose(lag_s, -shift / fs, atol=1 / fs)
        assert corr > 0.9


def _counts_from_log_R(
    log_R: np.ndarray,
    a: float,
    b: float,
    g: float,
    e_b: float,
    adc_fs: float,
    adc_bits: int,
) -> np.ndarray:
    """Inverse of log_r_from_counts: raw counts that convert to log_R exactly."""
    R = np.exp(log_R)
    Z = (1 - R) / (1 + R)
    return a + b * Z / ((adc_fs / 2**adc_bits) * 2 / (g * e_b))


class TestZeroLagEdgeCase:
    """A lag estimate landing exactly at zero (e.g. a reference with no
    real timing offset from the probe -- more likely with a heavily
    smoothed/coarse reference) must not crash. Regression for a bug in the
    T_ref-alignment branch that only special-cased lag_samples > 0, calling
    _lag_filter(T_ref, 0) for lag_samples == 0 and indexing out of bounds
    (_lag_filter assumes a strictly positive delay)."""

    def _make_ds(self, n: int = 3000, fs: float = 64.0) -> xr.Dataset:
        rng = np.random.default_rng(0)
        t = np.arange(n) / fs
        T_true = (
            10.0 + 2.0 * np.sin(2 * np.pi * 0.05 * t) + 0.01 * rng.standard_normal(n)
        )
        # Raw counts that convert (via the real electronics/Steinhart-Hart
        # constants) to exactly T_true -- the "probe" and "reference" are
        # then identical, guaranteeing find_lag's cross-correlation peaks
        # at exactly zero.
        log_R = BETA_1 * (1 / (T_true + 273.15) - 1 / T_0)
        counts = _counts_from_log_R(log_R, A, B, G, E_B, ADC_FS, ADC_BITS)

        ds = xr.Dataset(
            {
                "T1": ("t_slow", T_true.astype("f4")),
                "T1_counts": ("t_slow", counts),
                "aux_temperature": ("t_slow", T_true),
                "W_smooth": ("t_slow", np.full(n, 0.5)),
                "P_smooth": ("t_slow", np.full(n, 20.0)),
                "fs_slow": fs,
            },
            coords={"t_slow": t},
            attrs={"instrument_sn": "142"},
        )
        ds["T1"].attrs = {
            "cal_a": str(A),
            "cal_b": str(B),
            "cal_g": str(G),
            "cal_e_b": str(E_B),
            "cal_adc_fs": str(ADC_FS),
            "cal_adc_bits": str(ADC_BITS),
            "cal_t_0": str(T_0),
            "cal_beta_1": str(BETA_1),
            "cal_sn": "SYN1",
        }
        return ds

    def test_zero_lag_is_a_reachable_find_lag_result(self):
        # Confirms lag_samples == 0 is a real, reachable case (a perfectly
        # correlated, unshifted pair of signals) -- not a hypothetical one
        # -- before the other two tests force it through the full pipeline.
        ds = self._make_ds()
        lag_s, corr = find_lag(ds["T1"].values, ds["aux_temperature"].values, 64.0)
        assert lag_s == 0.0
        assert corr > 0.99

    def test_fit_does_not_crash_on_exact_zero_lag(self, monkeypatch):
        # fit_probe_calibration's own internal thermal-response-matching
        # filter has some phase lag of its own, so it doesn't reliably land
        # exactly on lag_samples == 0 for the ds above -- pin find_lag's
        # result directly to exercise that exact branch deterministically.
        import pyturb.fp07_calibration as fp07mod

        monkeypatch.setattr(fp07mod, "find_lag", lambda *a, **k: (0.0, 0.99))
        ds = self._make_ds()
        config = ProfileConfig()
        fit = fit_probe_calibration(
            ds,
            "T1",
            config,
            fit_file="synthetic",
            profile_index=0,
            ref="aux_temperature",
            order=1,
        )
        assert fit.lag_s == 0.0
        np.testing.assert_allclose(fit.new_T_0, T_0, rtol=1e-3)
        np.testing.assert_allclose(fit.new_beta_1, BETA_1, rtol=1e-2)

    def test_fit_multi_does_not_crash_on_exact_zero_lag(self, monkeypatch):
        import pyturb.fp07_calibration as fp07mod

        monkeypatch.setattr(fp07mod, "find_lag", lambda *a, **k: (0.0, 0.99))
        ds = self._make_ds()
        config = ProfileConfig()
        fit = fit_probe_calibration_multi(
            [(0, ds)],
            "T1",
            config,
            fit_file="synthetic",
            ref="aux_temperature",
            order=1,
        )
        assert fit.lag_s == 0.0
        np.testing.assert_allclose(fit.new_T_0, T_0, rtol=1e-3)


class TestReportIO:
    def test_round_trip(self, tmp_path):
        fit = ProbeCalibrationFit(
            probe="T1",
            sn="T1592",
            instrument_sn="142",
            fit_file="x.nc",
            profile_index=0,
            reference="JAC_T",
            order=1,
            old_T_0=289.301,
            old_beta_1=3143.55,
            old_beta_2=None,
            new_T_0=290.0,
            new_beta_1=3150.0,
            new_beta_2=None,
            lag_s=-0.05,
            lag_corr=0.95,
            n_points=1000,
            temperature_range_c=10.0,
            rms_diff_old_c=1.0,
            rms_diff_new_c=0.02,
            max_abs_diff_old_c=1.2,
            max_abs_diff_new_c=0.05,
            mean_bias_old_c=0.9,
            mean_bias_new_c=0.01,
            residual_std_c=0.01,
            fit_date="2026-01-01T00:00:00+00:00",
        )
        path = tmp_path / "report.yaml"
        write_report([fit], path)
        (loaded,) = read_report(path)
        assert loaded == fit

    def test_preserves_multiple_fits_with_the_same_probe_name(self, tmp_path):
        # calibrate-fp07 auto can fit the same channel (e.g. "T1") separately
        # per instrument/SN group -- a report keyed by probe name alone would
        # silently drop all but one of them.
        base = dict(
            probe="T1",
            reference="JAC_T",
            order=1,
            old_T_0=289.301,
            old_beta_1=3143.55,
            old_beta_2=None,
            new_T_0=290.0,
            new_beta_1=3150.0,
            new_beta_2=None,
            lag_s=-0.05,
            lag_corr=0.95,
            n_points=1000,
            temperature_range_c=10.0,
            rms_diff_old_c=1.0,
            rms_diff_new_c=0.02,
            max_abs_diff_old_c=1.2,
            max_abs_diff_new_c=0.05,
            mean_bias_old_c=0.9,
            mean_bias_new_c=0.01,
            residual_std_c=0.01,
            fit_date="2026-01-01T00:00:00+00:00",
        )
        fit_a = ProbeCalibrationFit(
            sn="SNA", instrument_sn="142", fit_file="a.nc", profile_index=0, **base
        )
        fit_b = ProbeCalibrationFit(
            sn="SNB", instrument_sn="194", fit_file="b.nc", profile_index=0, **base
        )
        path = tmp_path / "report.yaml"
        write_report([fit_a, fit_b], path)
        loaded = read_report(path)
        assert len(loaded) == 2
        assert {f.instrument_sn for f in loaded} == {"142", "194"}


@pytest.fixture(scope="module")
def prepared_profile():
    raw = to_xarray(load_pfile_phys(PFILE))
    config = ProfileConfig()
    return prepare_profile(raw, config), config


class TestFitProbeCalibrationRealData:
    def test_fits_without_error(self, prepared_profile):
        ds, config = prepared_profile
        fit = fit_probe_calibration(
            ds, "T1", config, fit_file=PFILE.name, profile_index=0, order=1
        )
        assert fit.probe == "T1"
        assert fit.sn == "T1592"
        assert fit.instrument_sn
        assert fit.n_points == ds.sizes["t_slow"]
        assert np.isfinite(fit.new_T_0)
        assert np.isfinite(fit.new_beta_1)

    def test_improves_agreement_with_reference(self, prepared_profile):
        ds, config = prepared_profile
        fit = fit_probe_calibration(
            ds, "T1", config, fit_file=PFILE.name, profile_index=0, order=1
        )
        assert fit.rms_diff_new_c <= fit.rms_diff_old_c

    def test_raises_without_raw_counts(self, prepared_profile):
        ds, config = prepared_profile
        legacy = ds.drop_vars(["T1_counts"])
        with pytest.raises(ValueError, match="T1_counts"):
            fit_probe_calibration(
                legacy, "T1", config, fit_file=PFILE.name, profile_index=0, order=1
            )


@pytest.fixture(scope="module")
def two_profile_list(prepared_profile):
    """Split the single real prepared profile into two halves and treat
    them as if they were two separate profiles -- real data, just chunked,
    enough to exercise fit_probe_calibration_multi's aggregation (median
    lag, concatenated regression) without needing a p-file with multiple
    naturally-detected profiles."""
    ds, config = prepared_profile
    n_slow = ds.sizes["t_slow"]
    n_fast = ds.sizes["t_fast"]
    ratio = n_fast // n_slow
    mid_slow = n_slow // 2
    mid_fast = mid_slow * ratio
    half_a = ds.isel(t_slow=slice(0, mid_slow), t_fast=slice(0, mid_fast))
    half_b = ds.isel(t_slow=slice(mid_slow, n_slow), t_fast=slice(mid_fast, n_fast))
    return [(0, half_a), (1, half_b)], config


class TestFitProbeCalibrationMulti:
    def test_fits_without_error(self, two_profile_list):
        profile_list, config = two_profile_list
        fit = fit_probe_calibration_multi(
            profile_list, "T1", config, fit_file=PFILE.name, order=1
        )
        assert fit.probe == "T1"
        assert fit.sn == "T1592"
        assert fit.profile_index == -1  # sentinel: aggregate fit
        assert np.isfinite(fit.new_T_0)
        assert np.isfinite(fit.new_beta_1)

    def test_aggregates_points_from_every_profile(self, two_profile_list):
        profile_list, config = two_profile_list
        fit = fit_probe_calibration_multi(
            profile_list, "T1", config, fit_file=PFILE.name, order=1
        )
        expected = sum(pds.sizes["t_slow"] for _, pds in profile_list)
        assert fit.n_points == expected

    def test_lag_is_median_of_per_profile_lags(self, two_profile_list):
        profile_list, config = two_profile_list
        single_fits = [
            fit_probe_calibration(
                pds, "T1", config, fit_file=PFILE.name, profile_index=pidx, order=1
            )
            for pidx, pds in profile_list
        ]
        agg_fit = fit_probe_calibration_multi(
            profile_list, "T1", config, fit_file=PFILE.name, order=1
        )
        expected_median = float(np.median([f.lag_s for f in single_fits]))
        np.testing.assert_allclose(agg_fit.lag_s, expected_median)

    def test_improves_agreement_with_reference(self, two_profile_list):
        profile_list, config = two_profile_list
        fit = fit_probe_calibration_multi(
            profile_list, "T1", config, fit_file=PFILE.name, order=1
        )
        assert fit.rms_diff_new_c <= fit.rms_diff_old_c

    def test_raises_with_no_usable_profiles(self, two_profile_list):
        profile_list, config = two_profile_list
        stripped = [(pidx, pds.drop_vars(["T1_counts"])) for pidx, pds in profile_list]
        with pytest.raises(ValueError, match="no usable profiles"):
            fit_probe_calibration_multi(stripped, "T1", config, fit_file=PFILE.name)


class TestReferenceBandwidthCap:
    """ref_fs caps the thermal-matching filter and lag-smoothing cutoff to
    what a discretely-sampled external reference (e.g. a 1 Hz glider CTD)
    can actually resolve -- see REF_BANDWIDTH_MARGIN's docstring. Spies on
    _matching_filter/find_lag to check the exact cutoff each was called
    with, rather than inferring it indirectly from the fit result."""

    def _spy(self, monkeypatch):
        import pyturb.fp07_calibration as fp07mod

        calls: dict = {}
        orig_matching = fp07mod._matching_filter
        orig_find_lag = fp07mod.find_lag

        def spy_matching(C, fs, f_tc):
            calls["fc"] = f_tc
            return orig_matching(C, fs, f_tc)

        def spy_find_lag(T, T_ref, fs, smooth_hz=4.0, **kwargs):
            calls["smooth_hz"] = smooth_hz
            return orig_find_lag(T, T_ref, fs, smooth_hz=smooth_hz, **kwargs)

        monkeypatch.setattr(fp07mod, "_matching_filter", spy_matching)
        monkeypatch.setattr(fp07mod, "find_lag", spy_find_lag)
        return calls

    def test_no_ref_fs_leaves_smoothing_at_default(self, prepared_profile, monkeypatch):
        ds, config = prepared_profile
        calls = self._spy(monkeypatch)
        fit_probe_calibration(
            ds, "T1", config, fit_file=PFILE.name, profile_index=0, order=1
        )
        assert calls["smooth_hz"] == 4.0

    def test_large_ref_fs_does_not_change_the_cutoff(
        self, prepared_profile, monkeypatch
    ):
        ds, config = prepared_profile
        calls_uncapped = self._spy(monkeypatch)
        fit_probe_calibration(
            ds, "T1", config, fit_file=PFILE.name, profile_index=0, order=1
        )
        fc_uncapped = calls_uncapped["fc"]

        calls_capped = self._spy(monkeypatch)
        fit_probe_calibration(
            ds,
            "T1",
            config,
            fit_file=PFILE.name,
            profile_index=0,
            order=1,
            ref_fs=1000.0,
        )
        assert calls_capped["fc"] == pytest.approx(fc_uncapped)
        assert calls_capped["smooth_hz"] == 4.0

    def test_small_ref_fs_caps_matching_filter_and_lag_smoothing(
        self, prepared_profile, monkeypatch
    ):
        ds, config = prepared_profile
        calls = self._spy(monkeypatch)
        fit_probe_calibration(
            ds,
            "T1",
            config,
            fit_file=PFILE.name,
            profile_index=0,
            order=1,
            ref_fs=1.0,
        )
        assert calls["fc"] == pytest.approx(REF_BANDWIDTH_MARGIN * 1.0)
        assert calls["smooth_hz"] == pytest.approx(REF_BANDWIDTH_MARGIN * 1.0)

    def test_fit_still_succeeds_with_a_capped_reference(self, prepared_profile):
        ds, config = prepared_profile
        fit = fit_probe_calibration(
            ds,
            "T1",
            config,
            fit_file=PFILE.name,
            profile_index=0,
            order=1,
            ref_fs=1.0,
        )
        assert np.isfinite(fit.new_T_0)
        assert np.isfinite(fit.new_beta_1)

    def test_ref_fs_threads_through_multi(self, two_profile_list, monkeypatch):
        # fit_probe_calibration_multi recomputes fc per profile (each may
        # have a different mean speed) -- confirm every call is capped, not
        # just the first.
        profile_list, config = two_profile_list
        import pyturb.fp07_calibration as fp07mod

        fc_calls: list[float] = []
        orig_matching = fp07mod._matching_filter

        def spy_matching(C, fs, f_tc):
            fc_calls.append(f_tc)
            return orig_matching(C, fs, f_tc)

        monkeypatch.setattr(fp07mod, "_matching_filter", spy_matching)
        fit_probe_calibration_multi(
            profile_list, "T1", config, fit_file=PFILE.name, order=1, ref_fs=1.0
        )
        assert len(fc_calls) >= len(profile_list)
        assert all(fc <= REF_BANDWIDTH_MARGIN * 1.0 + 1e-9 for fc in fc_calls)


class TestApplyProbeCalibration:
    def test_rebuilds_gradT_exactly(self, prepared_profile):
        ds_prepared, config = prepared_profile
        fit = fit_probe_calibration(
            ds_prepared, "T1", config, fit_file=PFILE.name, profile_index=0, order=1
        )
        raw = to_xarray(load_pfile_phys(PFILE))
        applied = apply_probe_calibration(raw, fit)
        assert applied is not raw
        assert applied["gradT1"].shape == raw["gradT1"].shape
        assert "T1_fp07_recalibrated" in applied["gradT1"].attrs
        # Raw-counts rebuild recovers the full record -- no NaN, unlike a
        # ratio-correction approach that would lose saturated samples.
        assert not np.isnan(applied["gradT1"].values).any()
        assert not np.allclose(applied["gradT1"].values, raw["gradT1"].values)

    def test_also_rebuilds_standalone_temperature(self, prepared_profile):
        # apply_probe_calibration must correct T1/T2 itself, not just
        # gradT1/gradT2 -- T1/T2 feed the binned temperature output directly.
        ds_prepared, config = prepared_profile
        fit = fit_probe_calibration(
            ds_prepared, "T1", config, fit_file=PFILE.name, profile_index=0, order=1
        )
        raw = to_xarray(load_pfile_phys(PFILE))
        applied = apply_probe_calibration(raw, fit)
        assert "T1_fp07_recalibrated" in applied["T1"].attrs
        assert not np.allclose(applied["T1"].values, raw["T1"].values)
        assert np.allclose(applied["T1"].values, applied["JAC_T"].values, atol=0.2)

    def test_stamps_confident_attr_reflecting_fit_is_confident(self, prepared_profile):
        # T1_qc/T2_qc/chi_N_qc read *_fp07_confident (not the resulting
        # value's plausibility) to flag a fit whose own quality is
        # untrustworthy -- see fit_is_confident's docstring.
        ds_prepared, config = prepared_profile
        raw = to_xarray(load_pfile_phys(PFILE))

        confident_fit = fit_probe_calibration(
            ds_prepared, "T1", config, fit_file=PFILE.name, profile_index=0, order=1
        )
        applied = apply_probe_calibration(raw, confident_fit)
        assert "T1_fp07_confident" in applied["T1"].attrs
        assert applied["T1"].attrs["T1_fp07_confident"] == fit_is_confident(
            confident_fit
        )
        assert applied["T1"].attrs["T1_fp07_lag_corr"] == confident_fit.lag_corr

        unconfident_fit = confident_fit.__class__(
            **{**confident_fit.__dict__, "lag_corr": 0.2}
        )
        applied_weak = apply_probe_calibration(raw, unconfident_fit)
        assert applied_weak["T1"].attrs["T1_fp07_confident"] == 0
        assert applied_weak["gradT1"].attrs["T1_fp07_confident"] == 0

    def test_matches_independent_manual_replication(self, prepared_profile):
        from pyturb._pfile import deconvolve, make_gradT
        from pyturb.conductivity import _lag_filter
        from pyturb.fp07_calibration import _channel_params

        ds_prepared, config = prepared_profile
        fit = fit_probe_calibration(
            ds_prepared, "T1", config, fit_file=PFILE.name, profile_index=0, order=1
        )
        raw = to_xarray(load_pfile_phys(PFILE))
        applied = apply_probe_calibration(raw, fit)

        t1_params = _channel_params(raw, "T1")
        dt1_params = _channel_params(raw, "T1_dT1")
        merged = dict(t1_params)
        merged.update(dt1_params)
        merged["t_0"] = fit.new_T_0
        merged["beta_1"] = fit.new_beta_1
        merged.pop("beta_2", None)

        fs_fast = float(raw.fs_fast)
        X_dX = raw["T1_dT1"].values.astype(float)
        X = raw["T1_counts"].values.astype(float)
        T_dec = deconvolve(X_dX, fs_fast, float(dt1_params["diff_gain"]), X)
        grad_manual = make_gradT(
            X_dX, merged, fs_fast, "high_pass", T_deconvolved=T_dec
        )

        lag_samples = -fit.lag_s * fs_fast
        if lag_samples > 0:
            grad_manual = _lag_filter(grad_manual, lag_samples)
        elif lag_samples < 0:
            grad_manual = _lag_filter(grad_manual[::-1], -lag_samples)[::-1]

        np.testing.assert_allclose(
            applied["gradT1"].values.astype(float),
            grad_manual.astype(np.float32).astype(float),
        )

    def test_skips_on_sn_mismatch(self, prepared_profile):
        ds_prepared, config = prepared_profile
        fit = fit_probe_calibration(
            ds_prepared, "T1", config, fit_file=PFILE.name, profile_index=0, order=1
        )
        mismatched = fit.__class__(**{**fit.__dict__, "sn": "not-a-real-sn"})
        raw = to_xarray(load_pfile_phys(PFILE))
        applied = apply_probe_calibration(raw, mismatched)
        assert applied is raw

    def test_preserves_cal_attrs_after_a_successful_apply(self, prepared_profile):
        # A successful apply must not wipe the probe's cal_* attrs -- a
        # later mismatched fit for the same probe relies on _channel_params
        # reading them to decide to skip.
        ds_prepared, config = prepared_profile
        fit = fit_probe_calibration(
            ds_prepared, "T1", config, fit_file=PFILE.name, profile_index=0, order=1
        )
        raw = to_xarray(load_pfile_phys(PFILE))
        applied = apply_probe_calibration(raw, fit)
        assert applied["T1"].attrs.get("cal_sn") == fit.sn

        mismatched = fit.__class__(**{**fit.__dict__, "sn": "not-a-real-sn"})
        again = apply_probe_calibration(applied, mismatched)
        assert again is applied
        assert again["T1"].attrs.get("cal_sn") == fit.sn

    def test_skips_on_instrument_sn_mismatch(self, prepared_profile):
        # Same probe SN, different instrument -- must not apply. Guards
        # against generic/placeholder probe SNs being reused across probes.
        ds_prepared, config = prepared_profile
        fit = fit_probe_calibration(
            ds_prepared, "T1", config, fit_file=PFILE.name, profile_index=0, order=1
        )
        mismatched = fit.__class__(
            **{**fit.__dict__, "instrument_sn": "not-a-real-instrument"}
        )
        raw = to_xarray(load_pfile_phys(PFILE))
        applied = apply_probe_calibration(raw, mismatched)
        assert applied is raw

    def test_raises_without_raw_counts(self, prepared_profile):
        ds_prepared, config = prepared_profile
        fit = fit_probe_calibration(
            ds_prepared, "T1", config, fit_file=PFILE.name, profile_index=0, order=1
        )
        raw = to_xarray(load_pfile_phys(PFILE))
        legacy = raw.drop_vars(["T1_dT1", "T1_counts"])
        with pytest.raises(ValueError, match="T1_counts"):
            apply_probe_calibration(legacy, fit)


class TestNoMaskingOnExtrapolation:
    """apply_probe_calibration applies the best available calibration as-is
    and does not mask/NaN implausible (e.g. extrapolated) values -- that
    would silently destroy data. Flagging physically implausible T1/T2 or
    chi is the eps step's job (see profile.py's range-based QC), not this
    conversion step's."""

    def _make_ds(self, counts: np.ndarray) -> xr.Dataset:
        n = counts.size
        cal_attrs = {
            "cal_a": str(A),
            "cal_b": str(B),
            "cal_g": str(G),
            "cal_e_b": str(E_B),
            "cal_adc_fs": str(ADC_FS),
            "cal_adc_bits": str(ADC_BITS),
            "cal_t_0": str(T_0),
            "cal_beta_1": str(BETA_1),
            "cal_sn": "SYN1",
        }
        ds = xr.Dataset(
            {
                "T1": ("t_slow", np.full(n, 10.0), cal_attrs),
                "T1_counts": ("t_slow", counts),
                "T1_dT1": ("t_fast", counts, {**cal_attrs, "cal_diff_gain": "0.9"}),
                "gradT1": ("t_fast", np.zeros(n)),
                "JAC_T": ("t_slow", np.full(n, 10.0)),
                "fs_slow": 64.0,
                "fs_fast": 64.0,
            },
            coords={"t_slow": np.arange(n) / 64.0, "t_fast": np.arange(n) / 64.0},
            attrs={"instrument_sn": "142"},
        )
        return ds

    def _make_fit(self) -> ProbeCalibrationFit:
        return ProbeCalibrationFit(
            probe="T1",
            sn="SYN1",
            instrument_sn="142",
            fit_file="synthetic",
            profile_index=0,
            reference="JAC_T",
            order=1,
            old_T_0=T_0,
            old_beta_1=BETA_1,
            old_beta_2=None,
            new_T_0=T_0,
            new_beta_1=BETA_1,
            new_beta_2=None,
            lag_s=0.0,
            lag_corr=1.0,
            n_points=100,
            temperature_range_c=10.0,
            rms_diff_old_c=0.0,
            rms_diff_new_c=0.0,
            max_abs_diff_old_c=0.0,
            max_abs_diff_new_c=0.0,
            mean_bias_old_c=0.0,
            mean_bias_new_c=0.0,
            residual_std_c=0.0,
            fit_date="2026-01-01T00:00:00+00:00",
        )

    def test_does_not_mask_extrapolation_blowup(self):
        n = 3000
        # -4000 counts -> ~9.8C (sane); 20000 counts -> ~59.6C (physically
        # implausible for seawater, but must be left as computed here).
        counts = np.full(n, -4000.0)
        counts[1000:1100] = 20000.0
        ds = self._make_ds(counts)
        fit = self._make_fit()

        applied = apply_probe_calibration(ds, fit)
        T1 = applied["T1"].values
        gradT1 = applied["gradT1"].values

        assert not np.isnan(T1).any()
        assert not np.isnan(gradT1).any()
        assert np.all((T1[:1000] > 0) & (T1[:1000] < 20))
        assert np.all(T1[1000:1100] > 40)  # implausible, but present
        assert "T1_fp07_n_out_of_range" not in applied["T1"].attrs
