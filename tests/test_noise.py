"""Tests for the RSI electronics noise models and the FP07 gradT noise floor."""

import numpy as np
import xarray as xr

from pyturb.fp07_calibration import steinhart_hart_forward
from pyturb.noise import (
    _channel_calibration_params,
    _log_R_from_T,
    noise_shearchannel,
    noise_thermchannel,
    thermistor_noise_phi,
)

T_0, BETA_1, BETA_2 = 289.301, 3143.55, 250000.0
# Real T2 electronics constants (RIOT_VMP412_0010.nc, SN T2248).
A, B, G, E_B, ADC_FS, ADC_BITS = -5.1, 0.99937, 6.0, 0.68268, 4.096, 16
DIFF_GAIN = 0.969

CAL_PARAMS = dict(
    a=A,
    b=B,
    g=G,
    e_b=E_B,
    adc_fs=ADC_FS,
    adc_bits=ADC_BITS,
    t_0=T_0,
    beta_1=BETA_1,
    beta_2=BETA_2,
    diff_gain=DIFF_GAIN,
)


class TestElectronicsNoiseModels:
    def test_noise_thermchannel_positive_finite(self):
        f = np.linspace(0.5, 200, 100)
        n = noise_thermchannel(f)
        assert np.all(np.isfinite(n))
        assert np.all(n > 0)

    def test_noise_shearchannel_positive_finite(self):
        f = np.linspace(0.5, 200, 100)
        n = noise_shearchannel(f)
        assert np.all(np.isfinite(n))
        assert np.all(n > 0)


class TestLogRFromT:
    def test_round_trips_through_steinhart_hart_forward(self):
        log_R = np.array([-0.5, 0.0, 0.8, 3.5])
        T_K = 1.0 / (1 / T_0 + log_R / BETA_1 + log_R**2 / BETA_2)
        T_celsius = T_K - 273.15
        recovered = _log_R_from_T(T_celsius, T_0, BETA_1, BETA_2)
        np.testing.assert_allclose(recovered, log_R, atol=1e-8)

    def test_round_trips_first_order(self):
        log_R = np.array([-0.5, 0.0, 0.8])
        T_K = 1.0 / (1 / T_0 + log_R / BETA_1)
        T_celsius = T_K - 273.15
        recovered = _log_R_from_T(T_celsius, T_0, BETA_1, beta_2=None)
        np.testing.assert_allclose(recovered, log_R, atol=1e-8)

    def test_matches_steinhart_hart_forward_inverse(self):
        # steinhart_hart_forward: log_R -> T. This is its inverse; round
        # tripping through both must be the identity.
        log_R = np.linspace(-1, 1, 11)
        T_K = steinhart_hart_forward(log_R, T_0, BETA_1, BETA_2)
        recovered = _log_R_from_T(T_K - 273.15, T_0, BETA_1, BETA_2)
        np.testing.assert_allclose(recovered, log_R, atol=1e-8)


class TestChannelCalibrationParams:
    def _make_ds(self, recalibrated=False):
        cal_attrs = {
            "cal_a": str(A),
            "cal_b": str(B),
            "cal_g": str(G),
            "cal_e_b": str(E_B),
            "cal_adc_fs": str(ADC_FS),
            "cal_adc_bits": str(ADC_BITS),
            "cal_t_0": str(T_0),
            "cal_beta_1": str(BETA_1),
            "cal_beta_2": str(BETA_2),
        }
        if recalibrated:
            cal_attrs.update(
                {
                    "T2_fp07_recalibrated": np.int8(1),
                    "T2_fp07_new_T_0": 280.0,
                    "T2_fp07_new_beta_1": 3000.0,
                    "T2_fp07_new_beta_2": np.nan,
                }
            )
        ds = xr.Dataset(
            {
                "T2": ("t_slow", np.array([10.0, 11.0]), cal_attrs),
                "T2_dT2": (
                    "t_fast",
                    np.array([1.0, 2.0]),
                    {"cal_diff_gain": str(DIFF_GAIN)},
                ),
            }
        )
        return ds

    def test_reads_cal_attrs_and_diff_gain(self):
        params = _channel_calibration_params(self._make_ds(), "T2")
        assert params["a"] == str(A)
        assert params["diff_gain"] == str(DIFF_GAIN)
        assert "T2_fp07_new_T_0" not in params

    def test_missing_probe_returns_empty(self):
        assert _channel_calibration_params(self._make_ds(), "T1") == {}

    def test_prefers_recalibrated_coefficients(self):
        params = _channel_calibration_params(self._make_ds(recalibrated=True), "T2")
        assert params["t_0"] == 280.0
        assert params["beta_1"] == 3000.0
        # NaN new_beta_2 (order-1 recalibration) -> falls back to no beta_2
        assert "beta_2" not in params


class TestThermistorNoisePhi:
    def test_positive_finite_away_from_dc(self):
        f = np.linspace(1, 200, 200)
        phi = thermistor_noise_phi(
            f, W=0.6, T_celsius=10.0, params=CAL_PARAMS, fs_fast=512.0
        )
        assert np.all(np.isfinite(phi))
        assert np.all(phi > 0)

    def test_larger_at_slower_fall_speed(self):
        # Lower W both concentrates the counts noise into a smaller physical
        # gradient scale (1/W^2) and steepens the FP07 thermal-response
        # correction -- noise floor should be higher at low W.
        f = np.linspace(1, 100, 50)
        phi_slow = thermistor_noise_phi(
            f, W=0.3, T_celsius=10.0, params=CAL_PARAMS, fs_fast=512.0
        )
        phi_fast = thermistor_noise_phi(
            f, W=1.2, T_celsius=10.0, params=CAL_PARAMS, fs_fast=512.0
        )
        assert np.all(phi_slow > phi_fast)
