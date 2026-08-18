"""Tests for JAC-CT conductivity/temperature matching (port of salinity_JAC.m)."""

import numpy as np

from pyturb.conductivity import match_conductivity_to_temperature

FS = 64.0


def _step_signal(n: int, fs: float, step_time: float = 10.0) -> np.ndarray:
    t = np.arange(n) / fs
    return np.where(t < step_time, 35.0, 36.0).astype(float)


def _crossing_index(x: np.ndarray, level: float = 35.5) -> int:
    return int(np.argmax(x > level))


class TestMatchConductivityToTemperature:
    def test_reduces_a_known_lag(self):
        # Simulate a conductivity sensor that responds *early* relative to
        # temperature by a known amount (matching the correction's own
        # convention: it delays C to bring it into alignment), then confirm
        # correcting with that same lag brings the step transition closer to
        # its true location.
        speed = 0.62  # == reference_speed, so scaled_lag == lag exactly
        lag_s = 0.05
        n = 3000
        C_true = _step_signal(n, FS)
        shift = int(round(lag_s * FS))
        C_measured = np.roll(C_true, -shift)
        C_measured[-shift:] = C_true[-1]

        C_matched = match_conductivity_to_temperature(
            C_measured, FS, speed, lag=lag_s, f_tc=20.0, reference_speed=speed
        )

        true_idx = _crossing_index(C_true)
        raw_idx = _crossing_index(C_measured)
        matched_idx = _crossing_index(C_matched)

        assert abs(matched_idx - true_idx) < abs(raw_idx - true_idx)

    def test_output_same_length_as_input(self):
        C = np.linspace(30.0, 40.0, 1000)
        out = match_conductivity_to_temperature(C, FS, 0.5)
        assert len(out) == len(C)

    def test_zero_lag_only_applies_matching_filter(self):
        C = _step_signal(2000, FS)
        out = match_conductivity_to_temperature(C, FS, 0.5, lag=0.0)
        # No crash, no shift introduced beyond the low-pass filter's own delay
        assert len(out) == len(C)
        assert np.all(np.isfinite(out))

    def test_invalid_speed_returns_input_unchanged(self):
        C = np.linspace(30.0, 40.0, 500)
        for bad_speed in (0.0, -1.0, np.nan):
            out = match_conductivity_to_temperature(C, FS, bad_speed)
            np.testing.assert_array_equal(out, C)

    def test_too_short_signal_returns_input_unchanged(self):
        C = np.array([35.0, 35.1, 35.2])
        out = match_conductivity_to_temperature(C, FS, 0.5)
        np.testing.assert_array_equal(out, C)

    def test_near_zero_speed_returns_input_unchanged(self):
        # A near-stationary segment blows scaled_lag/scaled_f_tc up towards
        # a filter length longer than the signal -- must degrade gracefully
        # with a clear guard, not an opaque shape-mismatch error.
        C = np.linspace(30.0, 40.0, 5120)
        out = match_conductivity_to_temperature(C, FS, speed=4.4e-05)
        np.testing.assert_array_equal(out, C)

    def test_higher_speed_scales_lag_down(self):
        # At double the reference speed, the effective (scaled) lag is half
        # -- so correcting a fixed synthetic lag with a higher input speed
        # should remove less of it than correcting at the reference speed.
        lag_s = 0.1
        n = 3000
        C_true = _step_signal(n, FS)
        shift = int(round(lag_s * FS))
        C_measured = np.roll(C_true, -shift)
        C_measured[-shift:] = C_true[-1]

        ref_speed = 0.62
        out_ref = match_conductivity_to_temperature(
            C_measured, FS, ref_speed, lag=lag_s, f_tc=20.0, reference_speed=ref_speed
        )
        out_fast = match_conductivity_to_temperature(
            C_measured,
            FS,
            2 * ref_speed,
            lag=lag_s,
            f_tc=20.0,
            reference_speed=ref_speed,
        )

        true_idx = _crossing_index(C_true)
        err_ref = abs(_crossing_index(out_ref) - true_idx)
        err_fast = abs(_crossing_index(out_fast) - true_idx)
        assert err_ref < err_fast
