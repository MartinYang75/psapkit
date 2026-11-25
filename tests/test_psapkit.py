from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import polars as pl
import pytest

from psapkit import (
    FlowCalibrationConfig, FlowSegment, ScreeningConfig, compute_psap_absorption,
    decode_psap_hex, detect_rapid_transients, process_l1_5_to_l2,
    process_l1_to_l1_5, process_raw_to_l1, run_psap_workflow,
)
from psapkit.absorption import MOD_40BIT


SAMPLE_DIR = Path(__file__).parents[1] / "examples" / "sample_data"
REAL_SAMPLES = (
    SAMPLE_DIR / "psap1_2013-01-01.raw",
    SAMPLE_DIR / "psap1_2015-01-01.ra5",
    SAMPLE_DIR / "psap1_2017-01-01.raw",
)


def test_hex_decode_and_bounds():
    decoded = decode_psap_hex("01,10,20,03,00")
    assert decoded["row_type"] == "BLUE"
    assert decoded["blue_ad_sample"] == 16


@pytest.mark.parametrize("path", REAL_SAMPLES, ids=lambda path: path.name)
def test_real_raw_samples_parse_to_l1(path):
    frame = process_raw_to_l1(path)
    assert frame.height == 1440
    assert frame["timestamp"].is_sorted()
    assert {"flow_lpm", "tr_green", "green_ad_sample"}.issubset(frame.columns)


def test_one_minute_assembly_from_memory():
    rows = []
    for second, payload in [(0, "00,00,100,20"), (1, "01,10,20,03,00"),
                            (2, "02,10,20,03,00"), (3, "03,10,20,03,00"),
                            (4, "04,00,10,20,30,40")]:
        rows.append(f'2020-01-01 00:00:{second:02d} x 1 1 1 0.9 0.9 0.9 1 x x x "{payload}"')
    frame = process_raw_to_l1(StringIO("\n".join(rows)))
    assert frame.height == 1
    assert frame["blue_ad_sample"][0] == 16


def l1_frame(n=20):
    timestamps = [datetime(2020, 1, 1) + timedelta(minutes=i) for i in range(n)]
    data = {"timestamp": timestamps, "flow_lpm": [1.] * n, "tr_blue": [.95-i*.001 for i in range(n)],
            "tr_green": [.95-i*.001 for i in range(n)], "tr_red": [.95-i*.001 for i in range(n)]}
    for band in ("blue", "green", "red"):
        data[f"{band}_ad_sample"] = [100000-i*200 for i in range(n)]
        data[f"{band}_ad_ref"] = [100000] * n
    return pl.DataFrame(data)


def test_spot_absorption_and_flow_calibration():
    output = process_l1_to_l1_5(l1_frame(), flow_config=FlowCalibrationConfig.disabled())
    assert "spot_id" in output.columns
    assert "tr_recomputed_green" in output.columns
    assert "sigma_0_green" in output.columns
    custom = FlowCalibrationConfig(segments=(FlowSegment(datetime(2019, 1, 1), 1., 0.),))
    calibrated = process_l1_to_l1_5(l1_frame(), flow_config=custom)
    assert calibrated["fq"][1] == pytest.approx(1.)


def test_flow_regime_and_q_corrected_alias_and_screen_flag():
    raw = l1_frame(4).with_columns(pl.Series("flow_lpm", [1.0, .8, .6, 1.0]))
    default_gt05 = FlowCalibrationConfig.dome_c_default(regime="gt05")
    default_all = FlowCalibrationConfig.dome_c_default(regime="all")
    assert default_gt05.regime == "gt05"
    assert default_all.regime == "all"
    calibrated = process_l1_to_l1_5(raw, flow_config=default_all)
    assert "flow_lpm_q_corrected" in calibrated.columns
    flagged = process_l1_5_to_l2(calibrated, config=ScreeningConfig(min_flow_lpm=.9, enable_transient_screening=False))
    assert flagged["flag_low_flow"].any()


def test_l2_without_and_with_meteo():
    stage = process_l1_to_l1_5(l1_frame(), flow_config=FlowCalibrationConfig.disabled())
    without = process_l1_5_to_l2(stage)
    assert not without["flag_wind_contam"].any()
    meteo = pl.DataFrame({"timestamp": stage["timestamp"], "wind_direction": [50.] * stage.height,
                          "wind_speed": [4.] * stage.height})
    with_meteo = process_l1_5_to_l2(stage, meteo_df=meteo)
    assert with_meteo["flag_wind_contam"].all()


@pytest.mark.parametrize("length", [12, 96, 900])
def test_multiscale_transients_stable_short_and_long(length):
    timestamps = [datetime(2021, 1, 1) + timedelta(hours=i) for i in range(length)]
    values = [0.01 + (0.001 if i % 17 == 0 else 0.) for i in range(length)]
    candidates = [True] * length
    if length > 30:
        for i in range(20, min(25, length)):
            candidates[i] = False
        if length > 40:
            values[min(30, length - 1)] = 0.05
    frame = pl.DataFrame({"timestamp": timestamps, "sigma_stp_green": values,
                          "flag_base_candidate": candidates})
    result = detect_rapid_transients(frame)
    assert result.height == length
    for col in ("med_13h", "med_73h", "med_361h", "iforest_prob", "flag_rapid_transient"):
        assert col in result.columns
    assert result["flag_rapid_transient"].null_count() == 0


def test_transient_detector_configurable_windows_and_small_iforest_fallback():
    frame = pl.DataFrame({
        "timestamp": [datetime(2022, 1, 1) + timedelta(hours=i) for i in range(5)],
        "sigma_0_green": [0.01, 0.01, 0.03, 0.01, 0.01],
        "flag_base_candidate": [True] * 5,
    })
    cfg = ScreeningConfig(hampel_windows=(3, 5, 7), hampel_min_samples=(1, 2, 2), iforest_min_samples=8)
    result = detect_rapid_transients(frame, cfg)
    assert result.height == 5
    assert result["flag_rapid_transient"].dtype == pl.Boolean


def test_real_sample_hourly_count_inversion():
    frame = process_raw_to_l1(REAL_SAMPLES[0])
    hourly = compute_psap_absorption(frame, resolution="1h")
    assert hourly.height == 24
    assert hourly["n_valid"].max() >= 30
    assert hourly["sigma_0_green"].drop_nulls().len() >= 20


def test_hourly_absorption_uses_integrated_counts_and_wraparound():
    start = datetime(2022, 1, 1)
    n = 180
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    sample = [(MOD_40BIT - 3000 + i * 100) % MOD_40BIT for i in range(n)]
    ref = [(MOD_40BIT - 6000 + i * 200) % MOD_40BIT for i in range(n)]
    data = {"timestamp": timestamps, "flow_lpm": [1.] * n, "tr_blue": [.95] * n,
            "tr_green": [.95] * n, "tr_red": [.95] * n}
    for band in ("blue", "green", "red"):
        data[f"{band}_ad_sample"] = sample
        data[f"{band}_ad_ref"] = ref
    hourly = compute_psap_absorption(pl.DataFrame(data), resolution="1h")
    assert hourly.height >= 2
    assert hourly["n_valid"].max() >= 30
    assert hourly["sigma_0_green"].drop_nulls().len() > 0


def test_real_samples_hourly_outputs_have_one_day_each():
    for path in REAL_SAMPLES:
        hourly = compute_psap_absorption(process_raw_to_l1(path), resolution="1h")
        assert hourly.height == 24, path.name
        assert hourly["n_valid"].max() >= 30, path.name


def test_workflow_end_to_end_three_raw_files(tmp_path):
    raw_lines = []
    packets = [(0, "00,00,100,20"), (1, "01,10,20,03,00"), (2, "02,10,20,03,00"),
               (3, "03,10,20,03,00"), (4, "04,00,10,20,30,40")]
    for minute in range(20):
        for second, payload in packets:
            raw_lines.append(f'2020-01-01 00:{minute:02d}:{second:02d} x 1 1 1 0.9 0.9 0.9 1 x x x "{payload}"')
    text = "\n".join(raw_lines)
    for index in range(3):
        (tmp_path / f"sample_{index}.raw").write_text(text, encoding="latin-1")
    result = run_psap_workflow(tmp_path, flow_config=FlowCalibrationConfig.disabled())
    assert result.l1.height == result.l1_5.height == result.l2.height == 20
    assert "flag_rapid_transient" in result.l2.columns


def test_workflow_end_to_end_filelike():
    raw = []
    for minute in range(4):
        for second, payload in [(0, "00,00,100,20"), (1, "01,10,20,03,00"),
                                (2, "02,10,20,03,00"), (3, "03,10,20,03,00"),
                                (4, "04,00,10,20,30,40")]:
            raw.append(f'2020-01-01 00:{minute:02d}:{second:02d} x 1 1 1 0.9 0.9 0.9 1 x x x "{payload}"')
    result = run_psap_workflow(StringIO("\n".join(raw)), flow_config=FlowCalibrationConfig.disabled())
    assert result.l1.height == 4
    assert "flag_clean_background" in result.l2.columns


def test_dome_c_flow_calibration_all_three_periods():
    from psapkit.calibration import apply_flow_calibration

    df = pl.DataFrame({
        "timestamp": [datetime(2013, 1, 1), datetime(2015, 1, 1), datetime(2020, 1, 1)],
        "flow_lpm": [1.0, 1.0, 1.0],
        "sigma_0_green": [0.1, 0.1, 0.1],
        "sigma_b1999_green": [0.04, 0.04, 0.04],
    })
    out = apply_flow_calibration(df, FlowCalibrationConfig.dome_c_default("gt05"))
    base = datetime(2007, 9, 29)
    expected_q = []
    for ts, (s0, i0, sd, id_) in zip(
        [datetime(2013, 1, 1), datetime(2015, 1, 1), datetime(2020, 1, 1)],
        [
            (0.996780, 0.254151, -8.5015e-5, -1.0025e-5),
            (0.266544, 0.449459, 1.7495e-4, -6.8862e-5),
            (0.798455, 0.149982, 2.4040e-5, -2.5906e-5),
        ],
    ):
        nd = (ts - base).total_seconds() / 86400.0
        expected_q.append((s0 + sd * nd) * 1.0 + (i0 + id_ * nd))
    for i, eq in enumerate(expected_q):
        assert out["flow_lpm_q_corrected"][i] == pytest.approx(eq)
        assert out["fq"][i] == pytest.approx(1.0 / eq)
    assert out["sigma_b1999_green_q_corrected"][2] == pytest.approx(0.04 / expected_q[2])


def test_l1_5_bond1999_and_clean_background_subset():
    result = run_psap_workflow(REAL_SAMPLES[0], resolution="1h")
    assert result.l1_5.height == 24
    assert "sigma_b1999_green_q_corrected" in result.l1_5.columns
    assert "flag_has_psap" in result.l2.columns
    assert "flag_spike_10m" in result.l2.columns
    # flag_clean_background must always be a subset of flag_base_candidate
    invalid_clean = result.l2.filter(pl.col("flag_clean_background") & (~pl.col("flag_base_candidate")))
    assert invalid_clean.height == 0
