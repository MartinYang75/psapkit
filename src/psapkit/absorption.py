"""PSAP spot tracking, counter-based transmittance and absorption inversion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from .pipeline import _save

BANDS = ("blue", "green", "red")
MOD_40BIT = 1 << 40
HALF_40BIT = 1 << 39
VIRKKULA2010_K: dict[str, tuple[float, float]] = {
    "blue": (0.355, 0.495),
    "green": (0.340, 0.470),
    "red": (0.315, 0.435),
}


@dataclass(frozen=True)
class AbsorptionConfig:
    spot_area_mm2: float = 17.81
    min_spot_separation_hours: float = 3.0
    spot_jump_threshold: float = 0.03
    loading_correction: Literal["bond1999", "virkkula2010", "none"] = "bond1999"
    bond_k1: float = 1.5557
    bond_k2: float = 1.0227
    virkkula_k: dict[str, tuple[float, float]] | None = None
    min_flow_lpm: float = 0.5

    @classmethod
    def dome_c_default(cls) -> "AbsorptionConfig":
        """Return the Dome C preset with empirical effective spot area 18.9 mm^2."""
        return cls(spot_area_mm2=18.9)


def _add_spot_tracking(df: pl.DataFrame, cfg: AbsorptionConfig) -> pl.DataFrame:
    if "spot_id" in df.columns:
        ids = df["spot_id"].to_list()
        if "_is_spot_transition" not in df.columns:
            per_spot_count: dict[int, int] = {}
            transitions = []
            for spot in ids:
                pos = per_spot_count.get(spot, 0)
                transitions.append(pos < 2)
                per_spot_count[spot] = pos + 1
            df = df.with_columns(pl.Series("_is_spot_transition", transitions, dtype=pl.Boolean))
        return df
    tr_values = df.get_column("tr_green").to_list() if "tr_green" in df.columns else [None] * df.height
    times = df["timestamp"].to_list()
    raw_changes = []
    last_valid = None
    for i, value in enumerate(tr_values):
        raw_changes.append(i == 0 or (value is not None and last_valid is not None and value - last_valid > cfg.spot_jump_threshold and value >= .85))
        if value is not None:
            last_valid = value
    accepted, last_change = [], None
    for i, is_change in enumerate(raw_changes):
        if is_change and (last_change is None or (times[i] - times[last_change]).total_seconds() >= cfg.min_spot_separation_hours * 3600):
            accepted.append(i)
            last_change = i
    is_new = [i in set(accepted) for i in range(df.height)]
    spot_ids, current = [], 0
    for change in is_new:
        current += int(change)
        spot_ids.append(current)
    # Exclude the first two records of every spot, matching the original
    # transition mask used to avoid counter discontinuities after a filter swap.
    per_spot_count: dict[int, int] = {}
    transition = []
    for spot in spot_ids:
        position = per_spot_count.get(spot, 0)
        transition.append(position < 2)
        per_spot_count[spot] = position + 1
    return df.with_columns(
        pl.Series("is_new_filter", is_new, dtype=pl.Boolean),
        pl.Series("spot_id", spot_ids, dtype=pl.UInt32),
        pl.Series("_is_spot_transition", transition, dtype=pl.Boolean),
    )


def _counter_expr(column: str) -> pl.Expr:
    delta = pl.col(column).cast(pl.Int64).diff()
    return ((delta % MOD_40BIT) + MOD_40BIT) % MOD_40BIT


def _prepare_counter_data(df: pl.DataFrame, cfg: AbsorptionConfig) -> pl.DataFrame:
    dt = (pl.col("timestamp") - pl.col("timestamp").shift(1)).dt.total_microseconds() / 1_000_000.0
    df = df.with_columns(dt.alias("step_dt_seconds"))
    exprs = []
    for band in BANDS:
        delta_sample = _counter_expr(f"{band}_ad_sample")
        delta_ref = _counter_expr(f"{band}_ad_ref")
        valid = (
            (pl.col("spot_id") == pl.col("spot_id").shift(1))
            & (~pl.col("_is_spot_transition"))
            & pl.col("step_dt_seconds").is_between(30., 90.)
            & (pl.col("flow_lpm") > cfg.min_flow_lpm)
            & (delta_sample > 0) & (delta_ref > 0)
            & (delta_sample < HALF_40BIT) & (delta_ref < HALF_40BIT)
            & pl.col(f"tr_{band}").is_between(.10, 1.05)
        )
        exprs.extend([
            pl.when(valid).then(delta_sample).otherwise(None).alias(f"_dsig_{band}"),
            pl.when(valid).then(delta_ref).otherwise(None).alias(f"_dref_{band}"),
            pl.when(valid).then(delta_sample.cast(pl.Float64) / delta_ref.cast(pl.Float64)).otherwise(None).alias(f"_ratio_{band}"),
        ])
    return df.with_columns(exprs)


def _recompute_transmittance(df: pl.DataFrame) -> pl.DataFrame:
    r0s = []
    for band in BANDS:
        normalized = pl.col(f"_ratio_{band}") / pl.col(f"tr_{band}")
        stable = (
            pl.col(f"_ratio_{band}").is_not_null()
            & pl.col(f"tr_{band}").is_between(.70, 1.02)
            & ((pl.col(f"tr_{band}") - pl.col(f"tr_{band}").shift(1)).abs() <= .002)
        )
        stats = df.with_columns(stable.alias("_stable")).group_by("spot_id").agg([
            pl.col("_stable").sum().alias("_stable_n"),
            pl.when(pl.col("_stable")).then(normalized).otherwise(None).median().alias("_r0_stable"),
            normalized.median().alias("_r0_all"),
        ]).select([
            "spot_id", pl.when(pl.col("_stable_n") >= 10).then(pl.col("_r0_stable")).otherwise(pl.col("_r0_all")).alias(f"_r0_{band}")
        ])
        r0s.append(stats)
    for stats in r0s:
        df = df.join(stats, on="spot_id", how="left")
    exprs = []
    for band in BANDS:
        value = pl.col(f"_ratio_{band}") / pl.col(f"_r0_{band}")
        valid = value.is_finite() & value.is_between(.10, 1.05) & ((value - pl.col(f"tr_{band}")).abs() <= .015)
        exprs.append(pl.when(valid).then(value).otherwise(None).alias(f"tr_recomputed_{band}"))
    return df.with_columns(exprs)


def _add_sigma_1m(df: pl.DataFrame, spot_area_m2: float, cfg: AbsorptionConfig) -> pl.DataFrame:
    volume = pl.col("flow_lpm").cast(pl.Float64) * (1e-3 / 60.) * pl.col("step_dt_seconds")
    valid_base = (
        (pl.col("spot_id") == pl.col("spot_id").shift(1))
        & (~pl.col("_is_spot_transition"))
        & pl.col("step_dt_seconds").is_between(30., 90.)
        & (pl.col("flow_lpm") > cfg.min_flow_lpm) & (volume > 0)
    )
    exprs = []
    for band in BANDS:
        curr, prev = pl.col(f"tr_recomputed_{band}"), pl.col(f"tr_recomputed_{band}").shift(1)
        valid = valid_base & curr.is_not_null() & prev.is_not_null() & (curr > .10) & (prev > .10)
        exprs.append(pl.when(valid).then(spot_area_m2 / volume * (prev.log() - curr.log()) * 1e6).otherwise(None).alias(f"sigma_0_{band}"))
    return df.with_columns(exprs)


def _add_sigma_1h(df: pl.DataFrame, spot_area_m2: float, cfg: AbsorptionConfig) -> pl.DataFrame:
    """Integrate raw 40-bit detector increments in hourly bins, then invert Tr."""
    one_minute = _add_sigma_1m(df, spot_area_m2, cfg)
    one_minute = one_minute.with_columns(
        pl.col("sigma_0_green").rolling_mean(window_size=10, min_samples=5).over("spot_id").alias("_s0_10m_ref_green")
    )
    valid_minutes = one_minute.filter(
        (pl.col("flow_lpm") > cfg.min_flow_lpm)
        & pl.col("step_dt_seconds").is_between(30., 90.)
    )
    if valid_minutes.is_empty():
        return pl.DataFrame(schema={"timestamp": pl.Datetime("us"), "n_valid": pl.UInt32, "spot_id": pl.UInt32,
                                    "flow_lpm": pl.Float64, "tr_blue": pl.Float64, "tr_green": pl.Float64,
                                    "tr_red": pl.Float64, "tr_recomputed_blue": pl.Float64,
                                    "tr_recomputed_green": pl.Float64, "tr_recomputed_red": pl.Float64,
                                    "sigma_0_blue": pl.Float64, "sigma_0_green": pl.Float64,
                                    "sigma_0_red": pl.Float64, "spike_10m": pl.Boolean})
    aggregations = [
        pl.col("tr_recomputed_green").drop_nulls().len().cast(pl.UInt32).alias("n_valid"),
        pl.col("flow_lpm").mean().alias("flow_lpm"),
        pl.col("spot_id").last().alias("spot_id"),
        (pl.col("spot_id").min() == pl.col("spot_id").max()).alias("_spot_consistent"),
        pl.col("_s0_10m_ref_green").drop_nulls().max().alias("_s0_10m_max_green"),
    ]
    for band in BANDS:
        aggregations.extend([
            pl.col(f"_dsig_{band}").sum().cast(pl.Float64).alias(f"_hourly_sig_{band}"),
            pl.col(f"_dref_{band}").sum().cast(pl.Float64).alias(f"_hourly_ref_{band}"),
            pl.col(f"_r0_{band}").drop_nulls().median().alias(f"_r0_{band}"),
            pl.col(f"tr_{band}").drop_nulls().median().alias(f"_firmware_tr_{band}"),
        ])
    hourly = valid_minutes.group_by_dynamic("timestamp", every="1h", period="1h", closed="left", label="left").agg(aggregations).sort("timestamp")
    tr_exprs = []
    for band in BANDS:
        ratio = (pl.col(f"_hourly_sig_{band}") / pl.col(f"_hourly_ref_{band}")) / pl.col(f"_r0_{band}")
        valid = (
            (pl.col(f"_hourly_sig_{band}") > 0) & (pl.col(f"_hourly_ref_{band}") > 0)
            & ratio.is_between(.10, 1.05)
            & ((ratio - pl.col(f"_firmware_tr_{band}")).abs() <= .015)
        )
        tr_exprs.extend([
            pl.when(valid).then(ratio).otherwise(None).alias(f"tr_recomputed_{band}"),
            pl.col(f"_firmware_tr_{band}").alias(f"tr_{band}"),
        ])
    hourly = hourly.with_columns(tr_exprs).with_columns(
        (.5 * (pl.col("flow_lpm") + pl.col("flow_lpm").shift(1))).alias("_flow_eff_lpm")
    ).with_columns((pl.col("_flow_eff_lpm") * 1e-3 * 60.).alias("_volume_1h"))
    dt = (pl.col("timestamp") - pl.col("timestamp").shift(1)).dt.total_seconds()
    valid_hour = (
        (pl.col("n_valid") >= 30) & (pl.col("n_valid").shift(1) >= 30)
        & pl.col("_spot_consistent") & pl.col("_spot_consistent").shift(1)
        & (pl.col("spot_id") == pl.col("spot_id").shift(1)) & (dt == 3600.)
        & (pl.col("_volume_1h") > 0)
    )
    sigmas = []
    for band in BANDS:
        curr, prev = pl.col(f"tr_recomputed_{band}"), pl.col(f"tr_recomputed_{band}").shift(1)
        valid = valid_hour & curr.is_not_null() & prev.is_not_null() & (curr > .10) & (prev > .10)
        sigmas.append(pl.when(valid).then(spot_area_m2 / pl.col("_volume_1h") * (prev.log() - curr.log()) * 1e6).otherwise(None).alias(f"sigma_0_{band}"))
    hourly = hourly.with_columns(sigmas)
    hourly = hourly.with_columns(
        ((pl.col("_s0_10m_max_green") > 10. * pl.col("sigma_0_green")) & (pl.col("sigma_0_green") > 0)).fill_null(False).alias("spike_10m")
    )
    return hourly.drop([c for c in (
        "_spot_consistent", "_s0_10m_max_green", "_flow_eff_lpm", "_volume_1h",
        *[f"_hourly_{kind}_{band}" for kind in ("sig", "ref") for band in BANDS],
        *[f"_r0_{band}" for band in BANDS], *[f"_firmware_tr_{band}" for band in BANDS],
    ) if c in hourly.columns])


def _apply_loading_correction(df: pl.DataFrame, cfg: AbsorptionConfig) -> pl.DataFrame:
    if cfg.loading_correction == "none" or df.is_empty():
        return df
    exprs: list[pl.Expr] = []
    if cfg.loading_correction == "bond1999":
        for band in BANDS:
            s0 = f"sigma_0_{band}"
            tr = f"tr_recomputed_{band}"
            if s0 in df.columns and tr in df.columns:
                exprs.append((pl.col(s0) / (cfg.bond_k1 * pl.col(tr) + cfg.bond_k2)).alias(f"sigma_b1999_{band}"))
    elif cfg.loading_correction == "virkkula2010":
        weights = cfg.virkkula_k or VIRKKULA2010_K
        for band in BANDS:
            s0 = f"sigma_0_{band}"
            tr = f"tr_recomputed_{band}"
            if s0 in df.columns and tr in df.columns and band in weights:
                k0, k1 = weights[band]
                exprs.append(((k0 - k1 * pl.col(tr).log()) * pl.col(s0)).alias(f"sigma_v2010_{band}"))
    else:
        raise ValueError(f"Unsupported loading_correction: {cfg.loading_correction!r}")
    return df.with_columns(exprs) if exprs else df


def compute_psap_absorption(df: pl.DataFrame, spot_area_mm2: float = 17.81,
                            resolution: Literal["1m", "1h"] = "1h",
                            config: AbsorptionConfig | None = None) -> pl.DataFrame:
    """Compute counter-derived Tr and Beer-Lambert absorption at 1h (default) or 1m."""
    if resolution not in {"1m", "1h"}:
        raise ValueError("resolution must be '1m' or '1h'")
    if df.is_empty():
        return df
    cfg = config or AbsorptionConfig(spot_area_mm2=spot_area_mm2)
    if "timestamp" not in df.columns or "flow_lpm" not in df.columns:
        raise ValueError("Input requires timestamp and flow_lpm columns")
    for band in BANDS:
        for column in (f"{band}_ad_sample", f"{band}_ad_ref", f"tr_{band}"):
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")
    df = _add_spot_tracking(df.sort("timestamp"), cfg)
    df = _prepare_counter_data(df, cfg)
    df = _recompute_transmittance(df)
    spot_area_m2 = cfg.spot_area_mm2 * 1e-6
    if resolution == "1h":
        result = _add_sigma_1h(df, spot_area_m2, cfg)
        return _apply_loading_correction(result, cfg)
    result = _add_sigma_1m(df, spot_area_m2, cfg)
    if "sigma_0_green" in result.columns:
        spike = (pl.col("sigma_0_green").abs() > 10. * pl.col("sigma_0_green").rolling_mean(window_size=10, min_samples=3).over("spot_id").abs()).fill_null(False)
        result = result.with_columns(spike.alias("spike_10m"))
    return _apply_loading_correction(result, cfg)


def process_l1_to_l1_5(df_l1: pl.DataFrame, absorption_config: AbsorptionConfig | None = None,
                       flow_config=None, output_path=None,
                       resolution: Literal["1m", "1h"] = "1h",
                       spot_area_mm2: float | None = None) -> pl.DataFrame:
    """Create optical L1.5 and optional flow-corrected fields."""
    if absorption_config is None and spot_area_mm2 is not None:
        absorption_config = AbsorptionConfig(spot_area_mm2=spot_area_mm2)
    result = compute_psap_absorption(df_l1, resolution=resolution, config=absorption_config)
    if flow_config is not None:
        from .calibration import apply_flow_calibration
        result = apply_flow_calibration(result, flow_config)
    _save(result, output_path)
    return result
