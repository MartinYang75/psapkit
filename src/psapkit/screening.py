"""Non-destructive L1.5 to L2 quality flags and transient detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest

from .pipeline import _save


@dataclass(frozen=True)
class ScreeningConfig:
    min_flow_lpm: float = .5
    tr_low_threshold: float = .70
    wd_contam_sector: tuple[float, float] = (20., 110.)
    min_wind_speed: float = 2.
    manual_blacklist: str | Path | pl.DataFrame | None = None
    enable_transient_screening: bool = True
    hampel_windows: tuple[int, ...] = (13, 73, 361)
    hampel_min_samples: tuple[int, ...] = (3, 12, 36)
    hampel_sigma: float = 3.0
    local_hampel_threshold: float = 3.5
    local_turnaround_threshold: float = 2.0
    local_iforest_threshold: float = .65
    extreme_local_threshold: float = 4.5
    synoptic_threshold: float = 6.0
    synoptic_local_threshold: float = 2.5
    oscillation_threshold: float = 3.5
    oscillation_synoptic_threshold: float = 1.8
    noise_floor: float = .002
    monthly_mad_cap_factor: float = 1.2
    monthly_sparse_count: int = 120
    synoptic_min_count: int = 36
    monthly_mad_min_count: int = 36
    iforest_estimators: int = 200
    iforest_contamination: float = .04
    iforest_random_state: int = 42
    iforest_n_jobs: int = -1
    iforest_min_samples: int = 8
    iforest_score_threshold: float = .65
    gap_window: int = 6
    gap_min_contaminated: int = 4
    gap_proximity_window: int = 19
    plume_seed_z: float = 3.5
    plume_seed_absorption: float = .032
    plume_primary_z: float = 2.8
    plume_synoptic_threshold: float = 2.2
    plume_long_threshold: float = 2.5
    plume_max_gap: int = 3
    enable_flow_screening: bool = True
    enable_tr_screening: bool = False
    enable_wind_screening: bool = True

    def __post_init__(self):
        if len(self.hampel_windows) != 3 or len(self.hampel_min_samples) != 3:
            raise ValueError("hampel_windows and hampel_min_samples must have three entries")
        if any(w < 1 for w in self.hampel_windows) or any(m < 1 for m in self.hampel_min_samples):
            raise ValueError("Hampel windows and minimum sample counts must be positive")
        if not 0 < self.iforest_contamination <= .5:
            raise ValueError("iforest_contamination must be in (0, 0.5]")


def _rolling_exprs(config: ScreeningConfig) -> list[pl.Expr]:
    (w13, w73, w361), (n13, n73, n361) = config.hampel_windows, config.hampel_min_samples
    base = pl.col("s_base")
    med13 = base.rolling_median(window_size=w13, min_samples=n13, center=True)
    med73 = base.rolling_median(window_size=w73, min_samples=n73, center=True)
    med361 = base.rolling_median(window_size=w361, min_samples=n361, center=True)
    mad13 = (base - med13).abs().rolling_median(window_size=w13, min_samples=n13, center=True)
    mad73 = (base - med73).abs().rolling_median(window_size=w73, min_samples=n73, center=True)
    mad361 = (base - med361).abs().rolling_median(window_size=w361, min_samples=n361, center=True)
    return [
        med13.alias("med_13h"), mad13.alias("mad_13h"),
        med73.alias("_med_73h_raw"), mad73.alias("_mad_73h_raw"),
        med361.alias("med_361h"), mad361.alias("mad_361h"),
        base.is_not_null().cast(pl.Int32).rolling_sum(window_size=w73, min_samples=1, center=True).alias("n_valid_73h"),
        base.is_not_null().cast(pl.Int32).rolling_sum(window_size=w361, min_samples=1, center=True).alias("n_valid_361h"),
    ]


def detect_rapid_transients(df: pl.DataFrame, config: ScreeningConfig | None = None,
                            signal_col: str | None = None,
                            candidate_col: str = "flag_base_candidate") -> pl.DataFrame:
    """Run multi-scale Hampel, Isolation Forest and bidirectional plume propagation.

    Missing hours are represented on a continuous hourly grid, so window sizes
    always describe elapsed hours rather than only observed samples.
    """
    cfg = config or ScreeningConfig()
    if df.is_empty():
        return df.with_columns(pl.Series("flag_rapid_transient", [], dtype=pl.Boolean))
    if "timestamp" not in df.columns:
        raise ValueError("Transient screening requires timestamp")
    signal = signal_col or next((c for c in (
        "sigma_stp_green", "sigma_b1999_green_q_corrected",
        "sigma_0_green_q_corrected", "sigma_0_green",
    ) if c in df.columns), None)
    if signal is None:
        raise ValueError("Transient screening requires a green absorption signal")
    if candidate_col not in df.columns:
        source = pl.col(signal).is_not_null().is_finite()
        df = df.with_columns(source.alias(candidate_col))

    source = df.select("timestamp", pl.col(signal).cast(pl.Float64).alias("_signal"), pl.col(candidate_col).fill_null(False).alias("_candidate"))
    source = source.group_by("timestamp").agg(pl.col("_signal").mean(), pl.col("_candidate").any()).sort("timestamp")
    start = source["timestamp"].min().replace(minute=0, second=0, microsecond=0)
    end = source["timestamp"].max().replace(minute=0, second=0, microsecond=0)
    grid = pl.DataFrame({"timestamp": pl.datetime_range(start, end, interval="1h", eager=True)})
    grid = grid.join(source, on="timestamp", how="left").sort("timestamp").with_columns(
        pl.when(pl.col("_candidate").fill_null(False)).then(pl.col("_signal")).otherwise(None).alias("s_base")
    ).with_columns(_rolling_exprs(cfg))

    grid = grid.with_columns(pl.col("timestamp").dt.month().alias("_month"))
    monthly = grid.filter(pl.col("_candidate").fill_null(False)).group_by("_month").agg([
        pl.col("_signal").median().alias("_mon_med"),
        (pl.col("_signal") - pl.col("_signal").median()).abs().median().alias("_mon_mad"),
        pl.col("_signal").count().alias("_mon_n"),
    ])
    grid = grid.join(monthly, on="_month", how="left").with_columns([
        pl.coalesce("med_361h", "_mon_med").alias("med_361h"),
        pl.coalesce("mad_361h", "_mon_mad").alias("mad_361h"),
    ]).with_columns(
        pl.when(pl.col("n_valid_361h") < cfg.monthly_sparse_count)
        .then(pl.min_horizontal(pl.col("mad_361h"), cfg.monthly_mad_cap_factor * pl.col("_mon_mad")))
        .otherwise(pl.col("mad_361h")).alias("mad_361h")
    ).with_columns([
        pl.when(pl.col("n_valid_73h") >= cfg.synoptic_min_count)
        .then(pl.min_horizontal(pl.col("_med_73h_raw"), pl.col("med_361h") * 1.5))
        .otherwise(pl.col("med_361h")).alias("med_73h"),
        pl.when(pl.col("n_valid_73h") >= cfg.synoptic_min_count)
        .then(pl.col("_mad_73h_raw")).otherwise(pl.col("mad_361h")).alias("mad_73h"),
    ]).with_columns([
        pl.coalesce("med_73h", "_mon_med").alias("med_73h"),
        pl.coalesce("mad_73h", "_mon_mad").alias("mad_73h"),
        pl.max_horizontal(1.4826 * pl.min_horizontal(pl.col("mad_13h"), pl.col("mad_73h")), cfg.noise_floor).alias("scale_local"),
        pl.max_horizontal(1.4826 * pl.col("mad_73h"), cfg.noise_floor).alias("scale_synoptic"),
        pl.max_horizontal(1.4826 * pl.col("mad_361h"), cfg.noise_floor).alias("scale_361h"),
    ])

    base = pl.col("s_base")
    prev = base.shift(1).fill_null(base.shift(2)).fill_null(base.shift(3))
    nxt = base.shift(-1).fill_null(base.shift(-2)).fill_null(base.shift(-3))
    grid = grid.with_columns([
        ((base - pl.col("med_13h")) / pl.col("scale_local")).alias("hampel_z_local"),
        ((base - pl.col("med_73h")) / pl.col("scale_synoptic")).alias("hampel_z_synoptic"),
        ((base - pl.col("med_361h")) / pl.col("scale_361h")).alias("hampel_z_361h"),
        ((base - .5 * (prev + nxt)) / pl.col("scale_local")).alias("turnaround_z"),
        pl.max_horizontal((base-prev).abs()/pl.col("scale_synoptic"), (base-nxt).abs()/pl.col("scale_synoptic")).alias("oscillation_z"),
    ])
    contaminated = (~pl.col("_candidate").fill_null(True)).cast(pl.Int32)
    grid = grid.with_columns((contaminated.rolling_sum(window_size=cfg.gap_window, min_samples=1, center=True) >= cfg.gap_min_contaminated).alias("_gap_seed"))
    grid = grid.with_columns(pl.col("_gap_seed").cast(pl.Int32).rolling_max(window_size=cfg.gap_proximity_window, min_samples=1, center=True).cast(pl.Boolean).alias("near_contam_24h"))

    feature_cols = ["hampel_z_local", "hampel_z_synoptic", "step_prev_z", "step_next_z", "turnaround_z"]
    grid = grid.with_columns([
        ((base-prev).abs()/pl.col("scale_synoptic")).alias("step_prev_z"),
        ((base-nxt).abs()/pl.col("scale_synoptic")).alias("step_next_z"),
    ])
    candidate_rows = grid.filter(pl.col("_candidate").fill_null(False))
    grid = grid.with_columns(pl.lit(None, dtype=pl.Float64).alias("iforest_prob"))
    if candidate_rows.height >= max(2, cfg.iforest_min_samples):
        train = candidate_rows.select(feature_cols).fill_nan(0).fill_null(0).to_numpy()
        fit = IsolationForest(n_estimators=cfg.iforest_estimators, contamination=cfg.iforest_contamination,
                              random_state=cfg.iforest_random_state, n_jobs=cfg.iforest_n_jobs)
        fit.fit(train)
        valid = grid.filter(pl.col("hampel_z_local").is_not_null())
        all_features = valid.select(feature_cols).fill_nan(0).fill_null(0).to_numpy()
        scores = fit.decision_function(all_features)
        low, high = float(scores.min()), float(scores.max())
        probability = (high - scores) / (high - low) if high > low else np.zeros_like(scores)
        probability = np.clip(probability, 0., 1.)
        prob_values = np.full(grid.height, np.nan)
        positions = grid.with_row_index("_row").filter(pl.col("hampel_z_local").is_not_null())["_row"].to_numpy()
        prob_values[positions] = probability
        grid = grid.with_columns(pl.Series("iforest_prob", prob_values, dtype=pl.Float64))

    cond1 = (pl.col("hampel_z_local") > cfg.local_hampel_threshold) & ((pl.col("turnaround_z") > cfg.local_turnaround_threshold) | (pl.col("iforest_prob") > cfg.local_iforest_threshold))
    cond2 = (pl.col("hampel_z_local") > cfg.extreme_local_threshold) | ((pl.col("hampel_z_synoptic") > cfg.synoptic_threshold) & (pl.col("hampel_z_local") > cfg.synoptic_local_threshold))
    cond3 = (pl.col("oscillation_z") > cfg.oscillation_threshold) & (pl.col("hampel_z_synoptic") > cfg.oscillation_synoptic_threshold)
    candidate = pl.col("_candidate").fill_null(False)
    primary = ((cond1 | cond2 | cond3) & candidate).fill_null(False)
    seed = (pl.col("near_contam_24h") & (pl.col("hampel_z_361h") > cfg.plume_seed_z) & (base > cfg.plume_seed_absorption) & candidate).fill_null(False)
    grid = grid.with_columns([primary.alias("_primary"), seed.alias("_seed")])

    primary_np = grid["_primary"].to_numpy().astype(bool)
    seed_np = (grid["_seed"] | (grid["_primary"] & grid["near_contam_24h"] & (grid["hampel_z_361h"] > cfg.plume_primary_z))).to_numpy().astype(bool)
    hz_syn = np.nan_to_num(grid["hampel_z_synoptic"].to_numpy(), nan=0.)
    hz_long = np.nan_to_num(grid["hampel_z_361h"].to_numpy(), nan=0.)
    candidate_np = grid["_candidate"].fill_null(False).to_numpy().astype(bool)
    propagated = primary_np | seed_np
    for direction in (1, -1):
        active = False
        gap = 0
        indices = range(grid.height) if direction == 1 else range(grid.height-1, -1, -1)
        for i in indices:
            if seed_np[i]:
                active, gap = True, 0
            if not active:
                continue
            if candidate_np[i]:
                if hz_syn[i] > cfg.plume_synoptic_threshold or hz_long[i] > cfg.plume_long_threshold:
                    propagated[i], gap = True, 0
                else:
                    active, gap = False, 0
            else:
                gap += 1
                if gap > cfg.plume_max_gap:
                    active, gap = False, 0

    diagnostic_cols = ["timestamp", "med_13h", "med_73h", "med_361h", "hampel_z_local", "hampel_z_synoptic", "hampel_z_361h", "turnaround_z", "oscillation_z", "iforest_prob"]
    diag = grid.select(diagnostic_cols).with_columns(pl.Series("flag_rapid_transient", propagated))
    return df.join(diag, on="timestamp", how="left").with_columns(pl.col("flag_rapid_transient").fill_null(False)).sort("timestamp")


def process_l1_5_to_l2(df_l1_5: pl.DataFrame, meteo_df: pl.DataFrame | None = None,
                       config: ScreeningConfig | None = None, output_path=None) -> pl.DataFrame:
    """Add screening flags while preserving source numeric values and rows."""
    cfg = config or ScreeningConfig()
    df = df_l1_5.sort("timestamp")
    if df.is_empty():
        additions = {
            "flag_has_psap": pl.Boolean, "flag_has_meteo": pl.Boolean,
            "flag_low_flow": pl.Boolean, "flag_spike_10m": pl.Boolean,
            "flag_wd_contam": pl.Boolean, "flag_ws_low": pl.Boolean,
            "flag_wind_contam": pl.Boolean, "flag_tr_low": pl.Boolean,
            "flag_non_positive": pl.Boolean, "flag_manual_blacklist": pl.Boolean,
            "flag_base_candidate": pl.Boolean, "flag_rapid_transient": pl.Boolean,
            "flag_transient": pl.Boolean, "flag_clean_background": pl.Boolean,
        }
        missing = {name: dtype for name, dtype in additions.items() if name not in df.columns}
        if missing:
            df = df.with_columns([pl.Series(name, [], dtype=dtype) for name, dtype in missing.items()])
        _save(df, output_path)
        return df

    exprs = []
    if cfg.enable_flow_screening:
        flow_checks = []
        if "flow_lpm" in df.columns:
            flow_checks.extend([
                (pl.col("flow_lpm") < cfg.min_flow_lpm).fill_null(False),
                pl.col("flow_lpm").is_null(),
            ])
        if "q_corrected" in df.columns:
            flow_checks.append((pl.col("q_corrected") < cfg.min_flow_lpm).fill_null(False))
        if "flow_lpm_q_corrected" in df.columns:
            flow_checks.append((pl.col("flow_lpm_q_corrected") < cfg.min_flow_lpm).fill_null(False))
        exprs.append(pl.any_horizontal(flow_checks).fill_null(False).alias("flag_low_flow") if flow_checks else pl.lit(False).alias("flag_low_flow"))
    else:
        exprs.append(pl.lit(False).alias("flag_low_flow"))

    tr_cols = [f"tr_recomputed_{b}" for b in ("blue", "green", "red") if f"tr_recomputed_{b}" in df.columns]
    if tr_cols:
        exprs.append(pl.any_horizontal([pl.col(c) < cfg.tr_low_threshold for c in tr_cols]).fill_null(False).alias("flag_tr_low"))
    else:
        exprs.append(pl.lit(False).alias("flag_tr_low"))

    if meteo_df is not None and cfg.enable_wind_screening:
        met = meteo_df
        if "timestamp" not in met.columns:
            raise ValueError("meteo_df requires timestamp")
        wd = next((c for c in ("wind_direction", "wd", "wd_vec") if c in met.columns), None)
        ws = next((c for c in ("wind_speed", "ws", "ws_vec") if c in met.columns), None)
        if wd is None or ws is None:
            raise ValueError("meteo_df requires wind_direction/wind_speed, wd/ws, or wd_vec/ws_vec")
        low, high = cfg.wd_contam_sector
        wd_contam_expr = (
            pl.col("has_wd_contam_1m").fill_null(False)
            if "has_wd_contam_1m" in met.columns
            else pl.col(wd).is_between(low, high).fill_null(False)
        ).alias("flag_wd_contam")
        ws_low_expr = (pl.col(ws) < cfg.min_wind_speed).fill_null(False).alias("flag_ws_low")
        has_meteo_checks = [pl.col(wd).is_not_null(), pl.col(ws).is_not_null()]
        for opt_col in ("temp", "pressure"):
            if opt_col in met.columns:
                has_meteo_checks.append(pl.col(opt_col).is_not_null())
        has_meteo_expr = pl.all_horizontal(has_meteo_checks).alias("flag_has_meteo")
        select_cols = ["timestamp", wd_contam_expr, ws_low_expr, has_meteo_expr]
        df = df.join(met.select(select_cols), on="timestamp", how="left").with_columns([
            pl.col("flag_wd_contam").fill_null(False),
            pl.col("flag_ws_low").fill_null(False),
            pl.col("flag_has_meteo").fill_null(False),
            (pl.col("flag_wd_contam").fill_null(False) | pl.col("flag_ws_low").fill_null(False)).alias("flag_wind_contam"),
        ])
    else:
        df = df.with_columns([
            pl.lit(False).alias("flag_wd_contam"),
            pl.lit(False).alias("flag_ws_low"),
            pl.lit(False).alias("flag_wind_contam"),
            pl.lit(False).alias("flag_has_meteo"),
        ])

    df = df.with_columns(exprs)
    if cfg.manual_blacklist is not None:
        blacklist = cfg.manual_blacklist if isinstance(cfg.manual_blacklist, pl.DataFrame) else pl.read_csv(cfg.manual_blacklist, try_parse_dates=True)
        if not {"start_time", "end_time"}.issubset(blacklist.columns):
            raise ValueError("manual_blacklist requires start_time and end_time columns")
        if blacklist.height > 0:
            times = df["timestamp"].to_list()
            starts, ends = blacklist["start_time"].to_list(), blacklist["end_time"].to_list()
            is_hourly = df.height >= 2 and (df["timestamp"].diff().drop_nulls().median().total_seconds() >= 3300.0)
            if is_hourly:
                from datetime import timedelta
                flags = [any((ts <= end) and (ts + timedelta(hours=1) >= start) for start, end in zip(starts, ends)) for ts in times]
            else:
                flags = [any(start <= ts <= end for start, end in zip(starts, ends)) for ts in times]
            df = df.with_columns(pl.Series("flag_manual_blacklist", flags, dtype=pl.Boolean))
        else:
            df = df.with_columns(pl.lit(False).alias("flag_manual_blacklist"))
    else:
        df = df.with_columns(pl.lit(False).alias("flag_manual_blacklist"))

    signal = next((c for c in ("sigma_b1999_green_q_corrected", "sigma_stp_green", "sigma_b1999_green", "sigma_0_green_q_corrected", "sigma_0_green") if c in df.columns), None)
    has_psap = pl.all_horizontal([pl.col(c).is_not_null() for c in tr_cols]) if tr_cols else pl.lit(True)
    if signal:
        has_psap = has_psap & pl.col(signal).is_not_null()
    flag_spike_10m = pl.col("spike_10m").fill_null(False) if "spike_10m" in df.columns else pl.lit(False)
    flag_non_positive = (pl.col(signal) <= 0.0).fill_null(False) if signal else pl.lit(False)

    base_candidate = (
        has_psap
        & (~flag_spike_10m)
        & (~pl.col("flag_low_flow"))
        & (~pl.col("flag_wind_contam"))
        & (~pl.col("flag_manual_blacklist"))
    )
    if meteo_df is not None and cfg.enable_wind_screening:
        base_candidate = base_candidate & pl.col("flag_has_meteo")

    df = df.with_columns([
        has_psap.fill_null(False).alias("flag_has_psap"),
        flag_spike_10m.alias("flag_spike_10m"),
        flag_non_positive.alias("flag_non_positive"),
        base_candidate.fill_null(False).alias("flag_base_candidate"),
    ])

    if cfg.enable_transient_screening and signal:
        df = detect_rapid_transients(df, cfg, signal_col=signal)
        transient = pl.col("flag_rapid_transient").fill_null(False)
    else:
        df = df.with_columns(pl.lit(False).alias("flag_rapid_transient"))
        transient = pl.lit(False)
    df = df.with_columns(transient.alias("flag_transient"))
    clean = pl.col("flag_base_candidate") & (~pl.col("flag_transient"))
    if cfg.enable_tr_screening:
        clean = clean & (~pl.col("flag_tr_low"))
    df = df.with_columns(clean.fill_null(False).alias("flag_clean_background"))
    _save(df, output_path)
    return df
