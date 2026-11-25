"""Customizable piecewise PSAP flow calibration."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from .pipeline import _save


@dataclass(frozen=True)
class FlowSegment:
    start_time: datetime
    slope: float
    intercept: float
    slope_per_day: float = 0.0
    intercept_per_day: float = 0.0
    reference_time: datetime = datetime(2007, 9, 29)


@dataclass(frozen=True)
class FlowCalibrationConfig:
    segments: tuple[FlowSegment, ...] = field(default_factory=tuple)
    factor: float | None = None
    enabled: bool = True
    regime: str = "gt05"

    @classmethod
    def disabled(cls) -> "FlowCalibrationConfig":
        return cls(enabled=False)

    @classmethod
    def dome_c_default(cls, regime: str = "gt05") -> "FlowCalibrationConfig":
        if regime not in {"gt05", "all"}:
            raise ValueError("regime must be 'gt05' or 'all'")
        if regime == "gt05":
            coefficients = (
                (.996780, .254151, -8.5015e-5, -1.0025e-5),
                (.266544, .449459, 1.7495e-4, -6.8862e-5),
                (.798455, .149982, 2.4040e-5, -2.5906e-5),
            )
        else:
            coefficients = (
                (1.021644, .206938, -7.2304e-5, -2.4512e-5),
                (.411895, .226463, 1.4566e-4, -2.3332e-5),
                (.850785, .053312, 1.7741e-5, -1.2932e-5),
            )
        starts = (datetime(2007, 9, 29), datetime(2014, 1, 1), datetime(2018, 12, 8))
        segments = tuple(
            FlowSegment(start, slope, intercept, slope_day, intercept_day)
            for start, (slope, intercept, slope_day, intercept_day) in zip(starts, coefficients)
        )
        return cls(segments=segments, regime=regime)


def apply_flow_calibration(df: pl.DataFrame, config: FlowCalibrationConfig) -> pl.DataFrame:
    if df.is_empty() and ("flow_lpm" not in df.columns or "timestamp" not in df.columns):
        return df.with_columns(
            pl.Series("q_corrected", [], dtype=pl.Float64),
            pl.Series("fq", [], dtype=pl.Float64),
        )
    if "flow_lpm" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("Flow calibration requires timestamp and flow_lpm")
    if not config.enabled:
        fq = pl.lit(1.0)
        qtrue = pl.col("flow_lpm").cast(pl.Float64)
    elif config.factor is not None:
        fq = pl.lit(float(config.factor))
        qtrue = pl.col("flow_lpm").cast(pl.Float64) / fq
    elif config.segments:
        expr = None
        for segment in sorted(config.segments, key=lambda s: s.start_time, reverse=True):
            days = (pl.col("timestamp") - pl.lit(segment.reference_time)).dt.total_seconds() / 86400
            slope = segment.slope + segment.slope_per_day * days
            intercept = segment.intercept + segment.intercept_per_day * days
            value = slope * pl.col("flow_lpm") + intercept
            condition = pl.col("timestamp") >= pl.lit(segment.start_time)
            expr = pl.when(condition).then(value) if expr is None else expr.when(condition).then(value)
        qtrue = expr.otherwise(pl.col("flow_lpm").cast(pl.Float64))
        fq = pl.when(qtrue > 0).then(pl.col("flow_lpm") / qtrue).otherwise(1.0)
    else:
        raise ValueError("Enabled flow calibration requires segments or factor")
    result = df.with_columns(
        qtrue.alias("q_corrected"),
        qtrue.alias("flow_lpm_q_corrected"),
        fq.alias("fq"),
    )
    for name in list(result.columns):
        if name.startswith("sigma_") and not name.endswith("_q_corrected"):
            result = result.with_columns((pl.col(name) * pl.col("fq")).alias(f"{name}_q_corrected"))
    return result


def process_l1_to_l1_5(df_l1: pl.DataFrame, absorption_config=None,
                       flow_config: FlowCalibrationConfig | None = None, output_path=None,
                       resolution: str = "1h", spot_area_mm2: float | None = None) -> pl.DataFrame:
    from .absorption import process_l1_to_l1_5 as _process_l1_to_l1_5
    return _process_l1_to_l1_5(
        df_l1,
        absorption_config=absorption_config,
        flow_config=flow_config,
        output_path=output_path,
        resolution=resolution,
        spot_area_mm2=spot_area_mm2,
    )
