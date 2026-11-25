"""End-to-end PSAP workflow."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .absorption import AbsorptionConfig
from .calibration import FlowCalibrationConfig, process_l1_to_l1_5
from .pipeline import process_raw_to_l1
from .screening import ScreeningConfig, process_l1_5_to_l2


@dataclass
class WorkflowResult:
    l1: pl.DataFrame
    l1_5: pl.DataFrame
    l2: pl.DataFrame
    spot_summary: pl.DataFrame


def _spot_summary(df: pl.DataFrame) -> pl.DataFrame:
    if "spot_id" not in df.columns:
        return pl.DataFrame()
    aggs = [pl.col("timestamp").min().alias("start_time"), pl.col("timestamp").max().alias("end_time"), pl.len().alias("n_minutes")]
    if "tr_recomputed_green" in df.columns:
        aggs.extend([pl.col("tr_recomputed_green").drop_nulls().first().alias("tr_start_green"), pl.col("tr_recomputed_green").drop_nulls().last().alias("tr_end_green")])
    return df.group_by("spot_id").agg(aggs).sort("spot_id")


def run_psap_workflow(raw_input, meteo_df: pl.DataFrame | None = None,
                      absorption_config: AbsorptionConfig | None = None,
                      flow_config: FlowCalibrationConfig | None = None,
                      screening_config: ScreeningConfig | None = None,
                      output_dir: str | Path | None = None,
                      save_intermediates: bool = False,
                      file_format: str = "parquet",
                      resolution: str = "1h",
                      spot_area_mm2: float | None = None) -> WorkflowResult:
    """Process raw telemetry to L2, optionally exporting every stage."""
    if file_format not in {"parquet", "csv"}:
        raise ValueError("file_format must be 'parquet' or 'csv'")
    out = Path(output_dir) if output_dir is not None else None
    def output(name: str):
        return out / f"{name}.{file_format}" if save_intermediates and out is not None else None
    l1 = process_raw_to_l1(raw_input, output_path=output("l1"))
    l1_5 = process_l1_to_l1_5(l1, absorption_config=absorption_config,
                              flow_config=flow_config or FlowCalibrationConfig.dome_c_default(),
                              output_path=output("l1_5"),
                              resolution=resolution,
                              spot_area_mm2=spot_area_mm2)
    l2 = process_l1_5_to_l2(l1_5, meteo_df=meteo_df, config=screening_config,
                            output_path=output("l2"))
    return WorkflowResult(l1=l1, l1_5=l1_5, l2=l2, spot_summary=_spot_summary(l1_5))
