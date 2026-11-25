"""Step-by-step and end-to-end psapkit demo; provide local raw files first."""
from pathlib import Path
import sys

import polars as pl

from psapkit import (
    AbsorptionConfig, FlowCalibrationConfig, ScreeningConfig,
    compute_psap_absorption, decode_psap_hex, process_l1_5_to_l2,
    process_l1_to_l1_5, process_raw_to_l1, run_psap_workflow,
)

ROOT = Path(__file__).parent
RAW = ROOT / "sample_data"
OUT = ROOT / "output"


def main() -> None:
    raw_files = sorted(
        p for p in RAW.iterdir()
        if p.is_file() and p.suffix.lower() in {".raw", ".ra5"}
    ) if RAW.exists() else []
    if not raw_files:
        print(f"Place the three PSAP .raw/.ra5 files in {RAW} and rerun this demo.")
        return
    if len(raw_files) != 3:
        print(f"Found {len(raw_files)} raw files; proceeding with all available .raw/.ra5 files.")

    # 1. Generic low-level APIs. Run the packet decoder on a representative
    # packet, then exercise the optical API on real data produced below.
    print(decode_psap_hex("01,100,200,10,0"))

    # 2. Explicit Raw -> L1 -> L1.5 -> L2 stages, saving each intermediate.
    l1 = process_raw_to_l1(RAW, output_path=OUT / "l1.parquet")
    l1_5 = process_l1_to_l1_5(
        l1,
        absorption_config=AbsorptionConfig(loading_correction="bond1999"),
        flow_config=FlowCalibrationConfig.disabled(),
        output_path=OUT / "l1_5.parquet",
    )
    # Use an empty, correctly typed blacklist template. Replace/add intervals
    # here with site diary events if available.
    blacklist = pl.DataFrame(schema={
        "start_time": pl.Datetime("us"), "end_time": pl.Datetime("us")
    })
    site_config = ScreeningConfig(
        wd_contam_sector=(35.0, 145.0),
        manual_blacklist=blacklist,
    )
    l2 = process_l1_5_to_l2(l1_5, config=site_config, output_path=OUT / "l2.parquet")
    print("Stepwise shapes:", l1.shape, l1_5.shape, l2.shape)

    # 3. One-call workflow at 1h resolution (matches multi-year hourly L2 pipeline).
    result = run_psap_workflow(
        RAW, flow_config=FlowCalibrationConfig.dome_c_default(),
        screening_config=site_config, output_dir=OUT / "workflow",
        save_intermediates=True, resolution="1h",
    )
    print("Workflow shapes:", result.l1.shape, result.l1_5.shape, result.l2.shape)
    print(result.spot_summary)


if __name__ == "__main__":
    main()
