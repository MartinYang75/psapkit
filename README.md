# psapkit

`psapkit` is a lightweight, station-independent Python toolkit for three-band Particle Soot Absorption Photometer (PSAP) data. It provides pure-Python telemetry decoding, raw `.raw`/`.ra5` ingestion, filter-spot tracking, firmware-anchored transmittance recalculation, Beer–Lambert absorption inversion, configurable flow calibration, and non-destructive quality flags. The core package does not require plotting or tabular-data frameworks such as pandas.

## Install

```bash
pip install psapkit
```

For development and release checks:

```bash
pip install "psapkit[dev]"
```

## Quick start

```python
from psapkit import (
    FlowCalibrationConfig, ScreeningConfig,
    process_raw_to_l1, process_l1_to_l1_5, process_l1_5_to_l2,
)

l1 = process_raw_to_l1("data/raw", output_path="out/l1.parquet")
l1_5 = process_l1_to_l1_5(
    l1, flow_config=FlowCalibrationConfig.dome_c_default(),
    resolution="1h", output_path="out/l1_5.parquet",
)
l2 = process_l1_5_to_l2(
    l1_5, config=ScreeningConfig(wd_contam_sector=(40, 150)),
    output_path="out/l2.parquet",
)
```

Or run all stages at once:

```python
from psapkit import run_psap_workflow

result = run_psap_workflow(
    "data/raw", meteo_df=None, output_dir="out",
    save_intermediates=True, file_format="parquet",
    resolution="1h",
)
print(result.l2.head())
print(result.spot_summary)
```

> **Note:** `process_l1_to_l1_5` and `run_psap_workflow` default to `resolution="1m"` for minute-level optical inversion. Because `process_l1_5_to_l2` evaluates its multi-scale Hampel and Isolation Forest transient screening on a 1-hour grid (`13h` / `73h` / `361h` windows), pass `resolution="1h"` before running hourly L2 quality control.

## Station-specific calibration and screening

`FlowCalibrationConfig.dome_c_default()` supplies the three operational regression periods used for the Dome C preset. For another site, provide `FlowSegment` objects (timestamp, slope, intercept, and optional time dependence), set a constant factor, or disable flow correction with `FlowCalibrationConfig.disabled()`.

`ScreeningConfig` exposes flow and transmittance thresholds, wind-contamination sector and calm-wind threshold, manual date-range blacklists (CSV or a Polars DataFrame with `start_time` and `end_time`), and transient-screening controls. With no meteorology input, wind contamination is explicitly flagged `False`. L2 preserves numeric data and records screening outcomes in Boolean flag columns; `flag_clean_background` is an explicit convenience flag, not a deletion operation.

## Stages and data

- Raw -> L1: `parse_raw_file` / `process_raw_to_l1`; supports one file, a directory, and readable file-like objects.
- L1 -> L1.5: `compute_psap_absorption` / `process_l1_to_l1_5`; provides `spot_id`, `tr_recomputed_*`, `sigma_0_*`, optional filter-loading correction, and 1-minute or hourly resolution.
- L1.5 -> L2: `process_l1_5_to_l2`; adds auditable QA flags without removing rows or numeric columns.

Optional stage outputs accept `.parquet` and `.csv` paths. The optional meteorology module includes UV wind decomposition and hourly aggregation.

## Scientific references

- Bond, T. C., Anderson, T. L., and Campbell, D. (1999). Calibration and intercomparison of filter-based measurements of visible light absorption by aerosols. *Aerosol Science and Technology*, 30, 582–600. https://doi.org/10.1080/027868299304435
- Virkkula, A. (2010). Correction of the calibration of the 3-wavelength particle soot absorption photometer (3λ PSAP). *Aerosol Science and Technology*, 44, 706–712. https://doi.org/10.1080/02786826.2010.482110
- Virkkula, A. et al. (2022). Aerosol light absorption at Dome C, Antarctica. *Atmospheric Chemistry and Physics*. (Use the journal record for final citation metadata.)

## License

MIT. See `LICENSE`.
