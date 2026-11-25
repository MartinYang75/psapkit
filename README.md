# psapkit

`psapkit` processes raw Hex data from three-wavelength Particle Soot Absorption Photometers (PSAP) into L2 datasets. Built on Polars, it handles hex packet decoding (L0–L1), transmittance ($T_r$), absorption coefficient ($\sigma$) re-calculation (L1–L1.5), and multi-stage QA/QC flagging (L1.5–L2).

## Install

```bash
pip install psapkit

```

## Quick start

```python
from psapkit import (
    AbsorptionConfig,
    FlowCalibrationConfig,
    ScreeningConfig,
    process_raw_to_l1,
    process_l1_to_l1_5,
    process_l1_5_to_l2,
)

l1 = process_raw_to_l1("data/raw", output_path="out/l1.parquet")
l1_5 = process_l1_to_l1_5(
    l1,
    absorption_config=AbsorptionConfig.dome_c_default(),
    flow_config=FlowCalibrationConfig.dome_c_default(),
    output_path="out/l1_5.parquet",
)
l2 = process_l1_5_to_l2(
    l1_5,
    config=ScreeningConfig(wd_contam_sector=(40, 150)),
    output_path="out/l2.parquet",
)

```

Or run all stages at once:

```python
from psapkit import run_psap_workflow

result = run_psap_workflow(
    "data/raw",
    meteo_df=None,
    output_dir="out",
    save_intermediates=True,
    file_format="parquet",
)
print(result.l2.head())
print(result.spot_summary)

```

> **Note:**
> - **Filter spot area (`spot_area_mm2`):** Defaults to the nominal PSAP spot area of `17.81 mm²` (4.76 mm diameter). For Dome C, use `AbsorptionConfig.dome_c_default()` (`18.9 mm²`, 4.9 mm diameter; Virkkula et al., 2022) or pass a custom `spot_area_mm2` value.
> - **Clean-filter baseline ($r_0$ at $t_0$):** Rather than relying on a single noisy minute immediately after a filter change, each spot's baseline ratio $r_0$ is phase-locked to the firmware transmittance using the median of $(\Delta I_{\mathrm{sample}} / \Delta I_{\mathrm{ref}}) / T_{r,\mathrm{firmware}}$ across stable minutes ($0.70 \le T_r \le 1.02$, $|\Delta T_r| \le 0.002$) within that `spot_id`.
> - **Temporal resolution (`resolution`):** Defaults to `"1h"`, implementing the $\Delta t = 60\ \mathrm{min}$ detector-count integration method of Virkkula et al. (2022, Eq. 10) to avoid 1-second/1-minute digitization noise at low absorption levels. All valid modulo-$2^{40}$ A/D counter deltas within each clock hour (`closed="left"`, requiring $\ge 30$ valid minutes) are accumulated, and $\sigma_0$ is computed from the logarithmic attenuation between consecutive hourly bins ($\Delta t = 60\ \mathrm{min}$) using their mean flow rate. Pass `resolution="1m"` if 1-minute output is needed.
> 

## Station-specific calibration and screening

`AbsorptionConfig.dome_c_default()` and `FlowCalibrationConfig.dome_c_default()` supply the spot area and three operational flow-regression periods used for the Dome C preset. For another site, pass a custom `spot_area_mm2`, provide `FlowSegment` objects (timestamp, slope, intercept, and optional time dependence), set a constant factor, or disable flow correction with `FlowCalibrationConfig.disabled()`.

`ScreeningConfig` exposes flow and transmittance thresholds, wind-contamination sector and calm-wind threshold, manual date-range blacklists (CSV or a Polars DataFrame with `start_time` and `end_time`), and transient-screening controls. With no meteorology input, wind contamination is explicitly flagged `False`.

## Stages and data

* Raw -> L1: `parse_raw_file` / `process_raw_to_l1`; supports one file, a directory, and readable file-like objects.
* L1 -> L1.5: `compute_psap_absorption` / `process_l1_to_l1_5`; provides `spot_id`, `tr_recomputed_*`, `sigma_0_*`, optional filter-loading correction, and hourly (default) or 1-minute resolution.
* L1.5 -> L2: `process_l1_5_to_l2`; adds QA flags without removing any numeric values.

Optional stage outputs accept `.parquet` and `.csv` paths. The optional meteorology module includes UV wind decomposition and hourly aggregation.

## References

- Bond, T. C., Anderson, T. L., and Campbell, D. (1999). Calibration and intercomparison of filter-based measurements of visible light absorption by aerosols. *Aerosol Science and Technology*, 30, 582–600. https://doi.org/10.1080/027868299304435
- Virkkula, A., Ahlquist, N. C., Covert, D. S., Arnott, W. P., Sheridan, P. J., Quinn, P. K., and Coffman, D. J. (2005). Modification, calibration and a field test of an instrument for measuring light absorption by particles. *Aerosol Science and Technology*, 39(1), 68–83. https://doi.org/10.1080/027868290901963
- Virkkula, A. (2010). Correction of the calibration of the 3-wavelength particle soot absorption photometer (3λ PSAP). *Aerosol Science and Technology*, 44(8), 706–712. https://doi.org/10.1080/02786826.2010.482110
- Virkkula, A., et al. (2022). Aerosol optical properties calculated from size distributions, filter samples and absorption photometer data at Dome C, Antarctica, and their relationships with seasonal cycles of sources. *Atmospheric Chemistry and Physics*, 22(7), 5033–5069.

## License

MIT. See `LICENSE`.
