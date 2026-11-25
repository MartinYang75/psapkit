"""Optional wind-vector decomposition and hourly meteorology aggregation."""
from __future__ import annotations

import polars as pl


def compute_uv_wind(df: pl.DataFrame, speed_col: str = "wind_speed", direction_col: str = "wind_direction") -> pl.DataFrame:
    """Add eastward (u) and northward (v) wind components from meteorological direction."""
    if speed_col not in df.columns or direction_col not in df.columns:
        raise ValueError("wind speed and direction columns are required")
    direction = pl.col(direction_col).cast(pl.Float64) * (3.141592653589793 / 180.)
    return df.with_columns((-pl.col(speed_col) * direction.sin()).alias("u_wind"), (-pl.col(speed_col) * direction.cos()).alias("v_wind"))


def aggregate_meteo_hourly(df: pl.DataFrame, timestamp_col: str = "timestamp") -> pl.DataFrame:
    """Aggregate numeric meteo fields to left-labeled hourly means."""
    values = [c for c, dtype in zip(df.columns, df.dtypes) if c != timestamp_col and dtype.is_numeric()]
    if not values:
        raise ValueError("No numeric meteorological columns to aggregate")
    result = df.sort(timestamp_col).group_by_dynamic(timestamp_col, every="1h", closed="left").agg([pl.col(c).mean() for c in values])
    if "u_wind" in result.columns and "v_wind" in result.columns:
        result = result.with_columns((pl.col("u_wind").pow(2) + pl.col("v_wind").pow(2)).sqrt().alias("wind_speed"), ((pl.arctan2(-pl.col("u_wind"), -pl.col("v_wind"))) * 180 / 3.141592653589793 % 360).alias("wind_direction"))
    return result
