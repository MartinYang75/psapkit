"""Raw PSAP ingestion and one-minute cycle assembly."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import IO

import polars as pl

from .decoder import decode_psap_hex, validate_decoded

# This field-anchored expression mirrors the logger record layout used by the
# original converter: timestamp, instrument clock, seven numeric channels,
# three logger metadata tokens, then the quoted packet payload.
LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})"
    r"\s+\S+"
    r"\s+([-+.\deE]+)\s+([-+.\deE]+)\s+([-+.\deE]+)"
    r"\s+([-+.\deE]+)\s+([-+.\deE]+)\s+([-+.\deE]+)"
    r"\s+([-+.\deE]+)"
    r"(?:\s+\S+){3}"
    r'\s+"([^\"]*"?)'
)


def _read_text(source: str | Path | IO[str] | IO[bytes]) -> tuple[str, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.read_text(encoding="latin-1", errors="replace"), path.suffix.lower()
    data = source.read()
    if isinstance(data, bytes):
        data = data.decode("latin-1", errors="replace")
    return data, ".raw"


def _parse_rows(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        match = LINE_RE.match(line.strip())
        if match is None:
            continue
        try:
            timestamp = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H:%M:%S")
            numeric = [float(match.group(i)) for i in range(3, 10)]
            payload = match.group(10).rstrip('"').strip()
            if "," not in payload:
                continue
            decoded = decode_psap_hex(payload)
        except (ValueError, IndexError):
            continue
        rows.append({
            "timestamp": timestamp, "abs_blue": numeric[0], "abs_green": numeric[1],
            "abs_red": numeric[2], "tr_blue": numeric[3], "tr_green": numeric[4],
            "tr_red": numeric[5], "flow_lpm": numeric[6], "decoded": decoded,
            "valid": not validate_decoded(decoded),
        })
    return rows


def _assemble(rows: list[dict], drifted: bool = False) -> pl.DataFrame:
    groups: list[list[dict]] = []
    if drifted:
        # 2014-era .ra5: logger-side timestamp gaps delimit asynchronous blocks.
        for row in rows:
            if not groups or (row["timestamp"] - groups[-1][-1]["timestamp"]).total_seconds() > 7:
                groups.append([row])
            else:
                groups[-1].append(row)
    else:
        # Standard 2008-2013 and continuous-era records retain the 00..04 anchor.
        index = 0
        while index + 4 < len(rows):
            window = rows[index:index + 5]
            if window[0]["decoded"]["cycle_second"] == 0 and window[-1]["decoded"]["cycle_second"] == 4:
                span = (window[-1]["timestamp"] - window[0]["timestamp"]).total_seconds()
                if span <= 6:
                    groups.append(window)
                    index += 5
                    continue
            index += 1
    records = []
    for group in groups:
        if len(group) < 4:
            continue
        record = {key: None for key in (
            "volume_sum", "seconds_since_E", "blue_ad_sample", "blue_ad_ref", "blue_n_meas", "blue_n_overflow",
            "green_ad_sample", "green_ad_ref", "green_n_meas", "green_n_overflow", "red_ad_sample", "red_ad_ref",
            "red_n_meas", "red_n_overflow", "dark_ad_sample", "dark_ad_ref", "dark_seconds_since_start",
            "dark_realtime_sample_current", "dark_realtime_ref_current",
        )}
        for row in group:
            record.update({k: v for k, v in row["decoded"].items() if k not in {"row_type", "cycle_second"}})
        record.update(
            timestamp=group[-1]["timestamp"],
            dt_seconds=max(1., (group[-1]["timestamp"] - group[0]["timestamp"]).total_seconds() + 1),
            n_hex_rows=len(group), has_validation_warn=not all(row["valid"] for row in group),
        )
        for column in ("abs_blue", "abs_green", "abs_red", "tr_blue", "tr_green", "tr_red", "flow_lpm"):
            record[column] = sum(row[column] for row in group) / len(group)
        records.append(record)
    if records:
        return pl.DataFrame(records)
    return pl.DataFrame(schema={
        "timestamp": pl.Datetime("us"), "dt_seconds": pl.Float64,
        "abs_blue": pl.Float64, "abs_green": pl.Float64, "abs_red": pl.Float64,
        "tr_blue": pl.Float64, "tr_green": pl.Float64, "tr_red": pl.Float64,
        "flow_lpm": pl.Float64,
        "blue_ad_sample": pl.UInt64, "blue_ad_ref": pl.UInt64,
        "green_ad_sample": pl.UInt64, "green_ad_ref": pl.UInt64,
        "red_ad_sample": pl.UInt64, "red_ad_ref": pl.UInt64,
    })


def _detect_era(source_suffix: str, rows: list[dict]) -> str:
    """Detect logger layout from extension and initial packet-cycle anchors."""
    if source_suffix == ".ra5":
        return "ERA_14to16"
    if not rows:
        return "ERA_08to13"
    for index in range(min(len(rows) - 4, 70)):
        if rows[index]["decoded"]["cycle_second"] == 0 and rows[index + 4]["decoded"]["cycle_second"] == 4:
            return "ERA_16on" if len(rows) > 20_000 else "ERA_08to13"
    return "ERA_14to16"


def parse_raw_file(source: str | Path | IO[str] | IO[bytes]) -> pl.DataFrame:
    """Parse one ``.raw``/``.ra5`` file or readable stream into minute L1 rows."""
    text, suffix = _read_text(source)
    rows = _parse_rows(text)
    era = _detect_era(suffix, rows)
    frame = _assemble(rows, drifted=(era == "ERA_14to16"))
    if not frame.is_empty():
        frame = frame.unique(subset=["timestamp"], keep="last").sort("timestamp")
    return frame


def _save(frame: pl.DataFrame, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.write_csv(path)
    elif path.suffix.lower() == ".parquet":
        frame.write_parquet(path)
    else:
        raise ValueError("output_path must have .parquet or .csv extension")


def process_raw_to_l1(raw_input: str | Path | IO[str] | IO[bytes], output_path: str | Path | None = None) -> pl.DataFrame:
    """Read a single file, folder, or file object and optionally save L1."""
    if isinstance(raw_input, (str, Path)) and Path(raw_input).is_dir():
        paths = sorted(p for p in Path(raw_input).rglob("*") if p.is_file() and p.suffix.lower() in {".raw", ".ra5"})
        frames = [parse_raw_file(path) for path in paths]
        nonempty = [frame for frame in frames if not frame.is_empty()]
        result = pl.concat(nonempty, how="diagonal_relaxed") if nonempty else _assemble([])
        if not result.is_empty():
            result = result.unique(subset=["timestamp"], keep="last").sort("timestamp")
    else:
        result = parse_raw_file(raw_input)
    _save(result, output_path)
    return result
