"""Pure-Python decoder for three-band PSAP hexadecimal telemetry."""
from __future__ import annotations

import re


def _payload(line: str) -> str:
    start = line.find('"')
    if start < 0:
        return line.strip()
    end = line.find('"', start + 1)
    return line[start + 1:end].strip() if end >= 0 else line[start + 1:].strip()


def decode_psap_hex(hex_str: str) -> dict[str, int | str | None]:
    """Decode one comma-separated PSAP packet (or a complete raw line).

    The first field is a decimal cycle-second; subsequent numeric fields are hex.
    INIT, LED and DARK packet layouts follow the common 40-bit PSAP logger format.
    """
    fields = [part.strip() for part in _payload(hex_str).split(",")]
    if len(fields) < 2:
        raise ValueError("PSAP payload must contain at least two comma-separated fields")
    try:
        second = int(fields[0], 10)
    except ValueError as exc:
        raise ValueError(f"Invalid decimal cycle second: {fields[0]!r}") from exc
    if not 0 <= second <= 59:
        raise ValueError(f"cycle second outside [0, 59]: {second}")

    def number(index: int) -> int:
        try:
            value = int(fields[index], 16)
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid or missing hexadecimal field {index}") from exc
        return value

    result: dict[str, int | str | None] = {"cycle_second": second}
    if second == 0:
        if len(fields) < 4:
            raise ValueError("INIT packet requires at least 4 fields")
        result.update(row_type="INIT", volume_sum=number(2), seconds_since_E=number(3))
    elif second % 4 == 1 or second % 4 == 2 or second % 4 == 3:
        if len(fields) < 5:
            raise ValueError("LED packet requires at least 5 fields")
        band = {1: "blue", 2: "green", 3: "red"}[second % 4]
        result.update(row_type=band.upper(), **{
            f"{band}_ad_sample": number(1), f"{band}_ad_ref": number(2),
            f"{band}_n_meas": number(3), f"{band}_n_overflow": number(4),
        })
    else:
        if len(fields) < 5:
            raise ValueError("DARK packet requires at least 5 fields")
        result.update(row_type="DARK", dark_seconds_since_start=number(1),
                      dark_ad_sample=number(2), dark_ad_ref=number(3),
                      dark_realtime_sample_current=number(4),
                      dark_realtime_ref_current=number(5) if len(fields) > 5 else None)
    return result


_BOUNDS = {
    "volume_sum": (0, 0xFFFFFFFF), "seconds_since_E": (0, 0xFFFFFFFF),
    "blue_ad_sample": (0, 0xFFFFFFFFFF), "blue_ad_ref": (0, 0xFFFFFFFFFF),
    "green_ad_sample": (0, 0xFFFFFFFFFF), "green_ad_ref": (0, 0xFFFFFFFFFF),
    "red_ad_sample": (0, 0xFFFFFFFFFF), "red_ad_ref": (0, 0xFFFFFFFFFF),
    "blue_n_meas": (0, 0xFFFFFFFF), "green_n_meas": (0, 0xFFFFFFFF),
    "red_n_meas": (0, 0xFFFFFFFF), "blue_n_overflow": (0, 0xFFFF),
    "green_n_overflow": (0, 0xFFFF), "red_n_overflow": (0, 0xFFFF),
    "dark_ad_sample": (0, 0xFFFFFFFFFF), "dark_ad_ref": (0, 0xFFFFFFFFFF),
}


def validate_decoded(record: dict) -> list[str]:
    """Return human-readable range/type violations for a decoded packet."""
    violations = []
    second = record.get("cycle_second")
    if not isinstance(second, int) or not 0 <= second <= 59:
        violations.append(f"cycle_second out of range: {second!r}")
    for key, (lower, upper) in _BOUNDS.items():
        value = record.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or not lower <= value <= upper:
            violations.append(f"{key}={value!r} outside [{lower}, {upper}]")
    return violations
