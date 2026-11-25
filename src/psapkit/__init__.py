"""General-purpose three-band PSAP data processing."""
from .decoder import decode_psap_hex, validate_decoded
from .pipeline import parse_raw_file, process_raw_to_l1
from .absorption import AbsorptionConfig, compute_psap_absorption, process_l1_to_l1_5
from .calibration import FlowCalibrationConfig, FlowSegment
from .screening import ScreeningConfig, detect_rapid_transients, process_l1_5_to_l2
from .workflow import WorkflowResult, run_psap_workflow

__all__ = [
    "decode_psap_hex", "parse_raw_file", "process_raw_to_l1", "compute_psap_absorption",
    "process_l1_to_l1_5", "process_l1_5_to_l2", "detect_rapid_transients", "run_psap_workflow",
    "FlowCalibrationConfig", "AbsorptionConfig", "ScreeningConfig", "FlowSegment", "WorkflowResult",
    "validate_decoded",
]
