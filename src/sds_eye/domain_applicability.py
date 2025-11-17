"""
Domain applicability helpers for SDS-Eye Irritation project.

This module is a thin wrapper around da_utils, so that other
modules can import a stable interface:
    from src.sds_eye.domain_applicability import (
        load_eye_training_fps,
        compute_da_for_sds,
    )
"""

from .da_utils import load_eye_training_fps, compute_da_for_sds

__all__ = ["load_eye_training_fps", "compute_da_for_sds"]
