"""
Domain applicability helpers for SDS-Eye Irritation project.

Stable public interface:

    from src.sds_eye.domain_applicability import (
        load_eye_training_fps,
        compute_da_for_sds,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .da_utils import compute_da_for_sds


def load_eye_training_fps(path: Union[str, Path]) -> np.ndarray:
    """
    Load training fingerprints for the EYE model.

    Parameters
    ----------
    path : str or Path
        Path to a CSV file with shape (n_compounds, n_bits),
        e.g. data/eye_training_fp_1024.csv

    Returns
    -------
    np.ndarray
        Binary matrix of shape (n_compounds, n_bits), dtype=int8.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Training fingerprint file not found: {path}")

    df = pd.read_csv(path)
    X = df.values.astype(np.int8)
    return X


__all__ = ["load_eye_training_fps", "compute_da_for_sds"]
