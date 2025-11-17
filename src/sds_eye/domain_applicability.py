"""
Domain applicability helpers for SDS-Eye Irritation project.

Public interface:

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

from .featurization import featurize_smiles_series_1024


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


def tanimoto_similarity_matrix(X_query: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    """
    Compute Tanimoto similarity between two sets of binary fingerprints.

    X_query : (n_query, n_bits)
    X_ref   : (n_ref,   n_bits)

    Returns
    -------
    np.ndarray
        Matrix (n_query, n_ref) with Tanimoto similarities.
    """
    X_query = X_query.astype(np.int8)
    X_ref = X_ref.astype(np.int8)

    inter = X_query @ X_ref.T  # (n_query, n_ref)

    bits_query = X_query.sum(axis=1).reshape(-1, 1)
    bits_ref = X_ref.sum(axis=1).reshape(1, -1)

    union = bits_query + bits_ref - inter
    sim = inter / np.clip(union, a_min=1, a_max=None)
    return sim


def compute_da_for_sds(
    df: pd.DataFrame,
    X_train_fp: np.ndarray,
    smiles_col: str = "SMILES",
    threshold_in: float = 0.40,
    threshold_borderline: float = 0.20,
) -> pd.DataFrame:
    """
    Compute Domain Applicability (DA) for each SDS row.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con almeno una colonna SMILES.
    X_train_fp : np.ndarray
        Fingerprint matrix (n_train, n_bits) from training compounds.
    smiles_col : str
        Column name containing SMILES.
    threshold_in : float
        Min Tanimoto max for 'in_domain'.
    threshold_borderline : float
        Min Tanimoto max for 'borderline' (below this is 'out_of_domain').

    Returns
    -------
    pd.DataFrame
        Copia di df con colonne aggiuntive:
        - eye_Tanimoto_max
        - eye_Tanimoto_mean
        - eye_DA_flag  (in_domain / borderline / out_of_domain)
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in DataFrame.")

    # Featurizza tutte le molecole (come nel predictor)
    X_query, valid_mask = featurize_smiles_series_1024(df[smiles_col])

    n = len(df)
    max_sim_all = np.full(n, np.nan, dtype=float)
    mean_sim_all = np.full(n, np.nan, dtype=float)
    da_flag_all = np.full(n, "out_of_domain", dtype=object)

    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) > 0:
        X_valid = X_query[valid_mask]
        sim_mat = tanimoto_similarity_matrix(X_valid, X_train_fp)

        max_sim = sim_mat.max(axis=1)
        mean_sim = sim_mat.mean(axis=1)

        max_sim_all[valid_idx] = max_sim
        mean_sim_all[valid_idx] = mean_sim

        flags = np.full(len(max_sim), "out_of_domain", dtype=object)
        flags[max_sim >= threshold_borderline] = "borderline"
        flags[max_sim >= threshold_in] = "in_domain"

        da_flag_all[valid_idx] = flags

    df_out = df.copy()
    df_out["eye_Tanimoto_max"] = max_sim_all
    df_out["eye_Tanimoto_mean"] = mean_sim_all
    df_out["eye_DA_flag"] = da_flag_all

    return df_out


__all__ = ["load_eye_training_fps", "compute_da_for_sds"]
