from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd


# Sezione 1: Caricamento fingerprint di training


def load_eye_training_fp(
    path: Optional[Path] = None
) -> np.ndarray:
    """
    Carica le fingerprint 1024-bit usate per addestrare il modello EYE.

    Parametri
    ----------
    path : Path opzionale
        Percorso esplicito al CSV. Se None, usa:
        <project_root>/data/eye_training_fp_1024.csv

    Ritorna
    -------
    X_train_fp : np.ndarray, shape (n_molecole, 1024)
        Matrice binaria (0/1 o int8) di fingerprint Morgan.
    """
    if path is None:
        this_file = Path(__file__).resolve()
        project_root = this_file.parents[2]
        default_path = project_root / "data" / "eye_training_fp_1024.csv"
        path = default_path

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Fingerprint di training non trovate: {path}.\n"
            "Assicurati di aver copiato 'eye_training_fp_1024.csv' in data/."
        )

    df_fp = pd.read_csv(path)
    X_train_fp = df_fp.values.astype(np.int8)
    return X_train_fp


# Sezione 2: Similarità Tanimoto


def tanimoto_similarity_matrix(
    X_query: np.ndarray,
    X_ref: np.ndarray
) -> np.ndarray:
    """
    Calcola la matrice di similarità Tanimoto tra query e ref.

    X_query : (n_query, n_bits)
    X_ref   : (n_ref,   n_bits)

    Ritorna
    -------
    sim : np.ndarray, shape (n_query, n_ref)
        Similarità Tanimoto in [0, 1]
    """
    X_query = X_query.astype(np.int8)
    X_ref = X_ref.astype(np.int8)

    inter = X_query @ X_ref.T

    bits_query = X_query.sum(axis=1).reshape(-1, 1)
    bits_ref = X_ref.sum(axis=1).reshape(1, -1)

    union = bits_query + bits_ref - inter
    sim = inter / np.clip(union, a_min=1, a_max=None)

    return sim


# Sezione 3: Calcolo Domain Applicability


def compute_da_index(
    X_query: np.ndarray,
    X_train_fp: np.ndarray,
    t_low: float = 0.20,
    t_high: float = 0.40,
    invalid_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola indice di Domain Applicability per ogni molecola.

    Parametri
    ----------
    X_query : np.ndarray, shape (n_mol, 1024)
        Fingerprint delle molecole SDS.
    X_train_fp : np.ndarray, shape (n_train, 1024)
        Fingerprint del training set EYE.
    t_low : float
        Soglia inferiore (es. 0.20) per out-of-domain.
    t_high : float
        Soglia superiore (es. 0.40) per borderline vs in_domain.
    invalid_mask : np.ndarray bool opzionale, shape (n_mol,)
        True dove la SMILES era invalida. Queste verranno etichettate come
        "invalid_smiles" a prescindere dalla similarità.

    Ritorna
    -------
    max_sim : np.ndarray, shape (n_mol,)
        Tanimoto massimo rispetto al training set.
    mean_sim : np.ndarray, shape (n_mol,)
        Tanimoto medio rispetto al training set.
    da_flag : np.ndarray di str, shape (n_mol,)
        Etichette: "in_domain", "borderline", "out_of_domain", "invalid_smiles".
    """
    sim_mat = tanimoto_similarity_matrix(X_query, X_train_fp)

    max_sim = sim_mat.max(axis=1)
    mean_sim = sim_mat.mean(axis=1)

    da_flag = np.empty(X_query.shape[0], dtype=object)
    da_flag[:] = "out_of_domain"

    da_flag[(max_sim >= t_low) & (max_sim < t_high)] = "borderline"
    da_flag[max_sim >= t_high] = "in_domain"

    if invalid_mask is not None:
        invalid_mask = np.asarray(invalid_mask, dtype=bool)
        da_flag[invalid_mask] = "invalid_smiles"
        max_sim[invalid_mask] = 0.0
        mean_sim[invalid_mask] = 0.0

    return max_sim, mean_sim, da_flag
