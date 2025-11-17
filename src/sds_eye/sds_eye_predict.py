"""
sds_eye_predict.py

Funzioni di predizione per il modello EYE (irritazione oculare) su SMILES:

- load_eye_model(): importato da model_eye.py
- predict_eye_single(): predice una singola molecola
- predict_eye_batch(): predice un DataFrame di molecole, con opzione return_fp
- CLI: uso da terminale con input/output CSV

Esempio d'uso in Python:

    from sds_eye.sds_eye_predict import load_eye_model, predict_eye_batch
    import pandas as pd

    model = load_eye_model()
    df_in = pd.read_csv("data/test_sds_eye.csv")
    df_out, X_fp = predict_eye_batch(df_in, model, smiles_col="SMILES",
                                     threshold=0.40, return_fp=True)

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

# Import locali dal pacchetto
try:
    from .featurization import (
        smiles_to_morgan_1024,
        featurize_smiles_series_1024,
    )
    from .model_eye import load_eye_model
except ImportError as e:
    # Fallback se il modulo viene eseguito in modo "flat" (non come package)
    from featurization import (
        smiles_to_morgan_1024,
        featurize_smiles_series_1024,
    )
    from model_eye import load_eye_model



# 1. Predizione singola
def predict_eye_single(
    smiles: str,
    model,
    threshold: float = 0.40,
) -> Tuple[float, int, bool]:
    """
    Predice l'irritazione oculare per una singola molecola (SMILES).

    Parametri
    ---------
    smiles : str
        Stringa SMILES della molecola.
    model  : oggetto sklearn-like
        Modello con metodo `predict_proba(X)`.
    threshold : float, default 0.40
        Soglia sulla probabilità per assegnare label 0/1.

    Restituisce
    -----------
    prob : float
        Probabilità prevista di irritazione oculare (classe 1).
    label : int
        1 se prob >= threshold, altrimenti 0.
    valid : bool
        False se lo SMILES non è stato parsato correttamente
        (in quel caso prob e label sono NaN).
    """
    fp = smiles_to_morgan_1024(smiles)
    if fp is None:
        # SMILES invalido
        return float("nan"), float("nan"), False

    X = fp.reshape(1, -1)
    prob = float(model.predict_proba(X)[0, 1])
    label = int(prob >= threshold)
    return prob, label, True


# 2. Predizione batch su DataFrame
def predict_eye_batch(
    df: pd.DataFrame,
    model,
    smiles_col: str = "SMILES",
    threshold: float = 0.40,
    return_fp: bool = False,
):
    """
    Applica il modello EYE a un DataFrame con una colonna SMILES.

    Parametri
    ---------
    df : pd.DataFrame
        DataFrame di input con almeno una colonna SMILES.
    model : oggetto sklearn-like
        Modello con metodo `predict_proba(X)`.
    smiles_col : str, default "SMILES"
        Nome della colonna che contiene gli SMILES.
    threshold : float, default 0.40
        Soglia sulla probabilità per assegnare label 0/1.
    return_fp : bool, default False
        Se True, restituisce anche la matrice di fingerprint (n, 1024).

    Restituisce
    -----------
    df_out : pd.DataFrame
        Copia di df con colonne aggiuntive:
            - eye_smiles_valid (bool)
            - eye_irritation_prob (float)
            - eye_irritation_flag (0/1, NaN se invalido)
    X_fp : np.ndarray (n, 1024), opzionale
        Matrice di fingerprint (solo se return_fp=True).
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Colonna '{smiles_col}' non trovata nel DataFrame.")

    # Featurizzazione SMILES → fingerprint 1024 bit
    X_fp, valid_mask = featurize_smiles_series_1024(df[smiles_col])

    n = len(df)
    probs = np.full(n, np.nan, dtype=float)
    labels = np.full(n, np.nan, dtype=float)

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) > 0:
        X_valid = X_fp[valid_mask]
        probs_valid = model.predict_proba(X_valid)[:, 1]

        probs[valid_idx] = probs_valid
        labels[valid_idx] = (probs_valid >= threshold).astype(int)

    df_out = df.copy()
    df_out["eye_smiles_valid"] = valid_mask
    df_out["eye_irritation_prob"] = probs
    df_out["eye_irritation_flag"] = labels

    if return_fp:
        return df_out, X_fp
    return df_out



# 3. CLI helper
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Applica il modello EYE di irritazione oculare "
            "a un CSV con colonna SMILES."
        )
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Percorso al CSV di input (deve contenere una colonna SMILES).",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help=(
            "Percorso al CSV di output. "
            "Se non specificato, aggiunge '_eye_pred' al nome in input."
        ),
    )
    parser.add_argument(
        "--smiles_col",
        default="SMILES",
        help="Nome della colonna SMILES (default: 'SMILES').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Soglia di classificazione (default: 0.40).",
    )
    return parser


def main_cli(args: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    parsed = parser.parse_args(args=args)

    input_path = Path(parsed.input_csv).resolve()
    if parsed.output_csv is None:
        output_path = input_path.with_name(
            input_path.stem + "_eye_pred" + input_path.suffix
        )
    else:
        output_path = Path(parsed.output_csv).resolve()

    if not input_path.exists():
        print(f"[ERRORE] File di input non trovato: {input_path}", file=sys.stderr)
        return 1

    print(f"[INFO] Carico input da: {input_path}")
    df_in = pd.read_csv(input_path)

    print("[INFO] Carico modello EYE...")
    model = load_eye_model()

    print("[INFO] Calcolo predizioni EYE...")
    df_out = predict_eye_batch(
        df_in,
        model=model,
        smiles_col=parsed.smiles_col,
        threshold=parsed.threshold,
        return_fp=False,
    )

    print(f"[INFO] Salvo output in: {output_path}")
    df_out.to_csv(output_path, index=False)
    print("[OK] Completato.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
