from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.sds_eye.model_eye import load_eye_model
from src.sds_eye.sds_eye_predict import predict_eye_batch
from src.sds_eye.domain_applicability import (
    load_eye_training_fps,
    compute_da_for_sds,
)


def get_project_root() -> Path:
    """Return the project root (folder that contains this script)."""
    return Path(__file__).resolve().parent


def main() -> None:
    root = get_project_root()

    input_csv = root / "data" / "test_sds_eye.csv"
    model_path = root / "models" / "eye_xgb_1024_scaffold.pkl"
    training_fp_path = root / "data" / "eye_training_fp_1024.csv"
    output_csv = root / "reports" / "sds_eye_demo_output.csv"

    print(f"Loading input SDS file: {input_csv}")
    df_in = pd.read_csv(input_csv)

    print(f"Loading EYE model from: {model_path}")
    eye_model = load_eye_model(model_path=model_path)

    print("Running predictions...")
    df_pred = predict_eye_batch(
        df_in,
        model=eye_model,
        smiles_col="SMILES",
        threshold=0.40,
    )

    print(f"Loading training fingerprints from: {training_fp_path}")
    X_train_fp = load_eye_training_fps(training_fp_path)

    print("Computing domain applicability (DA)...")
    df_pred_da = compute_da_for_sds(
        df_pred,
        X_train_fp=X_train_fp,
        smiles_col="SMILES",
        threshold_in=0.40,
        threshold_borderline=0.20,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_pred_da.to_csv(output_csv, index=False)
    print(f"Predictions + DA saved to: {output_csv}")

    print("Preview:")
    print(
        df_pred_da[
            [
                "Substance",
                "SMILES",
                "eye_irritation_prob",
                "eye_irritation_flag",
                "eye_Tanimoto_max",
                "eye_DA_flag",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()
