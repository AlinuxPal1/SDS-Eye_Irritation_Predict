import os
from pathlib import Path

import pandas as pd

from src.sds_eye.model_eye import load_eye_model
from src.sds_eye.sds_eye_predict import predict_eye_batch


def main():
    project_root = Path(__file__).resolve().parent

    data_path = project_root / "data" / "test_sds_eye.csv"
    model_path = project_root / "models" / "eye_xgb_1024_scaffold.pkl"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "sds_eye_demo_output.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {data_path}\n"
            f"Make sure data/test_sds_eye.csv exists."
        )

    print(f"Loading input SDS file: {data_path}")
    df_in = pd.read_csv(data_path)

    if not model_path.exists():
        print(f"Model not found at: {model_path}")
        print("This repository does not ship the trained model.")
        print("Place your trained model file as:")
        print("  models/eye_xgb_1024_scaffold.pkl")
        return

    print(f"Loading EYE model from: {model_path}")
    eye_model = load_eye_model(model_path=model_path)

    print("Running predictions...")
    df_pred = predict_eye_batch(
        df_in,
        model=eye_model,
        smiles_col="SMILES",
        threshold=0.40,
    )

    df_pred.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print("Preview:")
    print(df_pred.head())


if __name__ == "__main__":
    main()
