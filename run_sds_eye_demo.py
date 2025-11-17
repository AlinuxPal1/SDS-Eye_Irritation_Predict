from pathlib import Path

import pandas as pd

from src.sds_eye.model_eye import load_eye_model
from src.sds_eye.sds_eye_predict import predict_eye_batch
from src.sds_eye.featurization import featurize_smiles_series_1024
from src.sds_eye.domain_applicability import (
    load_eye_training_fp,
    compute_da_index,
)


# Sezione 1: Percorsi di base


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def get_paths():
    root = get_project_root()
    data_path = root / "data" / "test_sds_eye.csv"
    model_path = root / "models" / "eye_xgb_1024_scaffold.pkl"
    output_dir = root / "reports"
    output_path = output_dir / "sds_eye_demo_output.csv"
    return root, data_path, model_path, output_dir, output_path


# Sezione 2: Main demo


def main():
    root, data_path, model_path, output_dir, output_path = get_paths()

    print(f"Loading input SDS file: {data_path}")
    df_in = pd.read_csv(data_path)

    print(f"Loading EYE model from: {model_path}")
    eye_model = load_eye_model(model_path=model_path)

    print("Running predictions...")
    df_pred = predict_eye_batch(
        df_in,
        model=eye_model,
        smiles_col="SMILES",
        threshold=0.40,
    )

    print("Computing fingerprints for Domain Applicability...")
    X_sds, valid_mask = featurize_smiles_series_1024(df_in["SMILES"])

    print("Loading training fingerprints for EYE model...")
    X_train_fp = load_eye_training_fp()

    print("Computing Domain Applicability index...")
    max_sim, mean_sim, da_flag = compute_da_index(
        X_query=X_sds,
        X_train_fp=X_train_fp,
        t_low=0.20,
        t_high=0.40,
        invalid_mask=~df_pred["eye_smiles_valid"].astype(bool).values,
    )

    df_pred["eye_Tanimoto_max"] = max_sim
    df_pred["eye_Tanimoto_mean"] = mean_sim
    df_pred["eye_DA_flag"] = da_flag

    output_dir.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")
    print("Preview:")
    print(df_pred.head())


if __name__ == "__main__":
    main()
