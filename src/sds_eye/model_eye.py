import os
import joblib
from .featurization import featurize_smiles_series_1024, smiles_to_morgan_1024

# Path to model directory (relative to repo)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")

MODEL_NAME = "eye_xgb_1024_scaffold.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


def load_eye_model():
    """Load the XGBoost model for eye irritation."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print("Eye model loaded successfully.")
    return model


def predict_eye_single(smiles, model, threshold=0.40):
    """Predict irritation probability for a single SMILES."""
    fp = smiles_to_morgan_1024(smiles)
    if fp is None:
        return None, None, False

    prob = model.predict_proba(fp.reshape(1, -1))[0, 1]
    label = int(prob >= threshold)
    return float(prob), label, True


def predict_eye_batch(df, model, smiles_col="SMILES", threshold=0.40):
    """Run predictions over a dataframe of SMILES."""
    if smiles_col not in df.columns:
        raise ValueError(f"Column {smiles_col} not found in dataframe.")

    X, valid_mask = featurize_smiles_series_1024(df[smiles_col])

    probs = model.predict_proba(X)[:, 1]
    labels = (probs >= threshold).astype(int)

    df_out = df.copy()
    df_out["smiles_valid"] = valid_mask
    df_out["eye_irritation_prob"] = probs
    df_out["eye_irritation_flag"] = labels

    # invalid SMILES receive NaN
    df_out.loc[~valid_mask, ["eye_irritation_prob", "eye_irritation_flag"]] = float("nan")

    return df_out
