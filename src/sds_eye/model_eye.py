from pathlib import Path
import joblib


# Sezione 1: Path di progetto e modello
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"
DEFAULT_MODEL_NAME = "eye_xgb_1024_scaffold.pkl"
DEFAULT_MODEL_PATH = MODEL_DIR / DEFAULT_MODEL_NAME


# Sezione 2: Caricamento modello EYE
def load_eye_model(model_path=None):
    """
    Carica il modello di irritazione oculare EYE.

    Parametri
    ----------
    model_path : str o Path, opzionale
        Path esplicito al file .pkl del modello.
        Se None, usa models/eye_xgb_1024_scaffold.pkl
        nella root del progetto.

    Ritorna
    -------
    model : oggetto scikit-learn compatibile con predict_proba
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modello EYE non trovato in: {model_path}"
        )

    model = joblib.load(model_path)
    return model
