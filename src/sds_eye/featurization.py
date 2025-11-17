import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


# Sezione 1: SMILES → fingerprint singolo
def smiles_to_morgan_1024(smiles: str, radius: int = 2, n_bits: int = 1024):
    """
    Converte uno SMILES in fingerprint Morgan 1024-bit.
    Restituisce un array numpy (1024,) di int8.
    Se lo SMILES è invalido, restituisce None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# Sezione 2: SMILES → fingerprint per una serie
def featurize_smiles_series_1024(smiles_series):
    """
    Converte una pd.Series di SMILES in una matrice (n_molecole, 1024)
    e una mask booleana che indica quali SMILES sono validi.

    Ritorna:
      X          : np.ndarray shape (n, 1024) di int8
      valid_mask : np.ndarray shape (n,) di bool
    """
    fps = []
    valid_mask = []

    for smi in smiles_series:
        fp = smiles_to_morgan_1024(smi)
        if fp is None:
            valid_mask.append(False)
            fps.append(np.zeros(1024, dtype=np.int8))
        else:
            valid_mask.append(True)
            fps.append(fp)

    X = np.vstack(fps)
    valid_mask = np.array(valid_mask, dtype=bool)
    return X, valid_mask
