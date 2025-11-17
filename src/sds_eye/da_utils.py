import numpy as np

def tanimoto_similarity_matrix(X_query, X_ref):
    """
    Compute Tanimoto similarity between query fingerprints and reference set.

    X_query: (n_query, n_bits)
    X_ref:   (n_ref, n_bits)

    Returns:
        sim_matrix: (n_query, n_ref)
    """
    X_query = X_query.astype(np.int8)
    X_ref = X_ref.astype(np.int8)

    # Dot product counts intersecting bits
    inter = X_query @ X_ref.T  # shape (n_query, n_ref)

    # Number of bits = sum of bits per row
    bits_query = X_query.sum(axis=1).reshape(-1, 1)
    bits_ref = X_ref.sum(axis=1).reshape(1, -1)

    # Tanimoto denominator
    union = bits_query + bits_ref - inter
    union = np.clip(union, a_min=1, a_max=None)

    return inter / union


def compute_applicability_domain(X_test_fp, X_train_fp, threshold=0.30):
    """
    Compute AD for each query molecule:
      - max Tanimoto similarity to training set
      - mean Tanimoto similarity
      - domain flag (in/out)

    threshold: minimum max-similarity to be considered in-domain
    """
    sim_mat = tanimoto_similarity_matrix(X_test_fp, X_train_fp)

    max_sim = sim_mat.max(axis=1)
    mean_sim = sim_mat.mean(axis=1)

    da_flag = np.where(max_sim >= threshold, "in_domain", "out_of_domain")

    return max_sim, mean_sim, da_flag
