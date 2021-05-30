import numpy as np
from scipy.linalg import svd

DROP_TOLERANCE = 1e-8


def orth(snapshots, num=None, tol=None, orthogonalize=True):
    """Compute orthogonalization via SVD.

    Parameters
    ----------
    snapshots : np.array
    num : int, optional
        Number of modes to select.

    Returns
    -------
    Q : np.array
        Basis vectors.
    s : np.array
        Singular values (for all basis vectors).

    Notes
    -----
    Modes with a singular value smaller than 1e-8 are automatically removed.
    """
    if isinstance(snapshots, list):
        raise ValueError("You should use an array, not a list.")

    # L2 normalization
    if orthogonalize == True:
        l2_norms = np.linalg.norm(snapshots, axis=0)
        _snapshots = np.divide(snapshots, l2_norms)
    else:
        _snapshots = snapshots

    # SVD compression
    u, s, _ = svd(_snapshots, full_matrices=False)

    # Clean noisy basis vectors
    u = u[:, s > DROP_TOLERANCE]

    if num:
        Q = u[:, :num]
    else:
        Q = u

    return Q, s
