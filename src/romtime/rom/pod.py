import numpy as np
from scipy.linalg import svd

DROP_TOLERANCE = 1e-7


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
    u, s, _ = svd(_snapshots, full_matrices=False, lapack_driver="gesvd")

    # Compute system energy
    eigenvalues = np.power(s, 2)
    total = np.sum(eigenvalues)
    energy = np.cumsum(eigenvalues) / total

    # Tolerance in the energy ratio
    if tol:
        mask = energy < tol
        Q = u[:, mask]
    # Number of elements to retain
    elif num:
        Q = u[:, :num]
    # Clean noisy basis vectors
    else:
        Q = u[:, s > DROP_TOLERANCE]

    return Q, s, energy
