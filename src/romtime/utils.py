import fenics
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import find as get_nonzero_entries


def plot(A, x=None, plot_kwargs={}, title=None, show=True):

    if x:
        plt.plot(x, A, **plot_kwargs)
    else:
        plt.plot(A, **plot_kwargs)

    plt.grid(True)
    if title:
        plt.title(title)

    if show:
        plt.show()


def array_to_function(array, V):
    """Array to FE Function.

    Parameters
    ----------
    array : np.array
    V : fenics.FunctionSpace

    Returns
    -------
    u_n : fenics.Function
    """
    u_n = fenics.Function(V)
    u_n.vector().set_local(array)
    return u_n


def function_to_array(coef_func):
    """Return array from coefficient function.

    Parameters
    ----------
    coef_func : fenics.Expression

    Returns
    -------
    numpy.array
    """
    return coef_func.vector().vec().array


def functional_to_array(operator):
    """Return array from algebraic operator.

    Parameters
    ----------
    operator : dolfin.cpp.la.vector
        Algebraic operator.

    Returns
    -------
    array : np.array
        Dense array representation.
    """
    array = np.array(operator)

    return array


def bilinear_to_csr(matrix):
    """Return array from algebraic operator.

    Parameters
    ----------
    operator : dolfin.cpp.la.matrix
        Algebraic operator.

    Returns
    -------
    array : scipy.sparse.csr_matrix
        Dense array representation.
    """

    petsc_mat = fenics.as_backend_type(matrix).mat()
    array = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=petsc_mat.size)

    return array


def project_csr(Ah, V):
    """Project CSR matrix by basis matrix V.

    A_N = VT Ah V

    Parameters
    ----------
    Ah : scipy.sparse.csr_matrix
    V : np.array
        Basis matrix.

    Returns
    -------
    AN : np.array
    """
    AhV = Ah.dot(V)
    AN = np.matmul(V.T, AhV)
    return AN


def csr_to_vector(matrix):
    """Convert integrated bilinear form to vector.

    This means to keep only nonzero entries.

    Parameters
    ----------
    matrix : dolfin.cpp.la.matrix

    Returns
    -------
    np.array
    """

    csr_matrix = bilinear_to_csr(matrix)
    rows, cols, vector = get_nonzero_entries(csr_matrix)

    return vector


def vector_to_csr(entries, rows, cols):
    """Convert vector matrix to CSR format.

    Parameters
    ----------
    entries : np.array
    rows : np.array
    cols : np.array

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    return csr_matrix((entries, (rows, cols)))


def eliminate_zeros(Ah):
    """Eliminate zeroes from CSR matrix.

    Parameters
    ----------
    Ah : scipy.sparse.csr_matrix

    Returns
    -------
    Ah : scipy.sparse.csr_matrix
    """
    TOLERANCE = 1e-15
    mask = np.isclose(Ah.data, 0, rtol=TOLERANCE, atol=TOLERANCE)
    Ah.data[mask] = 0
    Ah.eliminate_zeros()

    return Ah
