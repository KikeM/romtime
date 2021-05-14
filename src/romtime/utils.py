import fenics
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import find as get_nonzero_entries


def plot(A, plot_kwargs={}, title=None):

    plt.plot(A, **plot_kwargs)
    plt.grid(True)
    if title:
        plt.title(title)
    plt.show()


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


def bilinear_to_array(matrix):
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


def matrix_to_vector(matrix):
    """Convert integrated bilinear form to vector.

    This means to keep only nonzero entries.

    Parameters
    ----------
    matrix : dolfin.cpp.la.matrix
        [description]

    Returns
    -------
    np.array
    """

    csr_matrix = bilinear_to_array(matrix)
    rows, cols, vector = get_nonzero_entries(csr_matrix)

    return vector


def round_parameters(sample, num=2):

    rounded_dict = dict((k, round(v, num)) for (k, v) in sample.items())

    return rounded_dict


def round_parameter_list(param_list, num=2):

    rounded_list = [
        dict((k, round(v, num)) for (k, v) in d.items()) for d in param_list
    ]

    return rounded_list
