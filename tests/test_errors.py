import fenics
import numpy as np
from romtime.rom.base import Reductor
from romtime.utils import function_to_array


def compute_error(u, ue):
    e = u - ue
    l2_error = np.linalg.norm(e, ord=2)

    N = len(u)
    l2_error /= np.sqrt(N)

    return l2_error


def test_compute_norm_l2_discrete():
    """This is not what we need for error calculation."""

    vec = np.array([1, 2, 3])
    l2 = np.linalg.norm(vec, ord=2)

    expected = vec[0] ** 2.0
    expected += vec[1] ** 2.0
    expected += vec[2] ** 2.0
    expected = np.sqrt(expected)

    assert np.isclose(expected, l2)


def test_compute_norm_error_eps():

    EPS = 1e-4

    vec = np.array([EPS, EPS, EPS])
    l2 = np.linalg.norm(vec, ord=2)

    l2 /= np.sqrt(3)

    assert np.isclose(EPS, l2)


def test_error_computation():

    nx = 5
    mesh = fenics.UnitIntervalMesh(nx)
    V = fenics.FunctionSpace(mesh, "P", 1)

    EPS = 1e-6
    func = fenics.Expression("x[0]", degree=1)
    func_eps = fenics.Expression("eps + x[0]", degree=1, eps=EPS)
    fh = fenics.interpolate(func, V)
    fh_eps = fenics.interpolate(func_eps, V)

    fh_arr = function_to_array(fh)
    fh_eps_arr = function_to_array(fh_eps)

    error = Reductor._compute_error(fh_arr, fh_eps_arr)

    assert np.isclose(EPS, error)
