from romtime.utils import bilinear_to_csr, function_to_array
import pytest
from romtime.fom import OneDimensionalSolver
from romtime.testing import MockSolverMoving
import numpy as np
import fenics

DEGREES = [1, 2, 3, 4, 5]


@pytest.mark.parametrize("degrees", DEGREES)
def test_function_interpolation(degrees):

    domain = {
        OneDimensionalSolver.L0: 1.0,
        OneDimensionalSolver.NX: 100,
        OneDimensionalSolver.T: 5.0,
        OneDimensionalSolver.NT: 100,
    }

    solver = MockSolverMoving(
        domain=domain,
        degrees=degrees,
        dirichlet=None,
        forcing_term=None,
    )
    solver.setup()

    V = solver.V

    f = fenics.Expression("x[0]", degree=degrees)

    fh = fenics.interpolate(f, V)
    fh = function_to_array(fh)

    SCALE = 0.33
    OneDimensionalSolver._move_mesh(solver, scale=SCALE)

    fhm = fenics.interpolate(f, V)
    fhm = function_to_array(fhm)

    OneDimensionalSolver._move_mesh(solver, back=True)

    fhb = fenics.interpolate(f, V)
    fhb = function_to_array(fhb)

    error = fh - fhb
    norm = np.linalg.norm(error)

    print()
    print("Degrees:", degrees, ", ||e||:", norm)

    assert np.isclose(norm, 0.0)


@pytest.mark.parametrize("degrees", DEGREES)
def test_in_time(degrees):

    domain = {
        OneDimensionalSolver.L0: 1.0,
        OneDimensionalSolver.NX: 100,
        OneDimensionalSolver.T: 5.0,
        OneDimensionalSolver.NT: 100,
    }

    # Define the function scaling the domain
    def Lt(omega, t, **kwargs):
        return 1.0 + np.sin(omega * t)

    solver = MockSolverMoving(
        domain=domain,
        degrees=degrees,
        dirichlet=None,
        forcing_term=None,
        Lt=Lt,
    )
    solver.setup()

    V = solver.V

    f = fenics.Expression("x[0]", degree=degrees)

    fh = fenics.interpolate(f, V)
    fh = function_to_array(fh)

    nt = 10
    T = 10.0
    ts = np.linspace(0.0, T, nt)

    mu = dict(omega=np.pi / 2.0 / T)

    for t in ts:

        fhm = solver.interpolate_func(f, V, mu=mu, t=t)
        fhm = function_to_array(fhm)

        _max = max(fhm)
        expected = Lt(t=t, **mu)
        assert np.isclose(_max, expected)


def test_decorator():

    domain = {
        OneDimensionalSolver.L0: 1.0,
        OneDimensionalSolver.NX: 5,
        OneDimensionalSolver.T: 5.0,
        OneDimensionalSolver.NT: 100,
    }

    def _scale_function(omega, t, **kwargs):
        return 1.0 + np.sin(omega * t)

    solver = MockSolverMoving(
        domain=domain,
        degrees=1,
        dirichlet=None,
        forcing_term=None,
        Lt=_scale_function,
    )
    solver.setup()

    T = 10
    omega = np.pi / 2.0 / T
    mu = {"alpha_0": 0.5, "epsilon": 0.0, "omega": omega}

    Ah0 = solver.assemble_stiffness(mu=mu, t=0.0)
    Ah0 = bilinear_to_csr(Ah0)
    Ah1 = solver.assemble_stiffness(mu=mu, t=5.0)
    Ah1 = bilinear_to_csr(Ah1)
    Ah2 = solver.assemble_stiffness(mu=mu, t=0.0)
    Ah2 = bilinear_to_csr(Ah2)

    # fmt:off
    expected0 = np.array(
        [1.0,0.0,-2.5,5.0,-2.5,-2.5,5.0,-2.5,
        -2.5,5.0,-2.5,-2.5,5.0,-2.5,0.0,1.0,]
    )

    expected1 = np.array([  
        1.        ,   0.        , -38.07611845,  76.15223689,
       -38.07611845, -38.07611845,  76.15223689, -38.07611845,
       -38.07611845,  76.15223689, -38.07611845, -38.07611845,
        76.15223689, -38.07611845,   0.        ,   1.        ])
    # fmt:on

    error0 = np.linalg.norm(expected0 - Ah0.data)
    error1 = np.linalg.norm(expected1 - Ah1.data)
    error2 = np.linalg.norm(expected0 - Ah2.data)

    assert np.isclose(error0, 0.0)
    assert np.isclose(error1, 0.0, rtol=1e-7, atol=1e-7)
    assert np.isclose(error2, 0.0)
