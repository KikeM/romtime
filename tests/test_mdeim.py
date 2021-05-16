from pprint import pprint

import fenics
import numpy as np
import pytest
from numpy.testing import assert_allclose
from romtime.base import OneDimensionalSolver
from romtime.parameters import get_uniform_dist
from romtime.rom.mdeim import MatrixDiscreteEmpiricalInterpolation
from romtime.utils import (
    bilinear_to_csr,
    functional_to_array,
    get_nonzero_entries,
    plot,
)
from sklearn.model_selection import ParameterSampler

DEGREES = [1, 2, 3, 4, 5]


class MockSolver(OneDimensionalSolver):
    def __init__(
        self,
        domain,
        dirichlet,
        degrees=1,
    ) -> None:
        super().__init__(
            domain=domain,
            dirichlet=dirichlet,
            poly_type="P",
            degrees=degrees,
        )

    def create_diffusion_coefficient(self, mu, t):
        """Create non-linear diffusion term.

        \\alpha(x) = \\alpha_0 (1 + \\varepsilon x^2)

        Returns
        -------
        alpha : fenics.Expression
        """

        alpha_0 = mu["alpha_0"]
        epsilon = mu["epsilon"]

        alpha = fenics.Expression(
            "alpha_0 * (1.0 + epsilon * x[0] * x[0]) * (1.0 + t*t)",
            degree=2,
            alpha_0=alpha_0,
            epsilon=epsilon,
            t=t,
        )

        return alpha

    def assemble_stiffness(self, mu, t, entries=None):

        # ---------------------------------------------------------------------
        # Weak Formulation
        # ---------------------------------------------------------------------
        dot, dx, grad = fenics.dot, fenics.dx, fenics.grad
        u, v = self.u, self.v

        alpha = self.create_diffusion_coefficient(mu=mu, t=t)
        Ah = alpha * dot(grad(u), grad(v)) * dx

        if entries:
            Ah_mat = self.assemble_local(Ah, entries=entries)
        else:
            bc = self.define_homogeneous_dirichlet_bc()
            Ah_mat = self.assemble_operator(Ah, bc)

        return Ah_mat

    def assemble_mass(self, mu, t):
        pass

    def assemble_forcing(self, mu, t, dofs=None):
        pass

    def assemble_lifting(self, mu, t):
        pass


@pytest.fixture
def problem_definition():

    domain = {"L": fenics.Constant(1.0), "nx": 100, "T": 5.0, "nt": 100}

    #  Boundary conditions
    b0 = "(1.0 - exp(- beta * t))"
    bL = "(1.0 - exp(- beta * t)) * (1.0 + delta*delta * L * L)"

    # TODO: This could be computed with sympy
    # but I don't need this overhead for the moment
    db0_dt = "beta * exp(- beta * t)"
    dbL_dt = "(beta * exp(- beta * t)) * (1.0 + delta*delta * L * L)"

    boundary_conditions = {"b0": b0, "bL": bL, "db0_dt": db0_dt, "dbL_dt": dbL_dt}

    # Forcing term
    forcing_term = """beta * exp(- beta * t) * (1.0 + delta * delta * x[0] * x[0]) - 2.0 * delta * delta * alpha_0 * (1.0 - exp(- beta * t))"""

    return domain, boundary_conditions, forcing_term


@pytest.fixture
def grid():

    _grid = {
        "delta": get_uniform_dist(min=0.01, max=2.0),
        "beta": get_uniform_dist(min=1.0, max=10.0),
        "alpha_0": get_uniform_dist(min=0.01, max=2.0),
        "epsilon": [0.0],
    }

    return _grid


@pytest.fixture
def sampler(grid):

    rng = np.random.RandomState(0)
    sampler = ParameterSampler(param_distributions=grid, n_iter=50, random_state=rng)

    return sampler


@pytest.mark.parametrize("degrees", DEGREES)
def test_local_assembler_complete_operator(problem_definition, sampler, degrees):

    domain, dirichlet, _ = problem_definition

    domain["nx"] = 100

    solver = MockSolver(domain=domain, dirichlet=dirichlet, degrees=degrees)
    solver.setup()

    mus = list(sampler)
    mu = mus[0]
    t = 5.0

    Ah = solver.assemble_stiffness(mu=mu, t=t)
    Ah = bilinear_to_csr(Ah)

    rows, cols, entries = get_nonzero_entries(Ah)

    idx = list(zip(rows, cols))
    check = solver.assemble_stiffness(mu=mu, t=t, entries=idx)

    #  Apply boundary conditions
    boundary_value = 1.0
    mask_ones = np.isclose(entries, boundary_value)
    check[mask_ones] = boundary_value

    assert_allclose(entries, check)


def test_mdeim_tree_walk(problem_definition, grid):

    domain, dirichet, _ = problem_definition
    domain["nx"] = 5

    solver = MockSolver(domain=domain, dirichlet=dirichet)
    solver.setup()

    ts = np.linspace(0, 5.0, 20)
    tree_walk = {"ts": ts, "num_snapshots": 50}
    Ah_mdeim = MatrixDiscreteEmpiricalInterpolation(
        assemble=solver.assemble_stiffness, tree_walk_params=tree_walk, grid=grid
    )

    rnd = np.random.RandomState(0)
    Ah_mdeim.setup(rnd=rnd)
    Ah_mdeim.run()

    #  -------------------------------------------------------------------------
    # Assemble with a used parameter
    #  -------------------------------------------------------------------------
    mu = Ah_mdeim.mu_space[Ah_mdeim.OFFLINE][0]
    print("Train mu:")
    pprint(mu)
    expected = solver.assemble_stiffness(mu=mu, t=1.0)
    expected = bilinear_to_csr(expected)
    expected.eliminate_zeros()
    expected = expected.data

    approximation = Ah_mdeim.interpolate(mu=mu, t=1.0)
    approximation = approximation.data

    # Apply boundary conditions
    boundary_value = 1.0
    mask_ones = np.isclose(expected, boundary_value)
    approximation[mask_ones] = boundary_value

    assert_allclose(expected, approximation)

    #  -------------------------------------------------------------------------
    # Assemble with an unseen parameter
    #  -------------------------------------------------------------------------
    rng = np.random.RandomState(19219)
    test_sampler = ParameterSampler(
        param_distributions=grid,
        n_iter=50,
        random_state=rng,
    )

    test_sampler = list(test_sampler)

    mu = test_sampler[0]

    print("Test mu:")
    pprint(mu)

    # Assemble full array
    expected = solver.assemble_stiffness(mu=mu, t=1.0)
    expected = bilinear_to_csr(expected)
    expected.eliminate_zeros()
    expected = expected.data

    approximation = Ah_mdeim.interpolate(mu=mu, t=1.0)
    approximation = approximation.data

    # Apply boundary conditions
    approximation[mask_ones] = boundary_value

    assert_allclose(expected, approximation)

    Ah_mdeim.evaluate(num=50, ts=tree_walk["ts"])

    pass
