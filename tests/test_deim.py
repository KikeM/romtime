from pprint import pprint

import fenics
import numpy as np
import pytest
from numpy.testing import assert_allclose
from romtime.base import OneDimensionalSolver
from romtime.parameters import get_uniform_dist
from romtime.rom.deim import DiscreteEmpiricalInterpolation
from romtime.utils import functional_to_array, plot
from sklearn.model_selection import ParameterSampler

DEGREES = [1, 2, 3, 4, 5]


class MockSolver(OneDimensionalSolver):
    def __init__(
        self,
        domain,
        dirichlet,
        forcing_term,
        degrees=1,
    ) -> None:
        super().__init__(
            domain=domain,
            dirichlet=dirichlet,
            poly_type="P",
            degrees=degrees,
            forcing_term=forcing_term,
        )

    def assemble_stiffness(self, mu, t):
        pass

    def assemble_mass(self, mu, t):
        pass

    def assemble_forcing(self, mu, t, entries=None):
        """Assemble test forcing term

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        dolfin.cpp.la.Vector
        """
        # Extract names to have a clean implementation
        dx = fenics.dx
        v = self.v
        forcing_term = self.forcing_term
        f = fenics.Expression(forcing_term, degree=2, t=t, **mu)

        # Weak form
        fh = f * v * dx

        #  Select between local assembly or global one
        if entries:
            fh_vec = self.assemble_local(form=fh, entries=entries)
        else:
            bc = self.define_homogeneous_dirichlet_bc()
            fh_vec = self.assemble_operator(weak=fh, bcs=bc)

        return fh_vec

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


@pytest.mark.parametrize("degree", DEGREES)
def test_local_assembler_complete_vector(problem_definition, sampler, degree):

    domain, dirichet, forcing_term = problem_definition

    solver = MockSolver(
        domain=domain,
        dirichlet=dirichet,
        forcing_term=forcing_term,
        degrees=degree,
    )
    solver.setup()

    mus = list(sampler)
    mu = mus[0]
    t = 1.0

    fh = solver.assemble_forcing(mu=mu, t=t)
    fh = functional_to_array(fh)

    entries = [(dof,) for dof in range(len(fh))]
    check = solver.assemble_forcing(mu=mu, t=t, entries=entries)

    #  Apply boundary conditions
    boundary_value = 0.0
    mask_ones = np.isclose(fh, boundary_value)
    check[mask_ones] = boundary_value

    assert_allclose(fh, check)


@pytest.mark.parametrize("degrees", [1, 2, 3, 4, 5])
def test_local_assembler_dofs(problem_definition, sampler, degrees):

    domain, dirichet, forcing_term = problem_definition
    domain["nx"] = 100

    solver = MockSolver(
        domain=domain, dirichlet=dirichet, forcing_term=forcing_term, degrees=degrees
    )
    solver.setup()

    mus = list(sampler)
    mu = mus[0]
    t = 1.0

    fh = solver.assemble_forcing(mu=mu, t=t)
    fh = functional_to_array(fh)

    target_dofs = [5, 47, 98, 55, 14]
    _target_dofs = [(dof,) for dof in target_dofs]
    fh_local = solver.assemble_forcing(mu=mu, t=t, entries=_target_dofs)
    fh_dofs = fh[target_dofs]

    assert_allclose(fh_dofs, fh_local)


def test_deim(problem_definition, sampler, grid):

    domain, dirichet, forcing_term = problem_definition

    solver = MockSolver(domain=domain, dirichlet=dirichet, forcing_term=forcing_term)
    solver.setup()

    t = 0.1
    snapshots = []
    for mu in sampler:
        fh = solver.assemble_forcing(mu=mu, t=t)
        fh = functional_to_array(fh)
        snapshots.append(fh)

    snapshots = np.array(snapshots).T

    fh_deim = DiscreteEmpiricalInterpolation(
        snapshots=snapshots, assemble=solver.assemble_forcing
    )

    fh_deim.run(num_pod=4)

    # Assemble with a used parameter
    print("Train mu:")
    pprint(mu)
    approximation = fh_deim._interpolate(mu=mu, t=1.0)
    expected = solver.assemble_forcing(mu=mu, t=1.0)
    expected = functional_to_array(expected)

    assert_allclose(expected, approximation, atol=1e-15)

    # Assemble with an unseen parameter
    rng = np.random.RandomState(567)
    test_sampler = ParameterSampler(
        param_distributions=grid, n_iter=50, random_state=rng
    )

    test_sampler = list(test_sampler)

    mu = test_sampler[0]

    print("Test mu:")
    pprint(mu)

    # Assemble full vector
    expected = solver.assemble_forcing(mu=mu, t=0.5)
    expected = functional_to_array(expected)

    approximation = fh_deim._interpolate(mu=mu, t=0.5)

    assert_allclose(expected, approximation, atol=1e-15)


def test_deim_tree_walk(problem_definition, grid):

    domain, dirichet, forcing_term = problem_definition

    solver = MockSolver(domain=domain, dirichlet=dirichet, forcing_term=forcing_term)
    solver.setup()

    ts = np.linspace(0, 5.0, 20)
    tree_walk = {"ts": ts, "num_snapshots": 50}
    fh_deim = DiscreteEmpiricalInterpolation(
        assemble=solver.assemble_forcing, tree_walk_params=tree_walk, grid=grid
    )

    rnd = np.random.RandomState(0)
    fh_deim.setup(rnd=rnd)
    fh_deim.run()

    # Assemble with a used parameter
    mu = fh_deim.mu_space[fh_deim.OFFLINE][0]
    print("Train mu:")
    pprint(mu)
    approximation = fh_deim._interpolate(mu=mu, t=1.0)
    expected = solver.assemble_forcing(mu=mu, t=1.0)
    expected = functional_to_array(expected)

    assert_allclose(expected, approximation, atol=1e-15)

    # Assemble with an unseen parameter
    rng = np.random.RandomState(19219)
    test_sampler = ParameterSampler(
        param_distributions=grid, n_iter=50, random_state=rng
    )

    test_sampler = list(test_sampler)

    mu = test_sampler[0]

    print("Test mu:")
    pprint(mu)

    # Assemble full vector
    expected = solver.assemble_forcing(mu=mu, t=0.5)
    expected = functional_to_array(expected)

    approximation = fh_deim._interpolate(mu=mu, t=0.5)

    assert_allclose(expected, approximation, atol=1e-15)

    fh_deim.evaluate(num=50, ts=tree_walk["ts"])
    pass
