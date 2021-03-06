from pprint import pprint

import fenics
import numpy as np
import pytest
from numpy.testing import assert_allclose
from romtime.conventions import OperatorType, Stage
from romtime.deim import MatrixDiscreteEmpiricalInterpolation
from romtime.parameters import get_uniform_dist
from romtime.testing import MockSolver
from romtime.utils import bilinear_to_csr, eliminate_zeros, get_nonzero_entries
from sklearn.model_selection import ParameterSampler

DEGREES = [1, 2, 3, 4, 5]
OPERATORS = [OperatorType.STIFFNESS, OperatorType.MASS, OperatorType.CONVECTION]


@pytest.fixture
def problem_definition():

    domain = {"L0": 1.0, "nx": 100, "T": 5.0, "nt": 100}

    # ┬áBoundary conditions
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
    }

    return _grid


@pytest.fixture
def sampler(grid):

    rng = np.random.RandomState(0)
    sampler = ParameterSampler(param_distributions=grid, n_iter=50, random_state=rng)

    return sampler


@pytest.mark.parametrize("degrees", DEGREES)
@pytest.mark.parametrize("operator", OPERATORS)
def test_local_assembler_complete_operator(
    problem_definition, sampler, degrees, operator
):

    domain, dirichlet, _ = problem_definition

    domain["nx"] = 100

    solver = MockSolver(
        domain=domain, dirichlet=dirichlet, degrees=degrees, forcing_term=None
    )
    solver.setup()

    mus = list(sampler)
    mu = mus[0]
    t = 5.0

    if operator == OperatorType.STIFFNESS:
        assemble = solver.assemble_stiffness
    elif operator == OperatorType.MASS:
        assemble = solver.assemble_mass
    elif operator == OperatorType.CONVECTION:
        assemble = solver.assemble_convection

    Ah = assemble(mu=mu, t=t)
    Ah = bilinear_to_csr(Ah)

    rows, cols, entries = get_nonzero_entries(Ah)
    idx = list(zip(rows, cols))

    check = assemble(mu=mu, t=t, entries=idx)

    assert_allclose(entries, check)


@pytest.mark.parametrize("degrees", DEGREES)
@pytest.mark.parametrize("operator", OPERATORS)
def test_interpolation_dense_matrix(
    problem_definition, sampler, degrees, operator, grid
):

    domain, dirichlet, _ = problem_definition

    domain["nx"] = 100

    solver = MockSolver(
        domain=domain, dirichlet=dirichlet, degrees=degrees, forcing_term=None
    )
    solver.setup()

    mus = list(sampler)
    mu = mus[0]
    t = 5.0

    if operator == OperatorType.STIFFNESS:
        assemble_fom = solver.assemble_stiffness
    elif operator == OperatorType.MASS:
        assemble_fom = solver.assemble_mass
    elif operator == OperatorType.CONVECTION:
        assemble_fom = solver.assemble_convection

    ts = np.linspace(0, 5.0, 20)
    tree_walk = {"ts": ts, "num_snapshots": 50}

    mdeim = MatrixDiscreteEmpiricalInterpolation(
        name=operator,
        assemble=assemble_fom,
        tree_walk_params=tree_walk,
        grid=grid,
    )

    rnd = np.random.RandomState(0)
    mdeim.setup(rnd=rnd)
    mdeim.run()

    Ch = mdeim.assemble(mu=mu, t=0.5)
    Ch = bilinear_to_csr(Ch)
    Ch = Ch.todense()

    Ch_int = mdeim.interpolate(
        mu=mu,
        t=0.5,
        which=OperatorType.FOM,
    )
    Ch_int = Ch_int.todense()

    assert_allclose(Ch, Ch_int)


@pytest.mark.parametrize("operator", OPERATORS)
def test_mdeim_tree_walk(problem_definition, grid, operator):

    domain, dirichet, _ = problem_definition
    domain["nx"] = 100

    solver = MockSolver(domain=domain, dirichlet=dirichet, forcing_term=None)
    solver.setup()

    ts = np.linspace(0, 5.0, 20)
    tree_walk = {"ts": ts, "num_snapshots": 50}

    if operator == OperatorType.STIFFNESS:
        assemble_fom = solver.assemble_stiffness
    elif operator == OperatorType.MASS:
        assemble_fom = solver.assemble_mass
    elif operator == OperatorType.CONVECTION:
        assemble_fom = solver.assemble_convection

    mdeim = MatrixDiscreteEmpiricalInterpolation(
        name=operator,
        assemble=assemble_fom,
        tree_walk_params=tree_walk,
        grid=grid,
    )

    rnd = np.random.RandomState(0)
    mdeim.setup(rnd=rnd)
    mdeim.run()

    # -------------------------------------------------------------------------
    # Assemble with a used parameter
    # -------------------------------------------------------------------------
    mu = mdeim.mu_space[Stage.OFFLINE][0]
    print("Train mu:")
    pprint(mu)
    expected = assemble_fom(mu=mu, t=1.0)
    expected = bilinear_to_csr(expected)
    expected = eliminate_zeros(expected)
    expected = expected.data

    approximation = mdeim.interpolate(mu=mu, t=1.0)
    approximation = approximation.data

    assert_allclose(expected, approximation)

    # -------------------------------------------------------------------------
    # Assemble with an unseen parameter
    # -------------------------------------------------------------------------
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
    expected = assemble_fom(mu=mu, t=1.0)
    expected = bilinear_to_csr(expected)
    expected = eliminate_zeros(expected)
    expected = expected.data

    approximation = mdeim.interpolate(mu=mu, t=1.0)
    approximation = approximation.data

    assert_allclose(expected, approximation)

    mdeim.evaluate(num=50, ts=tree_walk["ts"])

    pass
