"""Steady-State Manufactured Problem Implementation.
[MFP1]
"""
import fenics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from romtime.parameters import get_uniform_dist
from romtime.rom import RomConstructor
from romtime.solver_1d import OneDimensionalHeatEquationSolver
from romtime.utils import function_to_array, round_parameters
from sklearn.model_selection import ParameterSampler

fenics.set_log_level(50)

from pathlib import Path

HERE = Path(__file__).parent
PATH_DATA = HERE / "external" / "MPF1" / "fixed"


def define_mpf1_problem(L, nx, tf, nt):

    domain = {"L": L, "nx": nx, "T": tf, "nt": nt}

    #  Boundary conditions
    b0 = "(1.0 - exp(- beta * t))"
    bL = "(1.0 - exp(- beta * t)) * (1.0 + delta*delta * L * L)"

    # TODO: This could be computed with sympy
    # but I don't need this overhead for the moment
    db0_dt = "beta * exp(- beta * t)"
    dbL_dt = "(beta * exp(- beta * t)) * (1.0 + delta*delta * L * L)"

    boundary_conditions = {"b0": b0, "bL": bL, "db0_dt": db0_dt, "dbL_dt": dbL_dt}

    # Forcing term
    forcing_term = """beta * exp(- beta * t) * (1.0 + delta*delta*x[0]*x[0])-2.0 * delta * delta * alpha_0 * (1.0 - exp(- beta * t))"""

    # Initial condition
    u0 = fenics.Constant(0.0)

    # Exact solution
    ue = "(1.0 - exp(-beta*t)) * (1.0 + delta*delta * x[0]*x[0])"

    return domain, boundary_conditions, forcing_term, u0, ue


def create_solver(L, nx, nt, tf, grid_base):
    """Solve heat equation problem.

    Parameters
    ----------
    L : fenics.Constant
    nx : int
    nt : int
    ft : float
    parameters : tuple

    Returns
    -------
    solver : romtime.OneDimensionalHeatEquationSolver
    """

    domain, boundary_conditions, forcing_term, u0, ue = define_mpf1_problem(
        L, nx, tf, nt
    )

    # We let FEniCS determine what goes in the LHS and RHS automatically.
    solver = OneDimensionalHeatEquationSolver(
        domain=domain,
        dirichlet=boundary_conditions,
        parameters=grid_base,
        forcing_term=forcing_term,
        u0=u0,
        exact_solution=ue,
        filename=None,
    )

    solver.setup()

    return solver


@pytest.fixture
def parameters():

    delta = 1.0
    beta = 5.0
    alpha_0 = 1.0
    epsilon = 0.0

    return delta, beta, alpha_0, epsilon


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
def grid_base(parameters):

    delta, beta, alpha_0, epsilon = parameters

    _grid = dict(delta=delta, beta=beta, alpha_0=alpha_0, epsilon=epsilon)

    return _grid


@pytest.fixture
def domain():

    L = fenics.Constant("2")

    nx = 500

    return L, nx


def test_stiffness(domain, grid_base, grid):

    # Parametrization
    L, nx = domain

    nx = 3

    # Run loop
    nt = 500
    errors = []
    nt = int(nt)

    tf = 10.0

    rng = np.random.RandomState(0)
    sampler = ParameterSampler(param_distributions=grid, n_iter=3, random_state=rng)

    mat_Ah = []
    mat_Mh = []
    mat_fh = []
    mat_fgh_time = []
    mat_fgh_grad = []

    solver = create_solver(nx=nx, nt=nt, tf=tf, L=L, grid_base=grid_base)

    for sample in sampler:

        # Update parameters
        # solver.update_parameters(new=sample)

        Ah = solver.assemble_stiffness(mu=sample, t=0.0)
        Mh = solver.assemble_mass(mu=sample, t=0.0)
        fh = solver.assemble_forcing(mu=sample, t=0.0)
        fgh_time, fgh_grad = solver.assemble_lifting(mu=sample, t=0.0)

        mat_Ah.append(Ah.array().flatten())
        mat_Mh.append(Mh.array().flatten())
        mat_fh.append(np.array(fh).flatten())
        mat_fgh_time.append(np.array(fgh_time).flatten())
        mat_fgh_grad.append(np.array(fgh_grad).flatten())

    mat_Ah = np.array(mat_Ah)
    mat_Mh = np.array(mat_Mh)
    mat_fh = np.array(mat_fh)
    mat_fgh_time = np.array(mat_fgh_time)
    mat_fgh_grad = np.array(mat_fgh_grad)

    expected_mat_Ah = np.array(
        [
            [
                1.0,
                0.0,
                0.0,
                0.0,
                -1.65320831,
                3.30641662,
                -1.65320831,
                0.0,
                0.0,
                -1.65320831,
                3.30641662,
                -1.65320831,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            [
                1.0,
                0.0,
                0.0,
                0.0,
                -1.6414763,
                3.2829526,
                -1.6414763,
                0.0,
                0.0,
                -1.6414763,
                3.2829526,
                -1.6414763,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            [
                1.0,
                0.0,
                0.0,
                0.0,
                -1.32119783,
                2.64239565,
                -1.32119783,
                0.0,
                0.0,
                -1.32119783,
                2.64239565,
                -1.32119783,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ]
    )

    expected_mat_Mh = np.array(
        [
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.11111111,
                0.44444444,
                0.11111111,
                0.0,
                0.0,
                0.11111111,
                0.44444444,
                0.11111111,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.11111111,
                0.44444444,
                0.11111111,
                0.0,
                0.0,
                0.11111111,
                0.44444444,
                0.11111111,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.11111111,
                0.44444444,
                0.11111111,
                0.0,
                0.0,
                0.11111111,
                0.44444444,
                0.11111111,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ]
    )

    expected_mat_fh = np.array(
        [
            [0.0, 18.38874897, 8.71846778, 0.0],
            [0.0, 13.17828361, 6.00010814, 0.0],
            [0.0, 47.42510228, 17.611488, 0.0],
        ]
    )

    expected_mat_fgh_time = np.array(
        [
            [0.0, -24.29836526, -14.62808406, 0.0],
            [0.0, -17.56494639, -10.38677093, 0.0],
            [0.0, -65.64453323, -35.83091895, 0.0],
        ]
    )

    assert_array_almost_equal(
        expected_mat_Ah, mat_Ah, err_msg="Ah operator is not the same."
    )
    assert_array_almost_equal(
        expected_mat_Mh, mat_Mh, err_msg="Mh operator is not the same."
    )
    assert_array_almost_equal(
        expected_mat_fh, mat_fh, err_msg="fh operator is not the same."
    )
    assert_array_almost_equal(
        expected_mat_fgh_time,
        mat_fgh_time,
        err_msg="fgh_time operator is not the same.",
    )


def test_snapshot_generation(domain, parameters, grid_base, grid):

    # Parametrization
    # delta, beta, alpha_0, epsilon = parameters
    L, nx = domain

    # Run loop
    nt = 10
    errors = []
    nt = int(nt)

    tf = 10.0

    rng = np.random.RandomState(0)
    sampler = ParameterSampler(param_distributions=grid, n_iter=3, random_state=rng)

    solutions = []
    exact = []
    solver = create_solver(nx=nx, nt=nt, tf=tf, L=L, grid_base=grid_base)
    for sample in sampler:

        # Update parameters
        solver.update_parameters(new=sample)
        solver.solve()

        tf_eff = solver.timesteps[-1]
        sol = solver.solutions[tf_eff]
        ex = solver.exact[tf_eff]
        sol_vec = function_to_array(sol)
        solutions.append(sol_vec)

        exact_vec = function_to_array(ex)
        exact.append(exact_vec)

        errors_ss = pd.Series(solver.errors)
        errors_ss.name = str(round_parameters(sample, num=2))
        errors.append(errors_ss)

    # for sol, exa in zip(solutions, exact):
    #     plt.plot(solver.x, sol)
    #     plt.plot(solver.x, exa, "--")
    # plt.show()

    results = pd.DataFrame(errors).T
    results = results.apply(np.log10)

    expected = pd.Series(
        {
            "{'alpha_0': 1.1, 'beta': 7.44, 'delta': 1.21, 'epsilon': 0.0}": -5.52706214715911,
            "{'alpha_0': 1.09, 'beta': 4.81, 'delta': 1.3, 'epsilon': 0.0}": -5.451463640761813,
            "{'alpha_0': 0.88, 'beta': 9.03, 'delta': 1.93, 'epsilon': 0.0}": -4.431683790164045,
        },
        name=10.0,
    )

    _results = results.tail(1).T.squeeze()
    assert_series_equal(expected, _results)


def test_rom(domain, grid_base, grid):

    # Parametrization
    # delta, beta, alpha_0, epsilon = parameters
    L, nx = domain

    # Run loop
    nt = 100
    errors = []
    nt = int(nt)

    tf = 10.0

    fom = create_solver(nx=nx, nt=nt, tf=tf, L=L, grid_base=grid_base)

    rom = RomConstructor(fom=fom, grid=grid)

    # Offline phase
    rnd = np.random.RandomState(0)
    rom.setup(rnd=rnd)
    rom.build_reduced_basis(num_snapshots=10)

    # Online phase
    rnd2 = np.random.RandomState(1)
    sampler = rom.build_sampling_space(num=2, random_state=rnd2)

    for mu in sampler:
        mu = round_parameters(sample=mu, num=3)
        rom.solve(mu=mu)

    result = pd.DataFrame(rom.errors)
    expected = pd.read_csv(PATH_DATA / "errors-rom.csv", index_col=0)
    expected.columns = expected.columns.astype(int)

    assert_frame_equal(expected, result)


def test_convergence_rates_fast(domain, grid_base):

    # Parametrization
    L, nx = domain

    tf = -np.log(1.0 - 0.99) / grid_base["beta"]

    # Run loop
    nts = [1e1, 1e2, 1e3]
    errors = []
    for nt in nts:

        nt = int(nt)

        solver = create_solver(
            nx=nx,
            nt=nt,
            tf=tf,
            L=L,
            grid_base=grid_base,
        )

        solver.solve()

        errors_ss = pd.Series(solver.errors)
        errors_ss.name = nt
        errors.append(errors_ss)

    index = errors[0].index

    results = pd.DataFrame(errors).T
    results = results.interpolate(method="index").reindex(index)
    results = results.apply(np.log10)

    expected = pd.read_csv(PATH_DATA / "timestep-errors.csv", index_col=0)
    expected.columns = expected.columns.astype(int)

    expected = expected[results.columns]
    assert_frame_equal(expected, results, check_names=False)


def test_convergence_rates_slow(domain, grid_base):

    # Parametrization
    L, nx = domain

    tf = -np.log(1.0 - 0.99) / grid_base["beta"]

    # Run loop
    nts = [1e1, 2e3, 5e3]
    errors = []
    for nt in nts:

        nt = int(nt)

        solver = create_solver(
            nx=nx,
            nt=nt,
            tf=tf,
            L=L,
            grid_base=grid_base,
        )

        solver.solve()

        errors_ss = pd.Series(solver.errors)
        errors_ss.name = nt
        errors.append(errors_ss)

    index = errors[0].index

    results = pd.DataFrame(errors).T
    results = results.interpolate(method="index").reindex(index)
    results = results.apply(np.log10)

    expected = pd.read_csv(PATH_DATA / "timestep-errors.csv", index_col=0)
    expected.columns = expected.columns.astype(int)

    expected = expected[results.columns]
    assert_frame_equal(expected, results, check_names=False)
