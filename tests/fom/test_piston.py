"""[Gas Dynamics]
Moving Piston Problem Implementation.
"""
import pickle
from functools import partial
from pprint import pprint

import fenics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_almost_equal
from romtime.conventions import BDF, Domain, OperatorType, RomParameters, Stage
from romtime.deim.nonlinear import MatrixDiscreteEmpiricalInterpolationNonlinear
from romtime.fom import OneDimensionalBurgers
from romtime.parameters import get_uniform_dist
from romtime.problems.piston import define_piston_problem
from romtime.rom import RomConstructorNonlinear
from romtime.rom.hrom import HyperReducedPiston
from romtime.utils import function_to_array, plot
from sklearn.model_selection import ParameterSampler

fenics.set_log_level(50)

from pathlib import Path

HERE = Path(__file__).parents[1]
PATH_DATA = HERE / "external" / "piston"


def create_solver(L, nx, nt, tf, grid_base):
    """Solve burgers equation problem.

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

    (
        domain,
        boundary_conditions,
        forcing_term,
        u0,
        Lt,
        dLt_dt,
    ) = define_piston_problem(L, nx, tf, nt)

    solver = OneDimensionalBurgers(
        domain=domain,
        dirichlet=boundary_conditions,
        parameters=grid_base,
        forcing_term=forcing_term,
        degrees=1,
        u0=u0,
        exact_solution=None,
        Lt=Lt,
        dLt_dt=dLt_dt,
    )

    solver.setup()

    return solver


@pytest.fixture
def parameters():

    a0 = 10.0
    omega = 25.0
    alpha = 0.000001
    delta = 0.1

    return a0, omega, alpha, delta


@pytest.fixture
def grid():

    _grid = {
        "a0": get_uniform_dist(min=8.0, max=10.0),
        "omega": get_uniform_dist(min=15.0, max=20.0),
        "delta": get_uniform_dist(min=0.1, max=0.15),
        # Constants
        "alpha": get_uniform_dist(min=0.000001, max=0.000001),
        "gamma": get_uniform_dist(min=1.4, max=1.4),
    }

    return _grid


@pytest.fixture
def grid_base(parameters):

    a0, omega, alpha, delta = parameters
    _grid = dict(a0=a0, omega=omega, alpha=alpha, delta=delta)

    return _grid


def test_fom(parameters, grid_base):

    a0, omega, alpha, delta = parameters
    gamma = 1.4

    # Run loop
    L0 = 1
    nx = 1000
    nt = 1500

    tf = 1.0

    solver = create_solver(nx=nx, nt=nt, tf=tf, L=L0, grid_base=grid_base)
    mu = {
        "a0": a0,
        "omega": omega,
        "alpha": alpha,
        "delta": delta,
        "gamma": gamma,
    }

    solver.RUNTIME_PROCESS = True

    # Update parameters
    solver.update_parametrization(new=mu)
    solver.solve()

    solver.plot_solution(pics=20)
    ts = solver.timesteps[1:]
    uh_set = list(solver.solutions.values())

    solver.compute_mass_conservation(
        mu=solver.mu,
        ts=ts,
        solutions=uh_set,
        figure=True,
        title="Mass Conservation Check",
    )


@pytest.mark.parametrize("bdf", [BDF.ONE, BDF.TWO])
def test_fom_bdf_scheme(parameters, grid_base, bdf):

    a0, omega, alpha, delta = parameters
    gamma = 1.4

    # Run loop
    L0 = 1
    nx = 1000
    nt = 100

    tf = 1.0

    fom = create_solver(nx=nx, nt=nt, tf=tf, L=L0, grid_base=grid_base)
    mu = {
        "a0": a0,
        "omega": omega,
        "alpha": alpha,
        "delta": delta,
        "gamma": gamma,
    }

    fom.BDF_SCHEME = bdf
    fom.RUNTIME_PROCESS = True

    # Update parameters
    fom.update_parametrization(new=mu)
    fom.solve()

    # solver.plot_probes()
    # ts = fom.timesteps[1:]
    # uh_set = list(fom.solutions.values())
    # mc, outflow, mass, mass_change = fom.compute_mass_conservation(
    #     mu=fom.mu,
    #     ts=ts,
    #     solutions=uh_set,
    #     figure=False,
    #     title=f"BDF-{bdf}, dt = {fom.dt}",
    # )

    # mass_conservation = {
    #     "outflow": outflow,
    #     "mass": mass,
    #     "mass_change": mass_change,
    #     "dt": fom.dt,
    # }

    # with open(f"probes_bdf_{bdf}.pkl", mode="wb") as fp:
    #     pickle.dump(solver.probes, fp)
    # with open(f"mass_conservation_bdf_{bdf}.pkl", mode="wb") as fp:
    #     pickle.dump(mass_conservation, fp)

    path_data = PATH_DATA / "bdf"

    with open(path_data / f"fom_expected_solutions_bdf_{bdf}.pkl", mode="rb") as fp:
        expected_solutions = pickle.load(fp)

    assert_array_almost_equal(expected_solutions, fom._solutions)


@pytest.mark.parametrize("bdf", [BDF.ONE, BDF.TWO])
def test_rom_bdf_scheme(parameters, grid_base, grid, bdf):

    a0, omega, alpha, delta = parameters
    gamma = 1.4

    # Run loop
    L0 = 1
    nx = 1000
    # nt = 100
    nt = 100

    tf = 0.5

    fom = create_solver(nx=nx, nt=nt, tf=tf, L=L0, grid_base=grid_base)
    mu = {
        "a0": a0,
        "omega": omega,
        "alpha": alpha,
        "delta": delta,
        "gamma": gamma,
    }

    fom.BDF_SCHEME = bdf
    fom.RUNTIME_PROCESS = False

    rom = RomConstructorNonlinear(fom=fom, grid=grid)

    # Offline phase
    rnd = np.random.RandomState(0)
    rom.setup(rnd=rnd)
    rom.build_reduced_basis(num_snapshots=1)

    mu = rom.mu_space[Stage.OFFLINE][0]
    rom.solve(mu=mu, step=Stage.ONLINE)

    path_data = PATH_DATA / "bdf"

    with open(path_data / f"rom_expected_solutions_bdf_{bdf}.pkl", mode="rb") as fp:
        expected_solution = pickle.load(fp)

    assert_array_almost_equal(expected_solution, rom._solution)

    # ts = rom.timesteps[1:]
    # for mu_idx, uh_arrays in rom.solutions.items():
    #     fom.compute_mass_conservation(
    #         mu=mu, ts=ts, solutions=list(uh_arrays.values()), figure=True
    #     )


def test_rom(grid_base, grid):

    # Run loop
    L0 = 1
    nx = 1000
    nt = 100

    tf = 0.1

    solutions = []
    fom = create_solver(nx=nx, nt=nt, tf=tf, L=L0, grid_base=grid_base)

    rom = RomConstructorNonlinear(fom=fom, grid=grid)

    # Offline phase
    rnd = np.random.RandomState(0)
    rom.setup(rnd=rnd)
    rom.build_reduced_basis(num_snapshots=1)

    # pprint(rom.report[Stage.OFFLINE]["spectrum-mu"])
    # pprint(rom.report[Stage.OFFLINE]["energy-mu"])
    pprint(rom.report[Stage.OFFLINE]["basis-shape-after-tree-walk"])
    pprint(rom.report[Stage.OFFLINE]["basis-shape-final"])
    pprint(rom.report[Stage.OFFLINE][rom.BASIS_TIME])

    # Online phase
    rnd2 = np.random.RandomState(1)
    sampler = rom.build_sampling_space(num=1, rnd=rnd2)

    for mu in sampler:
        rom.solve(mu=mu, step=Stage.ONLINE)

    ts = rom.timesteps[1:]
    for mu_idx, uh_arrays in rom.solutions.items():
        fom.compute_mass_conservation(
            mu=mu, ts=ts, solutions=list(uh_arrays.values()), figure=True
        )


def test_nonlinear_mdeim(parameters, grid_base, grid):

    a0, omega, alpha, delta = parameters
    gamma = 1.4

    # Run loop
    L0 = 1
    nx = 100
    nt = 50

    tf = 5.0
    fom = create_solver(nx=nx, nt=nt, tf=tf, L=L0, grid_base=grid_base)

    # -------------------------------------------------------------------------
    # Collect "reduced basis vectors"
    ts = np.linspace(0.02, tf, nt)
    ts = ts[::2]

    def _get_solution(t, V):
        uh = fenics.Expression(
            "1.0 + x[0] * x[0] * x[0] + sin(x[0])+ cos(x[0] * x[0]) * sin(t)",
            degree=3,
            t=t,
        )
        uh = fenics.interpolate(uh, V)
        uh = function_to_array(uh)
        return uh

    get_solution = partial(_get_solution, V=fom.V)
    sols = []
    for t in ts:
        uh = get_solution(t=t)
        sols.append(uh)
    sols = np.array(sols).T

    # -------------------------------------------------------------------------
    # Nonlinear MDEIM
    tree_walk = {"ts": ts, "num_snapshots": 5}
    mdeim_nonlinear = MatrixDiscreteEmpiricalInterpolationNonlinear(
        assemble=fom.assemble_trilinear,
        name=OperatorType.TRILINEAR,
        grid=grid,
        tree_walk_params=tree_walk,
    )

    rnd = np.random.RandomState(0)
    mdeim_nonlinear.setup(rnd=rnd, V=fom.V)
    mdeim_nonlinear.run(u_n=sols)

    # -------------------------------------------------------------------------
    # Test over training sample
    mu_space = mdeim_nonlinear.mu_space[Stage.OFFLINE]
    mdeim_nonlinear.evaluate(ts=ts, mu_space=mu_space)

    # -------------------------------------------------------------------------
    # Online evaluation
    mdeim_nonlinear.evaluate(ts=ts, num=5)
    # pd.DataFrame(mdeim_nonlinear.errors_rom).to_csv("online_errors.csv")
    # pprint(mdeim_nonlinear.mu_space)

    # -------------------------------------------------------------------------
    # Assert test
    errors = pd.DataFrame(mdeim_nonlinear.errors_rom).copy()
    expected_errors = pd.read_csv(
        PATH_DATA / "errors_nonlinear_convection.csv", index_col=0
    )

    expected_errors.columns = expected_errors.columns.astype(int)

    assert_frame_equal(expected_errors, errors)


def test_hrom():

    grid_params = {
        "a0": {
            "min": 8.0,
            "max": 10.0,
        },
        "omega": {
            "min": 15.0,
            "max": 20.0,
        },
        "delta": {
            "min": 0.1,
            "max": 0.15,
        },
        # Constants
        "alpha": {
            "min": 1e-6,
            "max": 1e-6,
        },
        "gamma": {
            "min": 1.4,
            "max": 1.4,
        },
    }

    grid = {
        "a0": get_uniform_dist(**grid_params["a0"]),
        "omega": get_uniform_dist(**grid_params["omega"]),
        "delta": get_uniform_dist(**grid_params["delta"]),
        # Constants
        "alpha": get_uniform_dist(**grid_params["alpha"]),
        "gamma": get_uniform_dist(**grid_params["gamma"]),
    }

    domain = {
        Domain.L0: 1.0,
        Domain.NX: 1000,
        Domain.NT: 50,
        Domain.T: 1.0,
    }

    _, boundary_conditions, _, u0, Lt, dLt_dt = define_piston_problem(
        L=domain[Domain.L0],
        nt=domain[Domain.NT],
        nx=domain[Domain.NX],
        tf=domain[Domain.T],
    )

    fom_params = dict(
        grid_params=grid_params,
        domain=domain,
        dirichlet=boundary_conditions,
        u0=u0,
        Lt=Lt,
        dLt_dt=dLt_dt,
    )

    rom_params = dict(num_snapshots=5)

    RND_OFFLINE = 0
    rnd = np.random.RandomState(RND_OFFLINE)
    tf, nt = domain[Domain.T], domain[Domain.NT]
    ts = np.linspace(tf / nt, tf, nt // 4)

    deim_params = {
        "rnd_num": RND_OFFLINE,
        "ts": ts.tolist(),
        RomParameters.NUM_SNAPSHOTS: 1,
        RomParameters.NUM_ONLINE: 1,
    }

    models = {
        OperatorType.MASS: False,
        OperatorType.STIFFNESS: False,
        OperatorType.CONVECTION: False,
        OperatorType.TRILINEAR: False,
        OperatorType.NONLINEAR_LIFTING: False,
        OperatorType.RHS: False,
    }

    hrom = HyperReducedPiston(
        grid=grid,
        fom_params=fom_params,
        rom_params=rom_params,
        deim_params=deim_params,
        mdeim_params=deim_params,
        models=models,
        rnd=rnd,
    )

    hrom.setup()
    hrom.setup_hyperreduction()

    hrom.run_offline_rom()
    hrom.run_offline_hyperreduction(evaluate=False)
    hrom.rom.project_reductors()

    hrom.evaluate_validation()

    RND_ONLINE = 442
    online_params = dict(
        num=5,
        rnd_num=RND_ONLINE,
    )
    hrom.evaluate_online(
        params=online_params,
        rnd=np.random.RandomState(RND_ONLINE),
    )

    hrom.generate_summary()

    # hrom.plot_errors(which="hrom", save="fom_rom_errors", show=False)
    # hrom.plot_spectrums(save="spectrum_decay", show=False)

    # hrom.mdeim_mass.plot_errors(show=False, save=True)
    # hrom.mdeim_stiffness.plot_errors(show=False, save=True)
    # hrom.mdeim_convection.plot_errors(show=False, save=True)
    # hrom.mdeim_nonlinear.plot_errors(show=False, save=True)
    # hrom.mdeim_nonlinear_lifting.plot_errors(show=False, save=True)
    # hrom.deim_rhs.plot_errors(show=False, save=True)

    hrom.dump_mu_space("mu_space.json")
    hrom.dump_errors()
    hrom.dump_setup("setup.json")

    pass
