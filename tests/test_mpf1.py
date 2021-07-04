"""Steady-State Manufactured Problem Implementation.
[MFP1]
"""
from functools import partial

import fenics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from romtime.deim import (
    DiscreteEmpiricalInterpolation,
    MatrixDiscreteEmpiricalInterpolation,
)
from romtime.fom import HeatEquationMovingSolver, HeatEquationSolver
from romtime.parameters import get_uniform_dist, round_parameters
from romtime.problems.mfp1 import HyperReducedOrderModelFixed, define_mfp1_problem
from romtime.rom import RomConstructor
from romtime.utils import function_to_array, plot
from sklearn.model_selection import ParameterSampler

from romtime.conventions import OperatorType, RomParameters, Stage

fenics.set_log_level(50)

from pathlib import Path

HERE = Path(__file__).parent
PATH_FIXED = HERE / "external" / "MPF1" / "fixed"
PATH_MOVING = HERE / "external" / "MPF1" / "moving"


def create_solver(L, nx, nt, tf, grid_base, problem_class=HeatEquationSolver):
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

    domain, boundary_conditions, forcing_term, u0, ue, Lt, dLt_dt = define_mfp1_problem(
        L, nx, tf, nt
    )

    # We let FEniCS determine what goes in the LHS and RHS automatically.
    if problem_class is HeatEquationMovingSolver:

        solver = problem_class(
            domain=domain,
            dirichlet=boundary_conditions,
            parameters=grid_base,
            forcing_term=forcing_term,
            u0=u0,
            exact_solution=ue,
            Lt=Lt,
            dLt_dt=dLt_dt,
        )

    else:
        solver = problem_class(
            domain=domain,
            dirichlet=boundary_conditions,
            parameters=grid_base,
            forcing_term=forcing_term,
            u0=u0,
            exact_solution=ue,
        )

    solver.setup()

    return solver


@pytest.fixture
def parameters():

    delta = 1.0
    beta = 5.0
    alpha_0 = 1.0

    return delta, beta, alpha_0


@pytest.fixture
def grid():

    _grid = {
        "delta": get_uniform_dist(min=0.01, max=2.0),
        "beta": get_uniform_dist(min=1.0, max=10.0),
        "alpha_0": get_uniform_dist(min=0.01, max=2.0),
    }

    return _grid


@pytest.fixture
def grid_base(parameters):

    delta, beta, alpha_0 = parameters

    _grid = dict(delta=delta, beta=beta, alpha_0=alpha_0)

    return _grid


@pytest.fixture
def domain():

    L = 2.0

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

    solver = create_solver(nx=nx, nt=nt, tf=tf, L=L, grid_base=grid_base)

    for sample in sampler:

        # Update parameters
        # solver.update_parameters(new=sample)

        Ah = solver.assemble_stiffness(mu=sample, t=0.0)
        Mh = solver.assemble_mass(mu=sample, t=0.0)
        fh = solver.assemble_forcing(mu=sample, t=0.0)
        fgh_time = solver.assemble_lifting(mu=sample, t=0.0)

        mat_Ah.append(Ah.array().flatten())
        mat_Mh.append(Mh.array().flatten())
        mat_fh.append(np.array(fh).flatten())
        mat_fgh_time.append(np.array(fgh_time).flatten())

    mat_Ah = np.array(mat_Ah)
    mat_Mh = np.array(mat_Mh)
    mat_fh = np.array(mat_fh)
    mat_fgh_time = np.array(mat_fgh_time)

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
        solver.update_parametrization(new=sample)
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

    results = pd.DataFrame(errors).T
    results = results.apply(np.log10)

    expected = pd.Series(
        {
            "{'alpha_0': 1.1, 'beta': 7.44, 'delta': 1.21}": -5.52706214715911,
            "{'alpha_0': 1.09, 'beta': 4.81, 'delta': 1.3}": -5.451463640761813,
            "{'alpha_0': 0.88, 'beta': 9.03, 'delta': 1.93}": -4.431683790164045,
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
    sampler = rom.build_sampling_space(num=2, rnd=rnd2)

    for mu in sampler:
        mu = round_parameters(sample=mu, num=3)
        rom.solve(mu=mu, step=Stage.ONLINE)

    result = pd.DataFrame(rom.errors)
    expected = pd.read_csv(PATH_FIXED / "errors-rom.csv", index_col=0)
    expected.columns = expected.columns.astype(int)

    result.index = fom.timesteps[1:]

    assert_frame_equal(expected, result)


def test_rom_deim(domain, grid_base, grid):

    # Parametrization
    # delta, beta, alpha_0, epsilon = parameters
    L, nx = domain

    # Run loop
    nx = 100
    nt = 100
    tf = 10.0

    fom = create_solver(nx=nx, nt=nt, tf=tf, L=L, grid_base=grid_base)

    ###########################################################################
    # DEIM Offline
    ###########################################################################
    ts = np.linspace(tf / nt, tf, nt)
    tree_walk_deim = {"ts": ts, "num_snapshots": 10}

    # Small hack to prevent mistakes and cleaner code
    instantiate_deim = partial(
        DiscreteEmpiricalInterpolation,
        grid=grid,
        tree_walk_params=tree_walk_deim,
    )

    deim_rhs = instantiate_deim(name="RHS", assemble=fom.assemble_rhs)

    rnd = np.random.RandomState(0)

    deim_rhs.setup(rnd=rnd)
    deim_rhs.run()

    ###########################################################################
    # ROM Offline
    ###########################################################################
    rom = RomConstructor(fom=fom, grid=grid)

    # Offline phase
    rom.setup(rnd=rnd)
    rom.build_reduced_basis(num_snapshots=10)
    rom.add_hyper_reductor(reductor=deim_rhs, which=OperatorType.RHS)
    rom.project_reductors()

    ###########################################################################
    # ROM Online
    ###########################################################################
    rnd2 = np.random.RandomState(1)
    sampler = rom.build_sampling_space(num=10, rnd=rnd2)

    for mu in sampler:
        rom.solve(mu=mu, step=Stage.ONLINE)

    result = pd.DataFrame(rom.errors)

    expected = pd.read_csv(PATH_FIXED / "errors-rom-deim.csv", index_col=0)
    expected.columns = expected.columns.astype(int)

    result.index = fom.timesteps[1:]

    assert_frame_equal(expected, result)


def test_rom_deim_mdeim(domain, grid_base, grid):

    # Parametrization
    # delta, beta, alpha_0, epsilon = parameters
    L, nx = domain

    # Run loop
    nx = 100
    nt = 100
    tf = 10.0

    fom = create_solver(nx=nx, nt=nt, tf=tf, L=L, grid_base=grid_base)
    rnd = np.random.RandomState(0)
    ts = np.linspace(tf / nt, tf, nt)

    tree_walk_deim = {"ts": ts, "num_snapshots": 10}

    ###########################################################################
    # DEIM Offline
    ###########################################################################
    # Small hack to prevent mistakes and cleaner code
    instantiate_deim = partial(
        DiscreteEmpiricalInterpolation,
        grid=grid,
        tree_walk_params=tree_walk_deim,
    )

    deim_rhs = instantiate_deim(name="RHS", assemble=fom.assemble_rhs)

    deim_rhs.setup(rnd=rnd)
    deim_rhs.run()

    ###########################################################################
    # MDEIM Offline
    ###########################################################################
    mdeim_stiffness = MatrixDiscreteEmpiricalInterpolation(
        name="Stiffness",
        assemble=fom.assemble_stiffness,
        grid=grid,
        tree_walk_params=tree_walk_deim,
    )

    mdeim_mass = MatrixDiscreteEmpiricalInterpolation(
        name="Mass",
        assemble=fom.assemble_mass,
        grid=grid,
        tree_walk_params=tree_walk_deim,
    )

    mdeim_stiffness.setup(rnd=rnd)
    mdeim_stiffness.run()
    mdeim_mass.setup(rnd=rnd)
    mdeim_mass.run()

    ###########################################################################
    # ROM Offline
    ###########################################################################
    hrom = RomConstructor(fom=fom, grid=grid)

    # Build solution space
    hrom.setup(rnd=rnd)
    hrom.build_reduced_basis(num_snapshots=10)

    # Include the reduction for the algebraic operators
    hrom.add_hyper_reductor(reductor=deim_rhs, which=OperatorType.RHS)
    hrom.add_hyper_reductor(reductor=mdeim_stiffness, which=OperatorType.STIFFNESS)
    hrom.add_hyper_reductor(reductor=mdeim_mass, which=OperatorType.MASS)

    # Project the operators
    hrom.project_reductors()

    ###########################################################################
    # ROM Online
    ###########################################################################
    rnd2 = np.random.RandomState(1)
    sampler = hrom.build_sampling_space(num=10, rnd=rnd2)

    for mu in sampler:
        hrom.solve(mu=mu, step=Stage.ONLINE)

    result = pd.DataFrame(hrom.errors)

    expected = pd.read_csv(PATH_FIXED / "errors-rom-deim.csv", index_col=0)
    expected.columns = expected.columns.astype(int)

    result.index = fom.timesteps[1:]

    assert_frame_equal(expected, result)


def test_hrom(grid):

    domain = dict(
        L0=2.0,
        nx=200,
        nt=200,
        T=10.0,
    )

    _, boundary_conditions, forcing_term, u0, ue, _, _ = define_mfp1_problem()

    fom_params = dict(
        domain=domain,
        dirichlet=boundary_conditions,
        forcing_term=forcing_term,
        u0=u0,
        exact_solution=ue,
    )

    rom_params = dict(num_snapshots=10)

    rnd = np.random.RandomState(0)
    tf, nt = domain["T"], domain["nt"]
    ts = np.linspace(tf / nt, tf, nt)

    deim_params = {
        "ts": ts,
        RomParameters.NUM_SNAPSHOTS: 5,
        RomParameters.NUM_ONLINE: 10,
    }

    models = {
        OperatorType.MASS: True,
        OperatorType.STIFFNESS: True,
        OperatorType.CONVECTION: True,
        OperatorType.RHS: True,
    }

    hrom = HyperReducedOrderModelFixed(
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

    hrom.run_offline_hyperreduction()
    hrom.run_offline_rom()

    online_params = dict(num=20, rnd=np.random.RandomState(2))
    hrom.evaluate_online(params=online_params)

    hrom.generate_summary()

    result = hrom.summary_errors
    expected = pd.DataFrame(
        {
            "mean": {
                0: 8.807124184789381e-05,
                1: 0.00012132092122148534,
                2: 0.00028463644485177196,
                3: 0.0006767487487654447,
                4: 0.00014233023390544555,
                5: 0.00026333377698191303,
                6: 5.34900253584289e-05,
                7: 6.81550614978366e-05,
                8: 0.000252042718195461,
                9: 0.0003363697129489371,
                10: 0.0002505793006693687,
                11: 0.0001724433965770953,
                12: 2.141060522027912e-05,
                13: 0.0003331771639110306,
                14: 0.00014923863773726086,
                15: 0.0006907584644652322,
                16: 0.0005094981622827864,
                17: 0.003554832966891218,
                18: 1.973376436248429e-05,
                19: 0.01204110048706372,
            },
            "median": {
                0: 5.442206951341025e-06,
                1: 1.4519164041908465e-07,
                2: 2.181758488049534e-05,
                3: 1.6499537890485305e-05,
                4: 3.6883965144283754e-05,
                5: 2.893554040307506e-10,
                6: 3.007865061136411e-07,
                7: 4.656174690967966e-05,
                8: 7.246035231109269e-05,
                9: 0.00014675213011780221,
                10: 3.854281480262115e-07,
                11: 7.627181534842806e-08,
                12: 2.7990023340002043e-11,
                13: 1.7091435468539179e-09,
                14: 2.0569227958942892e-10,
                15: 9.299198850985781e-07,
                16: 6.26973010457531e-08,
                17: 0.0020272190215419195,
                18: 1.2940198322784453e-06,
                19: 0.00773069249918343,
            },
            "max": {
                0: 0.0005048646054000295,
                1: 0.001320404103368796,
                2: 0.0020374470442840902,
                3: 0.0058864192582803,
                4: 0.0007198874040654261,
                5: 0.005019912367314376,
                6: 0.0004771765572210609,
                7: 0.00019777202107224557,
                8: 0.001249150623993914,
                9: 0.0012569169540681032,
                10: 0.002283908299008748,
                11: 0.0021018586113048473,
                12: 0.0003637422284794813,
                13: 0.006159533273626118,
                14: 0.0022156722491855243,
                15: 0.007523096548718571,
                16: 0.007941441149434705,
                17: 0.011855186676262749,
                18: 0.00011062208185804066,
                19: 0.03872120341829028,
            },
            "min": {
                0: 1.1978144362448531e-08,
                1: 5.317650790677678e-12,
                2: 1.4820097795947299e-07,
                3: 2.662144965127215e-08,
                4: 1.2869087587980672e-06,
                5: 7.358543107281563e-15,
                6: 5.855373552635898e-11,
                7: 8.478755669236747e-06,
                8: 3.021361302906155e-06,
                9: 9.85702790352542e-06,
                10: 7.1536006950371556e-12,
                11: 9.59962033154823e-13,
                12: 9.328165487928845e-16,
                13: 7.666278763502845e-15,
                14: 1.215448795969909e-14,
                15: 4.025361010579316e-11,
                16: 2.035196498973005e-13,
                17: 0.0002479026719334095,
                18: 2.728502737677861e-09,
                19: 0.0013404369974043628,
            },
        }
    )

    assert_frame_equal(expected, result)

    pass


def test_convergence_fixed(domain, grid_base):

    # Parametrization
    L, nx = domain

    tf = -np.log(1.0 - 0.99) / grid_base["beta"]

    def Lt(t, **kwargs):
        return 1.0

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

        solver.Lt = Lt

        solver.solve()

        errors_ss = pd.Series(solver.errors)
        errors_ss.name = nt
        errors.append(errors_ss)

    index = errors[0].index

    results = pd.DataFrame(errors).T
    results = results.interpolate(method="index").reindex(index)
    results = results.apply(np.log10)

    expected = pd.read_csv(PATH_FIXED / "timestep-errors.csv", index_col=0)
    expected.columns = expected.columns.astype(int)

    expected = expected[results.columns]
    assert_frame_equal(expected, results, check_names=False)


@pytest.mark.skip(reason="Slow.")
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

    expected = pd.read_csv(PATH_FIXED / "timestep-errors.csv", index_col=0)
    expected.columns = expected.columns.astype(int)

    expected = expected[results.columns]
    assert_frame_equal(expected, results, check_names=False)


def test_convergence_moving(domain, grid_base):

    # Parametrization
    L, nx = domain

    nx = 100

    tf = -np.log(1.0 - 0.99) / grid_base["beta"]

    omega = np.pi / 4.0 / tf

    grid_base["omega"] = omega

    # Run loop
    # nts = [1e1, 1e2, 1e3, 2e3, 5e3]
    nts = [1e1, 1e2, 1e3]

    errors = []
    for nt in [1e1]:

        nt = int(nt)

        solver = create_solver(
            nx=nx,
            nt=nt,
            tf=tf,
            L=L,
            grid_base=grid_base,
            problem_class=HeatEquationMovingSolver,
        )

        solver.solve()

        errors_ss = pd.Series(solver.errors)
        errors_ss.name = nt
        errors.append(errors_ss)

    index = errors[0].index

    results = pd.DataFrame(errors).T
    results = results.interpolate(method="index").reindex(index)
    results = results.apply(np.log10)

    # for idx in range(solver.domain_x.shape[1]):

    #     x = solver.domain_x[:, idx]
    #     sol = solver._solutions[:, idx]
    #     plt.plot(x, sol)

    # plt.show()

    expected = pd.read_csv(PATH_MOVING / "timestep-errors.csv", index_col=0)
    expected.columns = expected.columns.astype(int)
    expected = expected[results.columns]

    assert_frame_equal(expected, results, check_names=False)
