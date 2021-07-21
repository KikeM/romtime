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
from romtime.fom import OneDimensionalBurgers
from romtime.parameters import get_uniform_dist, round_parameters
from romtime.problems.piston import define_piston_problem
from romtime.utils import function_to_array, plot
from sklearn.model_selection import ParameterSampler

from romtime.conventions import OperatorType, RomParameters, Stage

fenics.set_log_level(50)

from pathlib import Path

HERE = Path(__file__).parent
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
        u0=u0,
        exact_solution=None,
        Lt=Lt,
        dLt_dt=dLt_dt,
    )

    solver.setup()

    return solver


@pytest.fixture
def parameters():

    sound = 1.0
    omega = 0.04027158415806615
    alpha = 100.0

    return sound, omega, alpha


@pytest.fixture
def grid():

    sound = []
    omega = []
    alpha = [100.0, 100.0]

    _grid = {
        "sound": get_uniform_dist(min=0.01, max=2.0),
        "omega": get_uniform_dist(min=1.0, max=10.0),
        "alpha": get_uniform_dist(min=0.01, max=2.0),
    }

    return _grid


@pytest.fixture
def grid_base(parameters):

    sound, omega, alpha = parameters
    _grid = dict(sound=sound, omega=omega, alpha=alpha)

    return _grid


@pytest.fixture
def domain():

    L = 1.0
    nx = 100

    return L, nx


def test_fom(parameters, grid_base):

    sound, omega, alpha = parameters

    # Run loop
    L0 = 1
    nx = 100
    nt = 100

    tf = 5.0

    solutions = []
    solver = create_solver(nx=nx, nt=nt, tf=tf, L=L0, grid_base=grid_base)
    mu = {
        "sound": sound,
        "omega": omega,
        "alpha": alpha,
    }

    # Update parameters
    solver.update_parametrization(new=mu)
    solver.solve()

    breakpoint()

    tf_eff = solver.timesteps[-1]
    sol = solver.solutions[tf_eff]
    sol_vec = function_to_array(sol)
    solutions.append(sol_vec)