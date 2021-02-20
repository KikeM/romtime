import pytest

import fenics
from romtime.solver import HeatEquationSolver

import numpy as np


def test_solver():

    alpha = 3.0
    beta = 1.2
    u_D = fenics.Expression(
        "1 + x[0] * x[0] + alpha * x[1] * x[1] + beta * t",
        degree=2,
        alpha=alpha,
        beta=beta,
        t=0,
    )

    f = fenics.Constant(beta - 2.0 - 2.0 * alpha)

    # We let FEniCS determine what goes in the LHS and RHS automatically.
    T = 2.0
    num_steps = 100

    solver = HeatEquationSolver(nx=8, ny=8, f=f, u_D=u_D, T=T, num_steps=num_steps)
    solver.setup()
    solver.solve()

    errors = list(solver.errors.values())
    _max = np.max(errors)

    assert np.isclose(_max, 0.0)