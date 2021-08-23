"""Steady-State Manufactured Problem Implementation.
[MFP1]
"""

import fenics
import numpy as np
import pandas as pd
from romtime.solver_1d import OneDimensionalHeatEquationSolver

import matplotlib.pyplot as plt

fenics.set_log_level(50)


def solve_problem(L, nx, nt, delta, beta, alpha_0, epsilon):
    """Solve heat equation problem.

    Parameters
    ----------
    L : fenics.Constant
    nx : int
    nt : int
    delta : float
    beta : float
    alpha_0 : float
    epsilon : float

    Returns
    -------
    solver : romtime.OneDimensionalHeatEquationSolver
    tf : float
    domain : dict
    """

    t0 = 0.0
    tf = -np.log(1.0 - 0.99) / beta

    domain = {"L": L, "nx": nx, "T": tf, "nt": nt}

    #  Boundary conditions
    b0 = fenics.Expression("(1.0 - exp(- beta * t))", degree=1, t=t0, beta=beta)
    bL = fenics.Expression(
        "(1.0 - exp(- beta * t)) * (1.0 + delta*delta * L * L)",
        degree=1,
        t=t0,
        beta=beta,
        delta=delta,
        L=L,
    )

    # TODO: This could be computed with sympy
    # but I don't need this overhead for the moment
    db0_dt = fenics.Expression("beta * exp(- beta * t)", degree=1, t=t0, beta=beta)
    dbL_dt = fenics.Expression(
        "(beta * exp(- beta * t)) * (1.0 + delta*delta * L * L)",
        degree=1,
        t=t0,
        beta=beta,
        delta=delta,
        L=L,
    )

    boundary_conditions = {"b0": b0, "bL": bL, "db0_dt": db0_dt, "dbL_dt": dbL_dt}

    # Forcing term
    forcing_term = "beta * exp(- beta * t) * (1.0 + delta*delta*x[0]*x[0])"
    forcing_term += "-2.0 * delta * delta * alpha_0 * (1.0 - exp(- beta * t))"
    f = fenics.Expression(
        forcing_term, degree=2, beta=beta, delta=delta, alpha_0=alpha_0, t=t0
    )

    #  Diffusion term definition
    alpha_def = {"alpha_0": alpha_0, "epsilon": epsilon}

    # Initial condition
    u0 = fenics.Constant(0.0)

    # Exact solution
    ue = fenics.Expression(
        "(1.0 - exp(-beta*t)) * (1.0 + delta*delta * x[0]*x[0])",
        degree=2,
        delta=delta,
        beta=beta,
        t=t0,
    )

    # We let FEniCS determine what goes in the LHS and RHS automatically.
    solver = OneDimensionalHeatEquationSolver(
        domain=domain,
        dirichlet=boundary_conditions,
        alpha_data=alpha_def,
        f=f,
        u0=u0,
        exact_solution=ue,
        filename=None,
    )

    solver.setup()
    solver.solve()

    return solver, tf, domain


# Parametrization
delta = 1.0
beta = 5.0
alpha_0 = 1.0
epsilon = 0.0

# Space-Time Domain Definition
L = fenics.Constant("2")

nx, nt = 500, 100

nts = [1e1, 1e2, 1e3, 2e3, 5e3]
errors = []
for nt in nts:

    nt = int(nt)

    solver, tf, domain = solve_problem(
        nx=nx,
        nt=nt,
        delta=delta,
        beta=beta,
        alpha_0=alpha_0,
        epsilon=epsilon,
        L=L,
    )

    errors_ss = pd.Series(solver.errors)
    errors_ss.name = nt
    errors.append(errors_ss)

index = errors[0].index

results = pd.DataFrame(errors).T
results = results.interpolate(method="index").reindex(index)
results = results.apply(np.log10)

results.plot()
plt.show()

results.to_csv("MPF1-timestep-errors.csv")
