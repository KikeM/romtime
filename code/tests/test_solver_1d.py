import pytest

import fenics
from romtime.solver_1d import OneDimensionalHeatEquationSolver

import numpy as np

fenics.set_log_level(50)


@pytest.mark.skip
def test_lifting_function():

    # Parametrization
    alpha_0 = 5.0
    epsilon = 0.0
    delta = 1.0
    beta = 1.2
    t0 = 0.0

    # Space-Time Domain Definition
    L = fenics.Constant("3")
    domain = {"L": L, "nx": 32, "T": 2, "nt": 100}

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

    #  Diffusion term definition
    alpha_def = {"alpha_0": alpha_0, "epsilon": epsilon}

    # We let FEniCS determine what goes in the LHS and RHS automatically.
    solver = OneDimensionalHeatEquationSolver(
        domain=domain,
        dirichlet=boundary_conditions,
        alpha_data=alpha_def,
        f=None,
        u0=None,
        exact_solution=None,
    )

    solver.setup()

    g, dg_dt, grad_g = solver.create_lifting_operator(
        dirichlet=boundary_conditions,
        L=domain["L"],
    )

    # Fix a non-zero value for time
    tf = 0.5
    g.t = tf
    dg_dt.t = tf
    grad_g.t = tf
    b0.t = tf
    bL.t = tf

    _L = L.values().item()
    # Check lifting operator
    expected_g = 0.5 * (1.0 - np.exp(-beta / 2.0)) + 0.5 * (
        1.0 - np.exp(-beta / 2.0)
    ) * (1.0 + np.power(delta * _L, 2.0))

    result_g = g(_L / 2.0)

    assert np.isclose(expected_g, result_g), "The lifting operator is not right."

    # Check lifting operator time derivative
    f_prime = beta * np.exp(-beta / 2.0)
    expected_dg_dt = 0.5 * f_prime * (2.0 + (delta * _L) ** 2.0)

    result_dg_dt = dg_dt(_L / 2.0)

    assert np.isclose(
        expected_dg_dt, result_dg_dt
    ), "The lifting operator time derivative is not right."

    # Check lifting operator gradient
    grad_g.t = tf
    expected_grad_g = (1.0 - f_prime / beta) * ((delta * _L) ** 2.0) / _L
    result_grad_g = grad_g(_L / 2.0)

    assert np.isclose(
        expected_grad_g, result_grad_g
    ), "The lifting operator gradient is not right."

    # Check lifting forcing term
    # TODO: parametrize for epsilon != 0
    expected_f_g = -(
        expected_dg_dt + alpha_0 * (1.0 - f_prime / beta) * ((delta * _L) ** 2.0) / (_L)
    )

    f_g = solver.create_lifting_forcing_term()
    f_g.t = tf
    result_f_g = f_g(_L / 2.0)

    assert np.isclose(expected_f_g, result_f_g), "The lifting forcing term is not right"

    mesh = solver.mesh


def test_solver():

    # Parametrization
    delta = 1.0
    beta = 100.0
    alpha_0 = 1.0
    epsilon = 0.0
    t0 = 0.0

    # Space-Time Domain Definition
    L = fenics.Constant("2")
    tf = -np.log(1.0 - 0.99) / beta
    domain = {"L": L, "nx": 500, "T": tf, "nt": 1000}

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

    errors = list(solver.errors.values())
    _max = np.max(errors)

    print(tf)
    print(_max)
    print(domain)

    TOL = 1e-3
    assert np.abs(_max) < TOL