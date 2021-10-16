import fenics
import numpy as np
from romtime.fom.nonlinear import OneDimensionalBurgers


def define_constant_solution(L=None, nx=None, tf=None, nt=None, which=None):

    domain = {
        OneDimensionalBurgers.L0: L,
        OneDimensionalBurgers.T: tf,
        OneDimensionalBurgers.NX: nx,
        OneDimensionalBurgers.NT: nt,
    }

    # -------------------------------------------------------------------------
    # Boundary conditions
    bL = "1.0"
    dbL_dt = "0.0"

    boundary_conditions = {"bL": bL, "dbL_dt": dbL_dt}

    # -------------------------------------------------------------------------
    # Forcing term
    forcing_term = None

    # -------------------------------------------------------------------------
    # Initial condition
    u0 = fenics.Expression("1.0", degree=1)

    # -------------------------------------------------------------------------
    # Moving boundary function
    def Lt_moving(omega, delta, t, **kwargs):
        """Mesh scaling function (rest)."""
        return 1.0 - delta * (1.0 - np.cos(omega * t))

    def dLt_dt_moving(omega, delta, t, **kwargs):
        """Mesh scaling function time derivative (rest)."""
        return -omega * delta * np.sin(omega * t)

    def Lt_fixed(omega, delta, t, **kwargs):
        """Mesh scaling function (rest)."""
        return 1.0

    def dLt_dt_fixed(omega, delta, t, **kwargs):
        """Mesh scaling function time derivative (rest)."""
        return 0.0

    print("Domain type:", which)

    if which == "fixed":
        Lt = Lt_fixed
        dLt_dt = dLt_dt_fixed
    elif which == "moving":
        Lt = Lt_moving
        dLt_dt = dLt_dt_moving
    else:
        raise NotImplementedError(f"Which domain do you want? {which} was passed.")

    return domain, boundary_conditions, forcing_term, u0, Lt, dLt_dt
