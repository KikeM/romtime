import fenics
import numpy as np
from romtime.fom.nonlinear import OneDimensionalBurgers


def define_piston_problem(L=None, nx=None, tf=None, nt=None, which="sudden"):

    domain = {
        OneDimensionalBurgers.L0: L,
        OneDimensionalBurgers.T: tf,
        OneDimensionalBurgers.NX: nx,
        OneDimensionalBurgers.NT: nt,
    }

    # -------------------------------------------------------------------------
    # Boundary conditions
    print(f"Boundary conditions for {which}")
    if which == "sudden":
        bL = "- delta * (omega / a0) * cos(omega * t)"
        dbL_dt = "delta * omega * (omega / a0) * sin(omega * t)"
    elif which == "rest":
        bL = "- delta * (omega / a0) * sin(omega * t)"
        dbL_dt = "- delta * omega * (omega / a0) * cos(omega * t)"
    else:
        raise NotImplementedError("Which case do you want to solve?")

    boundary_conditions = {"bL": bL, "dbL_dt": dbL_dt}

    # -------------------------------------------------------------------------
    # Forcing term
    forcing_term = None

    # -------------------------------------------------------------------------
    # Initial condition
    u0 = fenics.Expression("0.0", degree=1)

    # -------------------------------------------------------------------------
    # Moving boundary function
    def Lt_sudden(omega, delta, t, **kwargs):
        """Mesh scaling function (sudden)."""
        return 1.0 - delta * np.sin(omega * t)

    def Lt_rest(omega, delta, t, **kwargs):
        """Mesh scaling function (rest)."""
        return 1.0 - delta * (1.0 - np.cos(omega * t))

    def dLt_dt_sudden(omega, delta, t, **kwargs):
        """Mesh scaling function time derivative (sudden)."""
        return -omega * delta * np.cos(omega * t)

    def dLt_dt_rest(omega, delta, t, **kwargs):
        """Mesh scaling function time derivative (rest)."""
        return -omega * delta * np.sin(omega * t)

    if which == "sudden":
        Lt = Lt_sudden
        dLt_dt = dLt_dt_sudden
    elif which == "rest":
        Lt = Lt_rest
        dLt_dt = dLt_dt_rest

    return domain, boundary_conditions, forcing_term, u0, Lt, dLt_dt
