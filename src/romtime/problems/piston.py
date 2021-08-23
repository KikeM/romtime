import fenics
import numpy as np
from romtime.fom.nonlinear import OneDimensionalBurgers
from romtime.rom.hrom import HyperReducedOrderModelMoving


def define_piston_problem(L=None, nx=None, tf=None, nt=None):

    domain = {
        OneDimensionalBurgers.L0: L,
        OneDimensionalBurgers.T: tf,
        OneDimensionalBurgers.NX: nx,
        OneDimensionalBurgers.NT: nt,
    }

    # -------------------------------------------------------------------------
    # Boundary conditions
    bL = "- delta * (omega / a0) * cos(omega * t)"
    dbL_dt = "delta * omega * (omega / a0) * sin(omega * t)"

    boundary_conditions = {"bL": bL, "dbL_dt": dbL_dt}

    # -------------------------------------------------------------------------
    # Forcing term
    forcing_term = None

    # -------------------------------------------------------------------------
    # Initial condition
    u0 = fenics.Expression("0.0", degree=1)

    # -------------------------------------------------------------------------
    # Moving boundary function
    def Lt(omega, delta, t, **kwargs):
        """Mesh scaling function.

        Parameters
        ----------
        omega : float
            Boundary frequency
        t : float

        Returns
        -------
        float
        """
        return 1.0 - delta * np.sin(omega * t)

    def dLt_dt(omega, delta, t, **kwargs):
        """Mesh scaling function time derivative.

        Parameters
        ----------
        omega : float
            Boundary frequency
        t : float

        Returns
        -------
        float
        """
        return -omega * delta * np.cos(omega * t)

    return domain, boundary_conditions, forcing_term, u0, Lt, dLt_dt
