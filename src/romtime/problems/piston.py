import fenics
import numpy as np
from romtime.fom.nonlinear import OneDimensionalBurgers


def define_piston_problem(L=None, nx=None, tf=None, nt=None):

    domain = {
        OneDimensionalBurgers.L0: L,
        OneDimensionalBurgers.T: tf,
        OneDimensionalBurgers.NX: nx,
        OneDimensionalBurgers.NT: nt,
    }

    # -------------------------------------------------------------------------
    # Boundary conditions
    bL = "- omega / a0 * cos(omega * t)"
    dbL_dt = "omega * omega / a0 * sin(omega * t)"

    boundary_conditions = {"bL": bL, "dbL_dt": dbL_dt}

    # -------------------------------------------------------------------------
    # Forcing term
    forcing_term = None

    # -------------------------------------------------------------------------
    # Initial condition
    u0 = fenics.Constant(0.0)

    # -------------------------------------------------------------------------
    # Moving boundary function
    def Lt(omega, t, **kwargs):
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
        return 1.0 - np.sin(omega * t)

    def dLt_dt(omega, t, **kwargs):
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
        return -omega * np.cos(omega * t)

    return domain, boundary_conditions, forcing_term, u0, Lt, dLt_dt
