from collections import defaultdict

import numpy as np
import pandas as pd
from romtime.conventions import ProblemType, Stage, Treewalk, TreewalkNonlinear
from sklearn.model_selection import ParameterSampler


class Reductor:

    FOM = ProblemType.FOM
    ROM = ProblemType.ROM

    BASIS_AFTER_WALK = Treewalk.BASIS_AFTER_WALK
    BASIS_FINAL = Treewalk.BASIS_FINAL
    BASIS_TIME = Treewalk.BASIS_TIME
    ENERGY_MU = Treewalk.ENERGY_MU
    ENERGY_TIME = Treewalk.ENERGY_TIME
    SPECTRUM_MU = Treewalk.SPECTRUM_MU
    SPECTRUM_TIME = Treewalk.SPECTRUM_TIME

    def __init__(self, grid=None) -> None:

        self.grid = grid

        self.mu_space = {
            Stage.OFFLINE: list(),
            Stage.ONLINE: list(),
            Stage.VALIDATION: list(),
        }
        self.report = defaultdict(dict)
        self.errors_rom = defaultdict(list)
        self.summary_errors = None

        self.mu = None

        self.random_state = None

    def __del__(self):

        del self.grid
        del self.mu_space[Stage.OFFLINE]
        del self.mu_space[Stage.ONLINE]
        del self.mu_space[Stage.VALIDATION]

        del self.report
        del self.errors_rom
        del self.summary_errors
        del self.random_state

    @staticmethod
    def _compute_error(u, ue):
        """Compute L2 error between two arrays.

        Parameters
        ----------
        u : np.array
        ue : np.array

        Returns
        -------
        l2_error : float
        """

        e = u - ue
        l2_error = np.linalg.norm(e, ord=2)

        # -------------------------------------------------------------------------
        # Adjust the norm by the length of the vector to obtain the error
        N = len(u)
        l2_error /= np.sqrt(N)

        return l2_error

    def add_mu(self, step, mu):
        """Add parameter vector mu to space.

        Parameters
        ----------
        step : str
            Stage.OFFLINE or Stage.ONLINE
        mu : dict
            Parameter vector.

        Returns
        -------
        idx : int
            Parameter index in the mu-space.
        """
        self.mu_space[step].append(mu)

        idx = self.mu_space[step].index(mu)

        self.mu = mu

        return idx, mu

    def build_sampling_space(self, num, rnd=None):
        """Build a ParameterSampler.

        Parameters
        ----------
        num : int
            Number of samples to produce.
        rnd : int, optional
            Random state, by default None

        Returns
        -------
        ParameterSampler
            Iterator over grid random values.
        """

        grid = self.grid

        sampler = ParameterSampler(
            param_distributions=grid, n_iter=num, random_state=rnd
        )

        return sampler

    def setup(self, rnd=None):
        """Prepare reductor for reduction process.

        Parameters
        ----------
        rnd : int
            Random state.
        """
        self.random_state = rnd

        # Reduced Basis
        self.report[Stage.OFFLINE][self.BASIS_AFTER_WALK] = None
        
        self.report[Stage.OFFLINE][self.BASIS_FINAL] = None
        self.report[Stage.OFFLINE][self.SPECTRUM_MU] = None
        self.report[Stage.OFFLINE][self.ENERGY_MU] = None
        
        self.report[Stage.OFFLINE][self.BASIS_TIME] = dict()
        self.report[Stage.OFFLINE][self.SPECTRUM_TIME] = dict()
        self.report[Stage.OFFLINE][self.ENERGY_TIME] = dict()

        # Nonlinear term
        self.report[Stage.OFFLINE][TreewalkNonlinear.BASIS_AFTER_WALK] = None
        
        self.report[Stage.OFFLINE][TreewalkNonlinear.BASIS_FINAL] = None
        self.report[Stage.OFFLINE][TreewalkNonlinear.SPECTRUM_MU] = None
        self.report[Stage.OFFLINE][TreewalkNonlinear.ENERGY_MU] = None
        
        self.report[Stage.OFFLINE][TreewalkNonlinear.BASIS_TIME] = dict()
        self.report[Stage.OFFLINE][TreewalkNonlinear.SPECTRUM_TIME] = dict()
        self.report[Stage.OFFLINE][TreewalkNonlinear.ENERGY_TIME] = dict()

    def create_errors_summary(self):

        summary_errors = defaultdict(dict)
        for idx, error in self.errors_rom.items():
            summary_errors[idx]["mean"] = np.mean(error)
            summary_errors[idx]["median"] = np.median(error)
            summary_errors[idx]["max"] = np.max(error)
            summary_errors[idx]["min"] = np.min(error)

        self.summary_errors = pd.DataFrame(summary_errors).T
