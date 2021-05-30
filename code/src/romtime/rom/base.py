from collections import defaultdict

import numpy as np
from romtime.parameters import round_parameters
from sklearn.model_selection import ParameterSampler


class Reductor:

    FOM = "fom"
    ROM = "rom"

    OFFLINE = "offline"
    ONLINE = "online"

    def __init__(self, grid) -> None:

        self.grid = grid

        self.mu_space = {self.OFFLINE: list(), self.ONLINE: list()}
        self.report = dict()
        self.errors_rom = defaultdict(list)

        self.random_state = None

    def _compute_error(self, u, ue):

        e = u - ue
        error = np.linalg.norm(e, ord=2)

        N = len(u)
        error /= N

        return error

    def add_mu(self, step, mu):
        """Add parameter vector mu to space.

        Parameters
        ----------
        step : str
            RomConstructor.OFFLINE or RomConstructor.ONLINE
        mu : dict
            Parameter vector.

        Returns
        -------
        idx : int
            Parameter index in the mu-space.
        """
        mu = round_parameters(sample=mu, num=3)
        self.mu_space[step].append(mu)

        idx = self.mu_space[step].index(mu)
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
