from copy import deepcopy

import numpy as np
from numpy import interp

from romtime.conventions import PistonParameters


class SolutionsStorage:
    def __init__(self, ts, mu, domain, fom, snapshots=None) -> None:

        ts = np.array(ts)
        self.ts = deepcopy(ts)
        self.mu = deepcopy(mu)

        # Collect snapshots as actual arrays
        self.snapshots = deepcopy(snapshots)
        self.fom = deepcopy(fom)

        self.domain = deepcopy(domain)

    def __del__(self):

        del self.ts
        del self.mu
        del self.snapshots
        del self.fom
        del self.domain

    def compute_at(self, x):

        # ---------------------------------------------------------------------
        # Linear interpolation between function values
        _interp = lambda xp, fp: interp(x, xp, fp)

        _range = range(len(self.ts))
        domain = self.domain
        func = self.fom

        # We flip because the domain goes from 1.0 to 0.0
        points = [
            _interp(np.flip(domain[:, idx]), np.flip(func[:, idx])) for idx in _range
        ]
        points = np.array(points)

        # ---------------------------------------------------------------------
        # Scale to physical units
        mu = self.mu
        a0 = mu[PistonParameters.A0]
        points *= a0

        return points


class RomSolutionsStorage(SolutionsStorage):
    def __init__(self, ts, mu, domain, fom, rom) -> None:
        super().__init__(ts=ts, mu=mu, domain=domain, fom=fom)

        self.rom = deepcopy(rom)

    def __del__(self):
        super().__del__()

        del self.rom
