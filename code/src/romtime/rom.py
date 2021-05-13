from functools import partial

import fenics
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import orth
from scipy.sparse.linalg import gmres
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm

from romtime.solver_1d import OneDimensionalHeatEquationSolver
from romtime.utils import (
    bilinear_to_array,
    function_to_array,
    functional_to_array,
    round_parameters,
)

RCOND = 1e-14


class RomConstructor:

    OFFLINE = "offline"
    ONLINE = "online"

    GMRES_OPTIONS = dict(atol=1e-7, tol=1e-7, maxiter=1000)

    def __init__(self, fom: OneDimensionalHeatEquationSolver, grid: dict) -> None:

        self.fom = fom
        self.grid = grid

        self.mu_space = dict(offline=list(), online=list())
        self.basis = None

        self.random_state = None

        self.timesteps = dict()
        self.solutions = dict()
        self.liftings = dict()

        self.errors = dict()
        self.exact = dict()

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
        self.mu_space[step].append(mu)

        idx = self.mu_space[step].index(mu)
        return idx

    def to_fom_vector(self, uN):

        V = self.basis
        uh = V.dot(uN)

        return uh

    def to_rom_vector(self, uh):

        V = self.basis
        uh_vec = function_to_array(uh)

        return V.T.dot(uh_vec)

    def to_rom_bilinear(self, Ah):

        Ah = bilinear_to_array(Ah)

        V = self.basis
        AhV = Ah.dot(V)
        AN = np.matmul(V.T, AhV)

        return AN

    def to_rom_functional(self, fh):

        fh = functional_to_array(fh)

        V = self.basis
        fN = V.T.dot(fh)

        return fN

    def _compute_error(self, u, ue):

        e = u - ue
        error = np.linalg.norm(e, ord=2)

        N = len(u)
        error /= N

        return error

    def build_sampling_space(self, num, random_state=None):

        sampler = ParameterSampler(
            param_distributions=self.grid, n_iter=num, random_state=random_state
        )

        return sampler

    def setup(self, rnd):

        self.random_state = rnd

        self.algebraic_solver = self.create_algebraic_solver()

    def build_reduced_basis(self, num_snapshots):
        """Build reduced basis.

        Parameters
        ----------
        num_snapshots : int
            Number of parameter snapshots to take.
        """

        sampler = self.build_sampling_space(
            num=num_snapshots, random_state=self.random_state
        )

        #  Put up the solver and start loop in parameter space
        solver = self.fom
        solver.setup()
        basis_list = list()
        for mu in tqdm(sampler):

            mu = round_parameters(sample=mu, num=3)

            # Save parameter
            self.add_mu(mu=mu, step=self.OFFLINE)

            # Solve FOM time-dependent problem
            solver.update_parameters(mu)
            solver.solve()

            # Orthonormalize the time-snapshots
            _basis = orth(solver._snapshots, rcond=RCOND)
            basis_list.append(_basis)

        # Compress again all the basis
        basis = np.hstack(basis_list)
        print("Basis shape after tree-walk:", basis.shape)
        basis = orth(basis, rcond=RCOND)
        print("Basis shape after compression:", basis.shape)

        # Store reduced basis
        self.N = basis.shape[1]
        self.basis = basis

    def create_algebraic_solver(self):

        solver = partial(gmres, **self.GMRES_OPTIONS)

        return solver

    def solve(self, mu):
        """Solve problem with ROM.

        Parameters
        ----------
        mu : dict
            Parameter-point.
        """

        idx_mu = self.add_mu(mu=mu, step=self.ONLINE)

        fom = self.fom

        # Start iteration
        solutions = dict()
        liftings = dict()

        if fom.exact_solution is not None:
            errors = dict()
            exact = dict()
        else:
            ue = None

        timesteps = [0.0]

        g, _, _ = fom.create_lifting_operator(mu=mu, t=0.0, L=fom.domain["L"])

        # prev_uh = None
        # prev_ue = None

        dt = fom.dt
        t = 0.0
        uN_n = np.zeros(shape=self.N)
        for timestep in tqdm(range(fom.domain["nt"])):

            # Update time
            t += dt
            g.t = t

            timesteps.append(t)

            ########################
            # Assemble linear system
            ########################
            MN_mat = self.assemble_mass(mu=mu, t=t)
            AN_mat = self.assemble_stiffness(mu=mu, t=t)
            KN_mat = MN_mat + dt * AN_mat

            fN_vec = self.assemble_forcing(mu=mu, t=t)
            fgN_vec = self.assemble_lifting(mu=mu, t=t)

            bN_vec = MN_mat.dot(uN_n) + dt * (fN_vec + fgN_vec)

            ###############
            # Solve problem
            ###############
            uN, info = self.algebraic_solver(A=KN_mat, b=bN_vec)

            # Update solution
            uN_n = uN.copy()

            ##############
            # FEM solution
            ##############
            gh = fenics.interpolate(g, fom.V)
            gh = function_to_array(gh)

            uh = self.to_fom_vector(uN)
            uc_h = uh + gh

            # Collect solutions
            solutions[t] = uc_h.copy()
            liftings[t] = gh.copy()

            # Compute error with exact solution
            if fom.exact_solution is not None:
                ue = fenics.Expression(fom.exact_solution, degree=2, t=t, **mu)
                ue_h = fenics.interpolate(ue, fom.V)
                ue_h = function_to_array(ue_h)
                exact[t] = ue_h.copy()

                error = self._compute_error(u=uc_h, ue=ue_h)

                # if (timestep % 30 == 0) | (timestep == 0):

                #     x = fom.x

                #     plt.plot(x, uc_h)
                #     plt.plot(x, ue_h, "--")

                #     if prev_ue is not None:
                #         plt.plot(x, prev_uh, c="b", alpha=0.5)
                #         plt.plot(x, prev_ue, "--", c="b", alpha=0.5)

                #     plt.title(f"t={t}, e={np.log10(error)}")
                #     plt.grid()
                #     plt.show()

                # prev_uh = uc_h.copy()
                # prev_ue = ue_h.copy()

                errors[t] = error

        self.timesteps.update({idx_mu: timesteps})
        self.solutions.update({idx_mu: solutions})
        self.liftings.update({idx_mu: liftings})

        if ue is not None:
            self.errors.update({idx_mu: errors})
            self.exact.update({idx_mu: exact})

    def assemble_mass(self, mu, t):

        # Assemble FOM operator
        Mh = self.fom.assemble_mass(mu, t)
        MN = self.to_rom_bilinear(Mh)

        return MN

    def assemble_stiffness(self, mu, t):

        # Assemble FOM operator
        Ah = self.fom.assemble_stiffness(mu, t)
        AN = self.to_rom_bilinear(Ah)

        return AN

    def assemble_forcing(self, mu, t):

        # Assemble FOM operator
        fh = self.fom.assemble_forcing(mu, t)
        fN = self.to_rom_functional(fh)

        return fN

    def assemble_lifting(self, mu, t):

        # Assemble FOM operator
        fgh_time_vec, fgh_grad_vec = self.fom.assemble_lifting(mu, t)
        fgN_time = self.to_rom_functional(fgh_time_vec)
        fgN_grad = self.to_rom_functional(fgh_grad_vec)

        fgN = fgN_time + fgN_grad

        return fgN
