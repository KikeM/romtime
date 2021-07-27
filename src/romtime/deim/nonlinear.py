import fenics
import numpy as np
from romtime.conventions import EmpiricalInterpolation, RomParameters, Stage
from romtime.rom.base import Reductor
from romtime.rom.pod import orth
from romtime.utils import (
    bilinear_to_csr,
    eliminate_zeros,
    get_nonzero_entries,
    project_csr,
    vector_to_csr,
)
from tqdm.std import tqdm

from .mdeim import MatrixDiscreteEmpiricalInterpolation


class MatrixDiscreteEmpiricalInterpolationNonlinear(
    MatrixDiscreteEmpiricalInterpolation
):

    TYPE = EmpiricalInterpolation.NONLINEAR

    def __init__(
        self,
        assemble,
        name=None,
        grid=None,
        tree_walk_params=None,
    ):
        super().__init__(
            assemble,
            name=name,
            grid=grid,
            tree_walk_params=tree_walk_params,
        )

        # External function basis
        self.u_n = None

    def get_matrix_topology(self, mu, t, u_n):
        """Get mesh rows and columns arrangement.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        rows : np.array
        cols : np.array
        """

        Ah = self._assemble_matrix(mu=mu, t=t, u_n=u_n)
        Ah = eliminate_zeros(Ah)
        rows, cols, _ = get_nonzero_entries(Ah)

        # Sort by rows
        _list = list(zip(rows, cols))
        rows_cols = sorted(_list, key=lambda x: x[0])

        rows = [x[0] for x in rows_cols]
        cols = [x[1] for x in rows_cols]

        return rows, cols

    def setup(self, rnd, V):
        """Set up reductor and create matrix topology.

        Parameters
        ----------
        rnd : int
            Random seed.

        Creates
        -------
        rows : np.array
        cols : np.array
        """
        Reductor.setup(self=self, rnd=rnd)

        sampler = self.build_sampling_space(num=1)
        mu = list(sampler)[0]

        # Constant function to make sure we span all the possibilities
        one = fenics.Expression("x[0]", degree=1)
        u_n = fenics.interpolate(one, V)
        rows, cols = self.get_matrix_topology(mu=mu, t=1.0, u_n=u_n)

        self.rows = rows
        self.cols = cols

    def run(self, u_n, mu_space=None):
        """Run N-MDEIM offline phase.

        Parameters
        ----------
        u_n : np.array
        num : int
            Number of basis vectors to collect from the snapshots.
        """

        # ---------------------------------------------------------------------
        # Save non-linear basis
        self.u_n = u_n

        # ---------------------------------------------------------------------
        # Tree walk parameters
        ts = self.tree_walk_params[RomParameters.TS]
        num_snapshots = self.tree_walk_params[RomParameters.NUM_SNAPSHOTS]

        num_mu = self.tree_walk_params.get(RomParameters.NUM_MU, None)
        num_t = self.tree_walk_params.get(RomParameters.NUM_TIME, None)
        num_basis = self.tree_walk_params.get(RomParameters.NUM_BASIS, None)
        tol_mu = self.tree_walk_params.get(RomParameters.TOL_MU, None)
        tol_t = self.tree_walk_params.get(RomParameters.TOL_TIME, None)
        tol_basis = self.tree_walk_params.get(RomParameters.TOL_BASIS, None)

        Vfh, sigmas = self.tree_walk(
            ts=ts,
            normalize=False,
            num_mu=num_mu,
            num_t=num_t,
            num_basis=num_basis,
            tol_mu=tol_mu,
            tol_t=tol_t,
            tol_basis=tol_basis,
            num_snapshots=num_snapshots,
            mu_space=mu_space,
        )

        # Store basis and spectrum
        self.Vfh = Vfh
        self.sigmas = sigmas

        self.Nh = self.Vfh.shape[0]
        self.N = self.Vfh.shape[1]

        dofs, P = self.build_interpolation_mesh()

        self.store_dofs(dofs)

        # Store interpolation matrix
        self.PT_U = np.matmul(P.T, self.Vfh)

        # Clean-up
        del P

    def interpolate(self, mu, t, u_n, which=None):
        """Compute operator interpolation for (mu, t) tuple.

        Parameters
        ----------
        mu : dict
            Parametrization.
        t : float
            Time instant.

        Returns
        -------
        csr_matrix : scipy.sparse.csr_matrix
        """
        # Compute vector form interpolation
        approximation = self._interpolate(mu, t, u_n, which=which)

        if which == self.ROM:

            # Reshape to matrix form
            N_V = self.N_V
            approximation = approximation.reshape((N_V, N_V))

        else:
            # Convert to CSR format for fast algebraic operations
            approximation = vector_to_csr(
                entries=approximation,
                rows=self.rows,
                cols=self.cols,
            )

        return approximation

    def _interpolate(self, mu, t, u_n, which=None):
        """Compute functional interpolation for (mu, t) tuple.

        Parameters
        ----------
        mu : dict
            Parametrization.
        t : float
            Time instant.

        Returns
        -------
        approximation : np.array
        """
        # Choose basis
        if (which is None) or (which == self.FOM):
            Vf = self.Vfh
        elif which == self.ROM:
            Vf = self.VfN

        # Local assembly on interpolation mesh
        dofs = self.dofs
        fh_local = self.assemble(mu=mu, t=t, u_n=u_n, entries=dofs)

        # Compute interpolation coefficients
        thetas = self.compute_thetas(rhs=fh_local)

        # Assemble approximation
        N = self.N
        approximation = np.sum([thetas[i] * Vf[:, i] for i in range(N)], axis=0)

        # Impose dirichlet boundary condition
        # TODO : Generalize boundary elements for MDEIM
        if which == self.FOM:
            approximation[0] = 1.0

        return approximation

    def assemble_snapshot(self, mu, t, u_n):
        """Assemble matrix in CSR format.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        Ah : np.array
        """
        Ah = self._assemble_matrix(mu, t, u_n)
        Ah = eliminate_zeros(Ah)

        return Ah.data

    def _assemble_matrix(self, mu, t, u_n):
        """Assemble matrix and convert to CSR format.

        Parameters
        ----------
        mu : dict
            Parametrization.
        t : float
            Time instant.

        Returns
        -------
        Ah : scipy.sparse.csr_matrix
        """
        Ah = self.assemble(mu=mu, t=t, u_n=u_n)
        Ah = bilinear_to_csr(matrix=Ah)
        return Ah

    def tree_walk(
        self,
        ts,
        normalize=True,
        num_mu=None,
        num_t=None,
        num_basis=None,
        tol_mu=None,
        tol_t=None,
        tol_basis=None,
        num_snapshots=None,
        mu_space=None,
    ):
        """Perform a tree walk in the parameter and time space.

        Parameters
        ----------
        ts : iterable of floats
            Time instants to collect.
        num_snapshots : int
            Number of parameters to sample.
        num_mu : int, optional
            Number of POD basis to take in the mu-branch, by default None
        num_t : int, optional
            Number of POD basis to take in the time-branch, by default None
        tol_mu : float, optional
            POD tolerance in the mu-branch, by default None
        tol_t : float, optional
            POD tolerance in the mu-branch, by default None

        Returns
        -------
        basis : np.array
            Final basis to interpolate the operator.
        sigmas : np.array
            Singular value decay.
        """

        # ---------------------------------------------------------------------
        # Sampling space
        if mu_space:
            space = mu_space
        else:
            space = self.build_sampling_space(num=num_snapshots, rnd=self.random_state)

        # ---------------------------------------------------------------------
        # Build basis for each parameter
        basis_time = []
        desc = f"({self.TYPE}-{self.name}) Walk in mu"
        for mu in tqdm(space, desc=desc, leave=True):

            mu_idx, mu = self.add_mu(step=Stage.OFFLINE, mu=mu)

            # POD in time
            _basis, sigmas_time, energy_time = self.walk_time(
                mu=mu,
                ts=ts,
                num_t=num_t,
                tol_t=tol_t,
                normalize=normalize,
            )

            self.report[Stage.OFFLINE]["spectrum-time"][mu_idx] = sigmas_time
            self.report[Stage.OFFLINE]["energy-time"][mu_idx] = energy_time
            self.report[Stage.OFFLINE]["basis-shape-time"][mu_idx] = _basis.shape[1]

            basis_time.append(_basis)

        basis = np.hstack(basis_time)
        self.report[Stage.OFFLINE]["basis-shape-after-tree-walk"] = basis.shape[1]

        # POD in mu
        basis, sigmas_mu, energy_mu = orth(
            snapshots=basis,
            num=num_mu,
            tol=tol_mu,
            normalize=normalize,
        )

        self.report[Stage.OFFLINE]["spectrum-mu"] = sigmas_mu
        self.report[Stage.OFFLINE]["energy-mu"] = energy_mu
        self.report[Stage.OFFLINE]["basis-shape-final"] = basis.shape[1]

        return basis, sigmas_mu

    def walk_time(
        self,
        mu,
        ts,
        normalize=True,
        num_t=None,
        tol_t=None,
        num_basis=None,
        tol_basis=None,
    ):
        """Walk in the time-branch of the tree walk.

        Parameters
        ----------
        mu : dict
        ts : iterable of floats
            Time instants to collect.
        num : int, optional
            Number of POD basis to keep, by default None
        tol : float, optional
            Tolerance for truncate the POD basis, by default None

        Returns
        -------
        basis : np.array
            Resulting basis from the time walk with a freezed parameter.
        sigmas : np.array
            Singular value decay.
        """

        basis_time = []
        u_n = self.u_n
        N_psi = u_n.shape[1]
        desc = f"({self.TYPE}) Walk in time"
        for t in tqdm(ts, desc=desc, leave=False, miniters=100, mininterval=5.0):

            # -----------------------------------------------------------------
            # POD of operator for each basis vector
            snapshots = [
                self.assemble_snapshot(mu=mu, t=t, u_n=u_n[:, idx_psi])
                for idx_psi in range(N_psi)
            ]

            snapshots = np.array(snapshots).T
            # TODO : Generalize boundary elements for MDEIM
            snapshots[0, :] = 0.0
            phi_psi, sigmas, energy = orth(
                snapshots=snapshots,
                num=num_t,
                tol=tol_t,
                normalize=normalize,
            )

            basis_time.append(phi_psi)

        basis_time = np.hstack(basis_time)
        phi, sigmas, energy = orth(
            snapshots=basis_time,
            num=num_t,
            tol=tol_t,
            normalize=normalize,
        )

        return phi, sigmas, energy

    def evaluate(self, ts, num=None, mu_space=None):
        """Evaluate online interpolation.

        Parameters
        ----------
        num : int
            Number of parameters to sample.
        ts : list
            Time instants to sample.
        """

        if mu_space:
            space = mu_space
        else:
            assert num, "Provide number of samples to test"
            space = self.build_sampling_space(num=num)

        msg_mu = f"({self.TYPE}-{self.name}-Evaluation) Walk in mu"
        msg_time = f"({self.TYPE}-Evaluation) Walk in time"
        # ---------------------------------------------------------------------
        # Loop through parameter space
        # N_psi to compute the mean erros across the basis elements
        N_psi = self.u_n.shape[1]
        u_n = self.u_n
        for mu in tqdm(space, desc=msg_mu, leave=True):
            mu_idx, mu = self.add_mu(step=Stage.ONLINE, mu=mu)

            # -----------------------------------------------------------------
            # Loop through timesteps
            for t in tqdm(
                ts,
                desc=msg_time,
                leave=False,
                mininterval=5.0,
                miniters=100,
            ):

                # -------------------------------------------------------------
                # Loop through all the basis elements
                error = 0.0
                for idx_psi in range(N_psi):

                    psi = u_n[:, idx_psi]

                    # ---------------------------------------------------------
                    # Exact solution
                    fh = self.assemble_snapshot(mu=mu, t=t, u_n=psi)
                    # ---------------------------------------------------------
                    # Compute approximation
                    fh_appr = self._interpolate(mu=mu, t=t, u_n=psi, which=self.FOM)
                    # ---------------------------------------------------------
                    # Error
                    error += self._compute_error(u=fh_appr, ue=fh)

                # -------------------------------------------------------------
                # Take average error over each basis vector
                error /= N_psi
                self.errors_rom[mu_idx].append(error)

            # -----------------------------------------------------------------
            self.errors_rom[mu_idx] = np.array(self.errors_rom[mu_idx])