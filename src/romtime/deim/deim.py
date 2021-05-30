import numpy as np
from romtime.rom.base import Reductor
from romtime.rom.pod import orth
from romtime.utils import functional_to_array, plot
from tqdm import tqdm


def basis_vector(size, index):
    ej = np.zeros((size, 1))
    ej[index, 0] = 1.0

    return ej


class DiscreteEmpiricalInterpolation(Reductor):

    TYPE = "DEIM"

    def __init__(
        self,
        assemble,
        snapshots=None,
        basis=None,
        grid=None,
        tree_walk_params=None,
        name=None,
    ) -> None:
        """Discrete Empirical Interpolation Operator.

        Parameters
        ----------
        assemble : OneDimensionalSolver.assemble-like method
            Method to call to obtain the interpolation cell values.
        snapshots : np.array
        basis : np.array

        Attributes
        ----------
        assemble : OneDimensionalSolver.assemble-like method
        Nh : int
        N : int
            Collateral basis size.
        N_V : int
            Projection basis size
        snapshots : np.array
        Vfh : np.array
        VfN : np.array
        sigmas : np.array
        dofs : list of tuples
        """

        super().__init__(grid=grid)

        self.name = name
        self.assemble = assemble

        if snapshots is not None:
            self.snapshots = snapshots.copy()
        else:
            self.snapshots = None

        if basis is not None:
            self.Vfh = basis.copy()
        else:
            self.Vfh = None

        self.tree_walk_params = tree_walk_params

        self.Nh = int
        self.N = int
        self.N_V = int  # Projection basis size
        self.VfN = None
        self.PT_U = None  # Interpolation matrix
        self.sigmas = list
        self.dofs = list

    def run(self, orthogonalize=True, num_pod=None, tol_pod=None):
        """Create all the structures necessary for DEIM.

        Parameters
        ----------
        num : int
            Number of basis vectors to collect from the snapshots.
        """

        if self.Vfh is None:
            # No snapshots, no basis, perform tree walk
            if self.snapshots is None:
                params = self.tree_walk_params
                assert params is not None, "Please provide tree-walk parameters."

                Vfh, sigmas = self.perform_tree_walk(
                    **params, orthogonalize=orthogonalize
                )
            else:
                Vfh, sigmas = self.compress_snapshots(num=num_pod, tol=tol_pod)

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

    def store_dofs(self, dofs):
        """Store vector entries.

        Parameters
        ----------
        dofs : list of ints
        """
        self.dofs = [(dof,) for dof in dofs]

    def evaluate(self, num, ts):
        """Evaluate online interpolation.

        Parameters
        ----------
        num : int
            Number of parameters to sample.
        ts : list
            Time instants to sample.
        """

        sampler_test = self.build_sampling_space(num=num)

        for mu in tqdm(
            sampler_test,
            desc=f"({self.TYPE}-{self.name}-Evaluation) Walk in mu",
            leave=True,
        ):

            mu_idx, mu = self.add_mu(step=self.ONLINE, mu=mu)
            for t in tqdm(
                ts, desc=f"({self.TYPE}-Evaluation) Walk in time", leave=False
            ):

                # Exact solution
                fh = self.assemble_snapshot(mu, t)

                #  Compute approximation
                fh_appr = self._interpolate(mu, t, which=self.FOM)

                error = self._compute_error(u=fh_appr, ue=fh)
                self.errors_rom[mu_idx].append(error)

            self.errors_rom[mu_idx] = np.array(self.errors_rom[mu_idx])

    def _assemble_functional(self, mu, t):
        """Assemble functional in vector form.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        fh : np.array
        """
        fh = self.assemble(mu=mu, t=t)
        fh = functional_to_array(fh)
        return fh

    def perform_tree_walk(
        self,
        ts,
        num_snapshots,
        orthogonalize=True,
        num_mu=None,
        num_t=None,
        tol_mu=None,
        tol_t=None,
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

        sampler = self.build_sampling_space(num=num_snapshots, rnd=self.random_state)

        basis_time = []
        for mu in tqdm(
            sampler, desc=f"({self.TYPE}-{self.name}) Walk in mu", leave=True
        ):

            idx_mu, mu = self.add_mu(step=self.OFFLINE, mu=mu)

            _basis, sigmas = self._walk_in_time(
                mu=mu,
                ts=ts,
                num=num_t,
                tol=tol_t,
                orthogonalize=orthogonalize,
            )

            basis_time.append(_basis)

        basis_time = np.hstack(basis_time)
        basis, sigmas = orth(
            snapshots=basis_time,
            num=num_mu,
            tol=tol_t,
            orthogonalize=orthogonalize,
        )

        return basis, sigmas

    def _walk_in_time(self, mu, ts, orthogonalize=True, num=None, tol=None):
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

        snapshots = []
        for t in tqdm(ts, desc=f"({self.TYPE}) Walk in time", leave=False):

            op = self.assemble_snapshot(mu, t)
            snapshots.append(op)

        snapshots = np.array(snapshots).T
        basis, sigmas = orth(
            snapshots=snapshots,
            num=num,
            tol=tol,
            orthogonalize=orthogonalize,
        )

        return basis, sigmas

    def assemble_snapshot(self, mu, t):
        """Assemble functional in vector form.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        fh : np.array
        """

        fh = self._assemble_functional(mu, t)

        return fh

    def _interpolate(self, mu, t, which=None):
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
        fh_local = self.assemble(mu=mu, t=t, entries=dofs)

        # Compute interpolation coefficients
        thetas = self.compute_thetas(rhs=fh_local)

        # Assemble approximation
        N = self.N
        approximation = np.sum([thetas[i] * Vf[:, i] for i in range(N)], axis=0)

        return approximation

    def interpolate(self, mu, t, which=None):
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

        Notes
        -----
        This method is necessary because the interpolation is always
        carried out in vector form.

        The public interface of the MDEIM returns a CSR matrix.
        """
        return self._interpolate(mu=mu, t=t, which=None)

    def compute_thetas(self, rhs):
        """Compute interpolation coefficients.

        Parameters
        ----------
        rhs : np.array
            Operator values at interpolation mesh dofs.

        Returns
        -------
        thetas : np.array
            Interpolation coefficients.
        """

        matrix = self.PT_U
        thetas = np.linalg.solve(matrix, rhs)
        return thetas

    def compress_snapshots(self, num=None, tol=None):
        """Build collateral basis via POD compression.

        Parameters
        ----------
        num : int

        Returns
        -------
        Vf : np.array
            Array containing the generated collateral basis.
        sigmas : np.array
        """
        #  Generate collateral basis with SVD
        # TODO : Use POD
        snap = self.snapshots
        Vf, sigmas = orth(snap, num)

        return Vf, sigmas

    def project_basis(self, V):
        """Project DEIM basis unto solution reduced basis.

        Parameters
        ----------
        V : np.array Nh x Nu
            Reduced basis for solution expansion.

        Creates
        -------
        self.VfN
        """

        Vfh = self.Vfh
        VfN = np.matmul(V.T, Vfh)

        self.VfN = VfN

    def build_interpolation_mesh(self):
        """Generate data for empirical interpolation using DEIM algorithm.

        Returns
        -------
        interpolation_dofs : list
            DOFs at which the vectors are interpolated.
        """

        Vf = self.Vfh
        Nh = Vf.shape[0]

        # Warm up
        U = Vf[:, 0]
        dof_1 = np.argmax(np.abs(U))
        P = basis_vector(size=Nh, index=dof_1)

        #  Reshape to have matrix-like shapes
        U = np.reshape(a=U, newshape=(Nh, 1))
        interpolation_dofs = [dof_1]

        # Begin iteration
        Ns = Vf.shape[1]
        for idx in range(1, Ns):

            uj = np.reshape(Vf[:, idx], (Nh, 1))

            # Obtain approximation coefficients
            matrix = np.matmul(P.T, U)
            b = np.matmul(P.T, uj)
            coeff = np.linalg.solve(matrix, b)

            # Compute residual between basis vector uj and current approximation
            residual = uj - np.matmul(U, coeff)

            # Find next interpolation coefficient
            dof_idx = np.argmax(np.abs(residual))

            # Update interpolation basis information
            e_idx = basis_vector(size=Nh, index=dof_idx)
            P = np.hstack((P, e_idx))
            U = np.hstack((U, uj))
            interpolation_dofs.append(dof_idx)

        return interpolation_dofs, P
