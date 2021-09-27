from copy import deepcopy
import fenics
import numpy as np
from romtime.conventions import EmpiricalInterpolation, RomParameters, Stage
from romtime.rom.pod import orth
from romtime.utils import (
    bilinear_to_csr,
    eliminate_zeros,
    get_nonzero_entries,
    project_csr,
    vector_to_csr,
)
from tqdm.std import tqdm

from .deim import DiscreteEmpiricalInterpolation


class MatrixDiscreteEmpiricalInterpolation(DiscreteEmpiricalInterpolation):

    TYPE = EmpiricalInterpolation.MDEIM

    def __init__(
        self,
        assemble,
        name=None,
        grid=None,
        tree_walk_params=None,
    ):
        """Matrix Discrete Empirical Interpolation Operator.

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
            Projection basis size.
        snapshots : np.array
        Vfh : np.array
        VfN : np.array
        sigmas : np.array
        dofs : list of tuples
        rows : list of ints
            Rows CSR pattern.
        cols : list of ints
            Columns CSR pattern.
        """

        super().__init__(
            assemble=assemble,
            name=name,
            grid=grid,
            tree_walk_params=tree_walk_params,
        )

        # Matrix topology
        self.rows = None
        self.cols = None

    def copy(self):

        new = super().copy()

        if self.rows is not None:
            new.rows = deepcopy(self.rows)
        if self.cols is not None:
            new.cols = deepcopy(self.cols)

        return new

    def setup(self, rnd):
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
        super().setup(rnd=rnd)

        sampler = self.build_sampling_space(num=1)
        mu = list(sampler)[0]

        rows, cols = self.get_matrix_topology(mu=mu, t=1.0)

        self.rows = rows
        self.cols = cols

    def get_entry(self, idx):
        """Get the row and column entry from the vector form index.

        Parameters
        ----------
        idx : int
            Vector entry in the vector format of the matrix.

        Returns
        -------
        row : int
        col : int
        """
        return self.rows[idx], self.cols[idx]

    def store_dofs(self, dofs):
        """Store matrix entries to interpolate.

        Parameters
        ----------
        dofs : list of ints
        """
        self.dofs = [self.get_entry(dof) for dof in dofs]

    def get_matrix_topology(self, mu, t):
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

        Ah = self._assemble_matrix(mu, t)
        Ah = eliminate_zeros(Ah)
        rows, cols, _ = get_nonzero_entries(Ah)

        # Sort by rows
        _list = list(zip(rows, cols))
        rows_cols = sorted(_list, key=lambda x: x[0])

        rows = [x[0] for x in rows_cols]
        cols = [x[1] for x in rows_cols]

        return rows, cols

    def project_basis(self, V):
        """Project MDEIM basis unto solution reduced basis.

        Parameters
        ----------
        V : np.array Nh x N
            Reduced basis for solution expansion.

        Creates
        -------
        self.VfN
        """

        Vfh = self.basis_fom
        N = self.N
        self.N_V = V.shape[1]

        VfN = []
        for basis_idx in range(N):

            mat_as_vec = Vfh[:, basis_idx]

            mat = vector_to_csr(
                entries=mat_as_vec,
                rows=self.rows,
                cols=self.cols,
            )
            AN = project_csr(mat, V)

            # Store as a vector, I will reshape when I interpolate
            AN = AN.flatten()
            VfN.append(AN)

        VfN = np.array(VfN).T

        # Previous clean-up to make sure no memory problems arise
        if self.basis_rom is not None:
            del self.basis_rom

        self.basis_rom = VfN

    def assemble_snapshot(self, mu, t):
        """Assemble matrix in CSR format.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        Ah : np.array
        """

        Ah = self._assemble_matrix(mu, t)
        Ah = eliminate_zeros(Ah)

        return Ah.data

    def _assemble_matrix(self, mu, t):
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
        Ah = self.assemble(mu=mu, t=t)
        Ah = bilinear_to_csr(matrix=Ah)
        return Ah

    def interpolate(self, mu, t, which=None):
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
        approximation = super()._interpolate(mu, t, which=which)

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
