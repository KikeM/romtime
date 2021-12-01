from copy import deepcopy
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from romtime.conventions import (
    EmpiricalInterpolation,
    FIG_KWARGS,
    RomParameters,
    Stage,
    Treewalk,
)
from romtime.rom.base import Reductor
from romtime.rom.pod import orth
from romtime.utils import dump_pickle, functional_to_array, plot, read_pickle
from tqdm import tqdm
import pickle


def basis_vector(size, index):
    ej = np.zeros((size, 1))
    ej[index, 0] = 1.0

    return ej


class DiscreteEmpiricalInterpolation(Reductor):

    TYPE = EmpiricalInterpolation.DEIM

    def __init__(
        self,
        assemble,
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
        self.tree_walk_params = tree_walk_params

        self.N_V = None  # Projection basis size
        self.PT_U = None  # Interpolation matrix
        self.sigmas = None
        self.dofs = None

        self.basis_fom = None
        self.basis_rom = None
        self.snapshots = None

        self.basis_pickle_name = self.__define_basis_name__()

    def __define_basis_name__(self):

        name = "_".join(self.name.lower().split())
        type = self.TYPE.lower()
        return f"basis_fom_{type}_{name}.pkl"

    def __del__(self):

        super().__del__()

        del self.N_V  # Projection basis size
        del self.PT_U  # Interpolation matrix
        del self.sigmas
        del self.dofs

        del self.basis_fom
        del self.basis_rom
        del self.snapshots

    def __str__(self) -> str:
        return f"{self.TYPE} - {self.name}"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def Nh(self):
        return self.basis_fom.shape[0]

    @property
    def N(self):
        return self.basis_fom.shape[1]

    def copy(self):

        new = self.__class__(
            assemble=self.assemble,
            grid=self.grid,
            tree_walk_params=self.tree_walk_params,
            name=self.name,
        )

        # Fill with data structures
        if self.basis_fom is not None:
            new.basis_fom = deepcopy(self.basis_fom)
        if self.basis_rom is not None:
            new.basis_rom = deepcopy(self.basis_rom)
        if self.PT_U is not None:
            new.PT_U = deepcopy(self.PT_U)
        if self.dofs is not None:
            new.dofs = deepcopy(self.dofs)
        if self.errors_rom is not None:
            new.errors_rom = deepcopy(self.errors_rom)

        return new

    def load_fom_basis(self, keep=None, basis=None):
        """Load FOM collateral basis and build interpolation mesh with it.

        Parameters
        ----------
        keep : int or float, optional
            Number of modes to keep from the basis, by default None
        basis : np.array, optional
            External basis to load into the reductor, by default None
        """

        print()
        print("------------------------------------------")
        print(f"Loading (M)DEIM basis for {self.name} ...")

        if basis is None:
            print(f"Loading from disk ...")
            filename = self.basis_pickle_name
            basis = read_pickle(filename)

        print(f"Basis size is {basis.shape}")

        if keep:

            # Allow original base size percentage
            if keep < 1.0:
                N = basis.shape[1]
                keep = np.floor(N * keep)
                keep = int(keep)
                if keep < 1:
                    keep = 1

            print(f"Keeping {keep} elements.")

            basis = basis[:, :keep]

        self.basis_fom = basis
        print(f"(Actual) Basis size is {self.basis_fom.shape}")

        dofs, P = self.build_interpolation_mesh()

        self.store_dofs(dofs)

        # Store interpolation matrix
        self.PT_U = np.matmul(P.T, self.basis_fom)

        # Clean-up
        del P

        print()

    def dump_fom_basis(self, path=None):
        """Dump FOM collateral basis."""

        if self.basis_fom is None:
            assert f"Trying to dump basis for {self.name} without building it!"
        else:
            if path is None:
                filename = self.basis_pickle_name
            else:
                filename = path

            dump_pickle(filename, obj=self.basis_fom)

    def run(self, normalize=True, mu_space=None):
        """Run DEIM offline phase.

        Parameters
        ----------
        num : int
            Number of basis vectors to collect from the snapshots.
        """

        ts = self.tree_walk_params[RomParameters.TS]
        num_snapshots = self.tree_walk_params[RomParameters.NUM_SNAPSHOTS]

        num_mu = self.tree_walk_params.get(RomParameters.NUM_MU, None)
        num_t = self.tree_walk_params.get(RomParameters.NUM_TIME, None)
        tol_mu = self.tree_walk_params.get(RomParameters.TOL_MU, None)
        tol_t = self.tree_walk_params.get(RomParameters.TOL_TIME, None)

        Vfh, sigmas = self.tree_walk(
            ts=ts,
            num_snapshots=num_snapshots,
            num_mu=num_mu,
            num_t=num_t,
            tol_mu=tol_mu,
            tol_t=tol_t,
            normalize=normalize,
            mu_space=mu_space,
        )

        # Store basis and spectrum
        self.basis_fom = Vfh
        self.sigmas = sigmas

        dofs, P = self.build_interpolation_mesh()

        self.store_dofs(dofs)

        # Store interpolation matrix
        self.PT_U = np.matmul(P.T, self.basis_fom)

        # Clean-up
        del P

        # ---------------------------------------------------------------------
        # Print reports
        keys_wanted = [
            Treewalk.BASIS_TIME,
            Treewalk.BASIS_AFTER_WALK,
            Treewalk.BASIS_FINAL,
        ]

        printed = dict(
            (key, self.report[Stage.OFFLINE][key])
            for key in keys_wanted
            if key in self.report[Stage.OFFLINE]
        )

        print(f"For {self.name}")
        pprint(printed)
        print()

    def store_dofs(self, dofs):
        """Store vector entries.

        Parameters
        ----------
        dofs : list of ints
        """
        self.dofs = [(dof,) for dof in dofs]

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

        for mu in tqdm(space, desc=msg_mu, leave=True):

            mu_idx, mu = self.add_mu(step=Stage.ONLINE, mu=mu)

            for t in tqdm(ts, desc=msg_time, leave=False):

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

    def tree_walk(
        self,
        ts,
        normalize=True,
        num_mu=None,
        num_t=None,
        tol_mu=None,
        tol_t=None,
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

        if mu_space:
            space = mu_space
        else:
            space = self.build_sampling_space(num=num_snapshots, rnd=self.random_state)

        basis_time = []
        for mu in tqdm(space, desc=f"({self.TYPE}-{self.name}) Walk in mu", leave=True):

            mu_idx, mu = self.add_mu(step=Stage.OFFLINE, mu=mu)

            # POD in time
            _basis, sigmas_time, energy_time = self.walk_time(
                mu=mu,
                ts=ts,
                num=num_t,
                tol=tol_t,
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

    def walk_time(self, mu, ts, normalize=True, num=None, tol=None):
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

        # Enforce boundary entries do not matter
        # TODO : Generalize boundary elements for MDEIM
        if self.TYPE == EmpiricalInterpolation.MDEIM:
            snapshots[0, :] = 0.0
        basis, sigmas, energy = orth(
            snapshots=snapshots,
            num=num,
            tol=tol,
            normalize=False,
        )

        return basis, sigmas, energy

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
            Vf = self.basis_fom
        elif which == self.ROM:
            Vf = self.basis_rom

        # Local assembly on interpolation mesh
        dofs = self.dofs
        fh_local = self.assemble(mu=mu, t=t, entries=dofs)

        # Compute interpolation coefficients
        thetas = self.compute_thetas(rhs=fh_local)

        # Assemble approximation
        N = self.N
        approximation = np.sum([thetas[i] * Vf[:, i] for i in range(N)], axis=0)

        # Enforce boundary elements
        # TODO : Generalize boundary elements for MDEIM
        if (which == self.FOM) & (self.TYPE == EmpiricalInterpolation.MDEIM):
            approximation[0] = 1.0

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
        return self._interpolate(mu=mu, t=t, which=which)

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

        Vfh = self.basis_fom
        VfN = np.matmul(V.T, Vfh)

        # Previous clean-up to make sure no memory problems arise
        if self.basis_rom is not None:
            del self.basis_rom

        self.basis_rom = VfN

    def build_interpolation_mesh(self):
        """Generate data for empirical interpolation using DEIM algorithm.

        Returns
        -------
        interpolation_dofs : list
            DOFs at which the vectors are interpolated.
        """

        Vf = self.basis_fom
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

    def plot_errors(self, new=True, save=None, show=True):

        if new:
            plt.figure()

        for error in self.errors_rom.values():
            plt.plot(self.tree_walk_params["ts"], np.log10(error))

        plt.grid(True)
        plt.xlabel("$t$")
        plt.ylabel("L2 Error")
        plt.title(f"(M)DEIM {self.name.title()} online errors")

        if show:
            plt.show()

        if save:
            name = f"mdeim_{self.name}_online_errors"
            plt.savefig(name + ".png", **FIG_KWARGS)
            plt.close()

    def plot_spectrum(self, which="sigmas"):

        if which == "sigmas":

            for sigma in self.report[Stage.OFFLINE][self.SPECTRUM_TIME].values():
                plt.plot(np.log10(sigma))

            sigma_mu = self.report[Stage.OFFLINE][self.SPECTRUM_MU]
            sigma_mu = np.log10(sigma_mu)
            plt.plot(sigma_mu, "--")

            title = f"(M)DEIM {self.name.title()} Spectrum Decay"
            ylabel = "$\\sigma$"

        elif which == "energy":

            for energy in self.report[Stage.OFFLINE][self.ENERGY_TIME].values():
                plt.plot(energy)

            energy_mu = self.report[Stage.OFFLINE][self.ENERGY_MU]
            plt.plot(energy_mu, "--")

            title = f"(M)DEIM {self.name.title()} Basis Energy"
            ylabel = "Energy"

        plt.grid(True)
        plt.xlabel("$i-th basis$")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()
