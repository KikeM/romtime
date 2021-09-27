from copy import deepcopy
from functools import partial, reduce

import fenics
import numpy as np
from numpy.random import standard_t
from romtime.fom.nonlinear import OneDimensionalBurgers
from dolfin.cpp.la import Matrix, Vector
from romtime.conventions import (
    BDF,
    PistonParameters,
    RomParameters,
    OperatorType,
    Stage,
    Treewalk,
)
from romtime.fom.base import OneDimensionalSolver
from romtime.utils import (
    array_to_function,
    bilinear_to_csr,
    function_to_array,
    functional_to_array,
    project_csr,
)
from scipy.sparse.linalg import gmres
from tqdm import tqdm

from .base import Reductor
from .pod import orth


class RomConstructor(Reductor):

    GMRES_OPTIONS = dict(atol=1e-7, tol=1e-7, maxiter=1e4)

    def __init__(self, fom: OneDimensionalSolver, grid: dict, name=None) -> None:

        super().__init__(grid=grid)

        self.fom = fom
        self.name = name

        self.basis = None

        self.timesteps = dict()
        self.solutions = dict()
        self._solution = np.array

        self.errors = dict()
        self.exact = dict()

        self.deim_fh = None
        self.deim_fgh = None
        self.deim_rhs = None

        self.mdeim_Mh = None
        self.mdeim_Ah = None
        self.mdeim_Ch = None
        self.mdeim_Nh = None
        self.mdeim_Nh_hat = None

    def __del__(self):

        del self.fom
        del self.basis

        del self.timesteps
        del self.solutions

        del self.errors
        del self.exact

        del self.deim_fh
        del self.deim_fgh
        del self.deim_rhs

        del self.mdeim_Mh
        del self.mdeim_Ah
        del self.mdeim_Ch
        del self.mdeim_Nh
        del self.mdeim_Nh_hat

    @property
    def N(self):
        return self.basis.shape[1]

    @property
    def shape(self):
        return self.basis.shape

    def to_fom_vector(self, uN):
        """Build FOM vector from ROM basis functions.

        uh = V uN

        Parameters
        ----------
        uN : np.array

        Returns
        -------
        uh : np.array
        """

        V = self.basis
        uh = V.dot(uN)

        return uh

    def to_rom_vector(self, uh):
        """Project FOM vector into ROM space.

        uN = V^T uh

        Parameters
        ----------
        uh : np.array

        Returns
        -------
        uN : np.array
        """

        V = self.basis
        uh_vec = function_to_array(uh)

        return V.T.dot(uh_vec)

    def to_rom(self, oph):
        """Project FOM operator to ROM space.

        Parameters
        ----------
        oph : dolfin.cpp.la.Matrix or dolfin.cpp.la.Vector
            FOM operator.

        Returns
        -------
        opN : np.array
            ROM operator.
        """

        V = self.basis

        if isinstance(oph, Matrix):
            oph = bilinear_to_csr(oph)
            opN = project_csr(oph, V)
        elif isinstance(oph, Vector):
            oph = functional_to_array(oph)
            opN = V.T.dot(oph)

        return opN

    def load_from_basis(self, basis, mu_space):

        self.basis = deepcopy(basis)
        # ---------------------------------------------------------------------
        # ROM mu space
        mu_space[Stage.ONLINE] = []
        mu_space[Stage.VALIDATION] = []
        self.mu_space = deepcopy(mu_space)

    def truncate(self, n):

        # Create truncated ROM
        truncated = self.__class__(fom=self.fom, grid=self.grid, name=self.name)
        truncated.setup(rnd=self.random_state)

        # Remove modes
        N = self.N
        assert n < N, "You want to remove too many modes from S-ROM to create ROM."
        truncated.basis = self.basis[:, : N - n]

        # Copy data structures
        truncated.mu_space = deepcopy(self.mu_space)
        truncated.report = deepcopy(self.report)
        truncated.report[Stage.OFFLINE][Treewalk.BASIS_FINAL] = truncated.N

        return truncated

    def setup(self, rnd):
        """Prepare reduction structures.

        Parameters
        ----------
        rnd : int
            Random state.
        """

        super().setup(rnd=rnd)

        self.algebraic_solver = self.create_algebraic_solver()

    def add_hyper_reductor(self, reductor, which):
        """Add hyperreductor object for algebraic operators.

        Parameters
        ----------
        reductor : DiscreteEmpiricalInterpolation-like object
            It can be either for a vector or for a matrix.
        which : str
            See RomConstructor conventions.

        Raises
        ------
        NotImplementedError
            If the reductor has not being implemented.
        """

        _reductor = reductor.copy()

        # Functionals
        if which == OperatorType.FORCING:
            self.deim_fh = _reductor
        elif which == OperatorType.LIFTING:
            self.deim_fgh = _reductor
        elif which == OperatorType.RHS:
            self.deim_rhs = _reductor

        #  Matrices
        elif which == OperatorType.MASS:
            self.mdeim_Mh = _reductor
        elif which == OperatorType.STIFFNESS:
            self.mdeim_Ah = _reductor
        elif which == OperatorType.CONVECTION:
            self.mdeim_Ch = _reductor
        elif which == OperatorType.NONLINEAR:
            self.mdeim_Nh = _reductor
        elif which == OperatorType.NONLINEAR_LIFTING:
            self.mdeim_Nh_hat = _reductor
        else:
            raise NotImplementedError(f"Which is this reductor? {which}")

    def project_reductors(self):
        """Project collateral basis unto the solution reduced space."""

        V = self.basis

        if self.deim_fh:
            self.deim_fh.project_basis(V=V)
        if self.deim_fgh:
            self.deim_fgh.project_basis(V=V)
        if self.deim_rhs:
            self.deim_rhs.project_basis(V=V)

        if self.mdeim_Mh:
            self.mdeim_Mh.project_basis(V=V)
        if self.mdeim_Ah:
            self.mdeim_Ah.project_basis(V=V)
        if self.mdeim_Ch:
            self.mdeim_Ch.project_basis(V=V)
        if self.mdeim_Nh:
            self.mdeim_Nh.project_basis(V=V)
        if self.mdeim_Nh_hat:
            self.mdeim_Nh_hat.project_basis(V=V)

    def build_reduced_basis(
        self,
        num_snapshots=None,
        mu_space=None,
        num_basis=None,
        tolerances=dict(),
    ):
        """Build solution reduced basis.

        Parameters
        ----------
        num_snapshots : int
            Number of parameter snapshots to take.
        """

        # Create random sampler
        if num_snapshots:
            space = self.build_sampling_space(num=num_snapshots, rnd=self.random_state)
        elif mu_space:
            space = mu_space
        else:
            raise NotImplementedError(
                "You need to provide a number of mu-snapshots or a space."
            )

        # ---------------------------------------------------------------------
        # Put up the solver and start loop in parameter space
        fom = self.fom
        if fom.is_setup == False:
            fom.setup()

        # ---------------------------------------------------------------------
        # Build Reduced Order Basis (V)
        # For validation phase
        fom_solutions = dict()
        basis_time = []
        tol_t = tolerances.get(RomParameters.TOL_TIME, None)
        for mu in tqdm(space, desc="(ROM) Building reduced basis"):

            # Save parameter
            mu_idx, mu = self.add_mu(mu=mu, step=Stage.OFFLINE)

            # -----------------------------------------------------------------
            # Solve FOM time-dependent problem
            # Is this necessary? Yes, for the probes
            fom.setup()
            fom.update_parametrization(mu)
            fom.solve()

            # -----------------------------------------------------------------
            # Store actual solution
            fom_solutions[mu_idx] = fom._solutions.copy()

            # -----------------------------------------------------------------
            # SVD - Time snapshots
            _basis, sigmas_time, energy_time = orth(fom._snapshots, tol=tol_t)
            basis_time.append(_basis)

            self.report[Stage.OFFLINE][Treewalk.SPECTRUM_TIME][mu_idx] = sigmas_time
            self.report[Stage.OFFLINE][Treewalk.ENERGY_TIME][mu_idx] = energy_time
            self.report[Stage.OFFLINE][Treewalk.BASIS_TIME][mu_idx] = _basis.shape[1]

            if fom.RUNTIME_PROCESS:
                name_probes = f"probes_offline_fom_{mu_idx}.csv"
                fom.save_probes(name=name_probes)

        basis = np.hstack(basis_time)

        self.report[Stage.OFFLINE][Treewalk.BASIS_AFTER_WALK] = basis.shape[1]

        # ---------------------------------------------------------------------
        # Compress all the parameter basis
        tol_mu = tolerances.get(RomParameters.TOL_MU, None)
        basis, sigmas_mu, energy_mu = orth(
            basis,
            num=num_basis,
            tol=tol_mu,
            normalize=False,
        )

        self.report[Stage.OFFLINE][Treewalk.SPECTRUM_MU] = sigmas_mu
        self.report[Stage.OFFLINE][Treewalk.ENERGY_MU] = energy_mu
        self.report[Stage.OFFLINE][Treewalk.BASIS_FINAL] = basis.shape[1]

        self.basis = basis

        assert (
            self.N != 0
        ), f"(ROM) There are no basis vectors. \n See tolerance according to mu-energy: {tolerances[RomParameters.TOL_MU]} < {energy_mu}"

        return fom_solutions

    def create_algebraic_solver(self):
        """Create algebraic solver for reduced problem

        Returns
        -------
        solver : scipy.sparse.linalg.gmres
            Iterative solver with prescribed parameters.
        """

        solver = partial(gmres, **self.GMRES_OPTIONS)

        return solver

    def runtime_process(self, u=None, mu=None, t=None):
        pass

    def solve(self, mu, step):
        """Solve problem with ROM.

        Parameters
        ----------
        mu : dict
            Parameter-point.
        """

        idx_mu, mu = self.add_mu(mu=mu, step=step)

        fom = self.fom

        # Start iteration
        solutions = dict()

        if fom.exact_solution is not None:
            errors = []
            exact = dict()
        else:
            ue = None

        timesteps = [0.0]

        dt = fom.dt
        t = 0.0
        #  TODO : project initial solution
        uN_n = np.zeros(shape=self.N)
        uh = self.to_fom_vector(uN_n)

        BDF_SCHEME = fom.BDF_SCHEME
        if BDF_SCHEME == BDF.TWO:
            uN_n1 = np.zeros_like(uN_n)
        else:
            uN_n1 = None

        # ---------------------------------------------------------------------
        # Solve in time
        solutions_rom = []
        for timestep in tqdm(
            range(fom.domain["nt"]),
            desc="(ROM) Solve in time",
            leave=False,
            miniters=100,
            mininterval=5.0,
        ):

            # Update time
            t += dt

            timesteps.append(t)

            bdf = 1.0
            if (BDF_SCHEME == BDF.TWO) & (timestep > 0):
                bdf = 1.5

            ########################
            # Assemble linear system
            ########################
            MN_mat, KN_mat = self.assemble_system(mu, t, uh, bdf)
            bN_vec = self.assemble_system_rhs(mu, t, MN_mat, uN_n, uN_n1)

            ###############
            # Solve problem
            ###############
            uN, info = self.algebraic_solver(A=KN_mat, b=bN_vec)

            solutions_rom.append(uN)

            # Update solution
            if BDF_SCHEME == BDF.TWO:
                uN_n1 = uN_n.copy()
            uN_n = uN.copy()

            ##############
            # FEM solution
            ##############
            if fom.Lt:
                fom.move_mesh(mu=mu, t=t)
                g, _, _ = fom.create_lifting_operator(mu=mu, t=t, L=fom.L)
                fom.move_mesh(back=True)
            else:
                g, _, _ = fom.create_lifting_operator(mu=mu, t=t, L=fom.domain[fom.L0])

            gh = fom.interpolate_func(g, fom.V, mu, t)
            gh = function_to_array(gh)

            uh = self.to_fom_vector(uN)
            uc_h = uh + gh

            # Collect solutions
            solutions[t] = uc_h.copy()

            # self.runtime_process(u=uc_h, mu=mu, t=t)

            # Compute error with exact solution
            if fom.exact_solution is not None:
                ue = fenics.Expression(fom.exact_solution, degree=1, t=t, **mu)
                ue_h = fom.interpolate_func(ue, fom.V, mu, t)
                ue_h = function_to_array(ue_h)
                exact[t] = ue_h.copy()

                error = self._compute_error(u=uc_h, ue=ue_h)
                errors.append(error)

        self.timesteps = timesteps
        self.solutions.update({idx_mu: solutions})
        _solution = [np.array(uh_n) for uh_n in solutions.values()]
        self._solution = np.vstack(_solution).T
        self.solutions_rom = np.vstack(solutions_rom).T

        if ue is not None:
            self.errors.update({idx_mu: np.array(errors)})
            self.exact.update({idx_mu: exact})

        return idx_mu

    def assemble_system_rhs(self, mu, t, uN_n, MN_mat):

        fN_vec = self.assemble_rhs(mu=mu, t=t)

        dt = self.fom.dt
        bN_vec = MN_mat.dot(uN_n) + dt * fN_vec
        return bN_vec

    def assemble_system(self, mu, t, uh=None):

        MN_mat = self.assemble_mass(mu=mu, t=t)
        AN_mat = self.assemble_stiffness(mu=mu, t=t)

        dt = self.fom.dt
        KN_mat = MN_mat + dt * AN_mat

        return MN_mat, KN_mat

    def assemble_mass(self, mu, t):
        """Assemble reduced mass operator.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        MN : np.array
        """
        if self.mdeim_Mh:
            MN = self.mdeim_Mh.interpolate(mu=mu, t=t, which=self.ROM)
        else:
            Mh = self.fom.assemble_mass(mu, t)
            MN = self.to_rom(Mh)

        return MN

    def assemble_stiffness(self, mu, t):
        """Assemble stiffness operator.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        AN : np.array
        """

        if self.mdeim_Ah:
            AN = self.mdeim_Ah.interpolate(mu=mu, t=t, which=self.ROM)
        else:
            # Assemble FOM operator
            Ah = self.fom.assemble_stiffness(mu, t)
            AN = self.to_rom(Ah)

        return AN

    def assemble_rhs(self, mu, t):
        """Assemble forcing and lifting together.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        fN : np.array
        """

        # Assemble FOM operator
        if self.deim_rhs:
            fN_vec = self.deim_rhs.interpolate(mu=mu, t=t, which=self.ROM)
        else:
            fh = self.fom.assemble_forcing(mu, t)
            fgh = self.fom.assemble_lifting(mu, t)

            fN = self.to_rom(fh)
            fgN = self.to_rom(fgh)
            fN_vec = fN + fgN

        return fN_vec

    def assemble_forcing(self, mu, t):
        """Assemble reduced forcing term.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        fN : np.array
        """

        # Assemble FOM operator
        if self.deim_fh:
            fN = self.deim_fh.interpolate(mu=mu, t=t, which=self.ROM)
        else:
            fh = self.fom.assemble_forcing(mu, t)
            fN = self.to_rom(fh)

        return fN

    def assemble_lifting(self, mu, t):
        """Assemble reduced lifting term.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        fgN : np.array
        """

        if self.deim_fgh:
            fgN = self.deim_fgh.interpolate(mu=mu, t=t, which=self.ROM)
        else:
            fgh = self.fom.assemble_lifting(mu, t)
            fgN = self.to_rom(fgh)

        return fgN


class RomConstructorMoving(RomConstructor):
    def __init__(self, fom: OneDimensionalSolver, grid: dict, name=None) -> None:
        super().__init__(fom=fom, grid=grid, name=name)

    def assemble_convection(self, mu, t):
        """Assemble stiffness operator.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        AN : np.array
        """

        if self.mdeim_Ch:
            CN = self.mdeim_Ch.interpolate(mu=mu, t=t, which=self.ROM)
        else:
            # Assemble FOM operator
            Ch = self.fom.assemble_convection(mu, t)
            CN = self.to_rom(Ch)

        return CN

    def assemble_system(self, mu, t, uh=None):
        """Assemble algebraic ROM system.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        MN_mat : np.array
            Mass matrix.
        KN_mat : np.array
        """

        MN_mat = self.assemble_mass(mu=mu, t=t)
        AN_mat = self.assemble_stiffness(mu=mu, t=t)
        CN_mat = self.assemble_convection(mu=mu, t=t)

        dt = self.fom.dt
        KN_mat = MN_mat + dt * (AN_mat + CN_mat)

        return MN_mat, KN_mat


class RomConstructorNonlinear(RomConstructorMoving):
    def __init__(self, fom: OneDimensionalBurgers, grid: dict, name=None) -> None:
        super().__init__(fom=fom, grid=grid, name=name)

        self.probe_location = fom.probe_location
        self.probes = None

    def build_sampling_space(self, num, rnd=None):
        """Build sampling space according to filling linearity slope.

        Parameters
        ----------
        num : int
        rnd : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """

        print("Building linearity sampling space ...")

        grid = self.grid

        linearity_space = self.compute_linearity_space(grid=grid, num=num)

        sampler = super().build_sampling_space(rnd=rnd, num=1000)

        samples = []
        domains = [
            (start, end) for start, end in zip(linearity_space, linearity_space[1:])
        ]
        for sample in sampler:

            forcing = self.compute_forcing_magnitude(sample)

            remove = None
            for domain in domains:
                start, end = domain

                is_ge = forcing >= start
                is_le = forcing <= end
                inside = is_ge & is_le

                if inside:

                    sample[PistonParameters.FORCING] = forcing
                    samples.append(sample)

                    remove = domain
                    break

            if remove is not None:
                domains.remove(remove)

            if len(domains) == 0:
                break

        # Add sorting so the idx makes sense
        samples = sorted(samples, key=lambda x: x[PistonParameters.FORCING])

        return samples

    @staticmethod
    def compute_forcing_magnitude(sample):

        A0 = PistonParameters.A0
        OMEGA = PistonParameters.OMEGA
        DELTA = PistonParameters.DELTA

        forcing = sample[DELTA] * sample[OMEGA] / sample[A0]

        return forcing

    @staticmethod
    def compute_linearity_space(grid, num):

        A0 = PistonParameters.A0
        OMEGA = PistonParameters.OMEGA
        DELTA = PistonParameters.DELTA

        params = [A0, OMEGA, DELTA]
        support = {}
        for var in params:
            _support = grid[var].support()
            support[var] = {"min": min(_support), "max": max(_support)}

        # Less input into the system, maximum linearity
        linearity_max = (
            support[DELTA]["min"] * support[OMEGA]["min"] / support[A0]["max"]
        )
        # Maximum input into the system, maximum linearity
        linearity_min = (
            support[DELTA]["max"] * support[OMEGA]["max"] / support[A0]["min"]
        )

        # Add some constraints to maximum nonlinearity
        linearity_min *= 0.70

        space = np.linspace(start=linearity_max, stop=linearity_min, num=num + 1)

        return space

    def runtime_process(self, u, mu, t):

        fom = self.fom

        # fom.move_mesh(mu=mu, t=t)
        uh = array_to_function(u, fom.V)
        # fom.move_mesh(back=True)

        num_probs = len(self.probe_location)
        for idx in range(num_probs):
            loc = self.probe_location[idx]
            self.probes[idx].append(uh(loc))

        # Probe at the piston movement
        idx_L = idx + 1
        loc = fom.L - 10.0 * fenics.DOLFIN_EPS
        self.probes[idx_L].append(uh(loc))

    def assemble_system(self, mu, t, uh, bdf=1.0):
        """Assemble algebraic ROM system.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        MN_mat : np.array
            Mass matrix.
        KN_mat : np.array
        """

        MN_mat = self.assemble_mass(mu=mu, t=t)
        AN_mat = self.assemble_stiffness(mu=mu, t=t)
        CN_mat = self.assemble_convection(mu=mu, t=t)
        NN_mat = self.assemble_nonlinear(mu=mu, t=t, uh=uh)
        NhatN_mat = self.assemble_nonlinear_lifting(mu=mu, t=t)

        dt = self.fom.dt
        KN_mat = bdf * MN_mat + dt * (AN_mat + CN_mat + NN_mat + NhatN_mat)

        return MN_mat, KN_mat

    def assemble_system_rhs(self, mu, t, MN_mat, uN_n, uN_n1=None):

        # ---------------------------------------------------------------------
        # No forcing term for Burgers equation ...
        fgN_vec = self.assemble_lifting(mu=mu, t=t)

        # ---------------------------------------------------------------------
        # BDF 1
        if uN_n1 is None:
            bdf = MN_mat.dot(uN_n)
        # ---------------------------------------------------------------------
        # BDF 2
        else:
            u_sum = 2.0 * uN_n - 0.5 * uN_n1
            bdf = MN_mat.dot(u_sum)

        dt = self.fom.dt
        bN_vec = bdf + dt * fgN_vec
        return bN_vec

    def assemble_nonlinear(self, mu, t, uh):
        """Assemble nonlinear convection operator.

        Parameters
        ----------
        mu : dict
        t : float
        uh : ??

        Returns
        -------
        NN : np.array
        """

        if self.mdeim_Nh:
            NN = self.mdeim_Nh.interpolate(mu=mu, t=t, u_n=uh, which=self.ROM)
        else:
            # Assemble FOM operator
            Nh = self.fom.assemble_nonlinear(mu=mu, t=t, u_n=uh)
            NN = self.to_rom(Nh)

        return NN

    def assemble_nonlinear_lifting(self, mu, t):
        """Assemble linearized convection operator containing lifting terms.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        NN_hat : np.array
        """

        if self.mdeim_Nh_hat:
            NN_hat = self.mdeim_Nh_hat.interpolate(mu=mu, t=t, which=self.ROM)
        else:
            # Assemble FOM operator
            Nh_hat = self.fom.assemble_nonlinear_lifting(mu, t)
            NN_hat = self.to_rom(Nh_hat)

        return NN_hat
