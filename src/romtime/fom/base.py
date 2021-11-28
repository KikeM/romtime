from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from functools import wraps
from itertools import product

import fenics
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from romtime.base import SolutionsStorage
from romtime.conventions import BDF, Domain, FIG_KWARGS
from romtime.utils import (
    bilinear_to_csr,
    compute_displacement,
    eliminate_zeros,
    function_to_array,
)
from scipy.sparse import find as get_nonzero_entries
from tqdm import tqdm


def move_mesh(assemble):
    """Decorator to move forth and back the mesh during
    the assembly of the operator.

    Parameters
    ----------
    assemble : OneDimensionalSolver.assemble-like method

    Returns
    -------
    dolfin.cpp.la.Matrix or dolfin.cpp.la.Vector
    """

    @wraps(assemble)
    def _move_mesh(self, mu, t, entries=None, u_n=None):

        self.move_mesh(mu, t)
        if u_n is None:
            operator = assemble(self, mu, t, entries)
        else:
            operator = assemble(self, mu, t, entries, u_n)

        self.move_mesh(back=True)

        return operator

    return _move_mesh


class OneDimensionalSolver(ABC):

    RUNTIME_PROCESS = True

    DIRICHLET_ENTRY = 1.0
    DIRICHLET_VALUE = 0.0

    NX = Domain.NX
    NT = Domain.NT
    L0 = Domain.L0
    T = Domain.T

    B0 = "b0"
    BL = "bL"

    DB0_DT = "db0_dt"
    DBL_DT = "dbL_dt"

    BDF_SCHEME = BDF.TWO

    def __init__(
        self,
        domain=None,
        dirichlet=None,
        parameters=None,
        forcing_term=None,
        u0=None,
        Lt=None,
        dLt_dt=None,
        filename=None,
        degrees=None,
        project_u0=False,
        exact_solution=None,
    ) -> None:

        self.filename = filename
        self.file = None

        self.domain = domain.copy() if domain else None
        self.dirichlet = dirichlet.copy() if dirichlet else None
        self.mu = parameters.copy() if parameters else None
        self.forcing_term = forcing_term
        self.u0 = u0
        self.Lt = Lt
        self.dLt_dt = dLt_dt

        self.degrees = degrees

        self.project_u0 = project_u0

        self._scale = None  # Private variable to be used by the moving mesh method

        # Structures to collect information about the exact solution
        self.exact_solution = exact_solution
        self.exact = None

        # FEM structures
        self.algebraic_solver = None

        # Mappings for local integration
        self.cell_to_dofs = None
        self.dof_to_cells = None
        self.entries_dirichlet = None
        self.dofs_dirichlet = None
        self.ale_x = None

        # Snapshots Collection
        self.solutions = None  # SolutionsStorage

        self.is_setup = False

    def __del__(self):

        del self.filename
        del self.file
        del self.domain
        del self.dirichlet
        del self.mu
        del self.forcing_term
        del self.u0
        del self.Lt
        del self.dLt_dt
        del self.degrees
        del self.project_u0
        del self._scale

        # Structures to collect information about the exact solution
        del self.exact_solution
        del self.exact
        # FEM structures
        del self.algebraic_solver
        # Mappings for local integration
        del self.cell_to_dofs
        del self.dof_to_cells
        del self.entries_dirichlet
        del self.dofs_dirichlet
        del self.ale_x
        # Snapshots Collection
        del self.solutions
        del self.is_setup

    @property
    def x(self) -> np.array:
        """DOF coordinates distribution.

        Returns
        -------
        x : np.array
        """
        return self.V.tabulate_dof_coordinates()

    @property
    def L(self):
        """Domain length.

        Returns
        -------
        L : float
        """
        return np.max(self.x)

    @property
    def scale_solutions(self):
        return 1.0

    @property
    def dt(self):
        # Time domain step size
        dt = self.domain[self.T] / self.domain[self.NT]
        return dt

    @property
    def timesteps(self):
        return self.solutions.ts

    @staticmethod
    def dict_to_array(my_dict):
        return np.array(
            [function_to_array(element) for element in list(my_dict.values())]
        ).T

    def dump_solutions(self, name):
        self.solutions.to_pickle(name)

    def build_cell_to_dofs(self):
        """Create mapping between mesh cells and dofs."""

        mesh = self.mesh
        V = self.V
        dofmap = V.dofmap()
        cells = fenics.cells(mesh)

        cell_to_dofs = dict()
        for cell in cells:
            idx = cell.index()
            coords = cell.get_coordinate_dofs()
            dofs = dofmap.cell_dofs(idx)
            cell_to_dofs[(idx, str(coords), cell)] = list(dofs)

        self.cell_to_dofs = cell_to_dofs

    def build_dofs_to_cells(self):
        """Create mapping between dofs and mesh cells.

        This map is fundamental for the DEIM implementation.
        """

        cell_to_dofs = self.cell_to_dofs
        msg = "Please, build the map cell-to-dofs first."
        assert cell_to_dofs is not None, msg

        dof_to_cells = defaultdict(list)
        for cell, dofs in cell_to_dofs.items():
            for dof in dofs:
                dof_to_cells[dof].append(cell)

        self.dof_to_cells = dof_to_cells

    def _move_mesh(self, displacement=None, back=False):
        """Move mesh according to scale factor.

        Parameters
        ----------
        scale : float, optional
            Coefficient by which we should scale the Interval, by default None
        back : bool, optional
            Rescale the mesh back to the original size, by default False
        """

        # ---------------------------------------------------------------------
        # Remove displacement
        if back == True:

            displacement = self._scale

            # X = x - displacement
            for idx, x in enumerate(self.mesh.coordinates()):
                x[0] -= displacement[idx]

        # ---------------------------------------------------------------------
        # Apply displacement
        elif back == False:

            self._scale = np.copy(displacement)

            # x = X + displacement
            for idx, x in enumerate(self.mesh.coordinates()):
                x[0] += displacement[idx]

    def move_mesh(self, mu=None, t=None, back=False):
        """Move mesh according to displacement field.

        Parameters
        ----------
        mu : dict, optional if `back` is True
        t : float, optional if `back` is True
        back : bool, optional
            Rescale the mesh back to the original size, by default False
        """

        # -----------------------------------------------------------------
        # Move mesh back to reference domain
        if back == True:
            return self._move_mesh(back=back)

        # -----------------------------------------------------------------
        # Compute displacement
        X = self.mesh.coordinates()  # Reference domain

        displacement, _ = compute_displacement(mu, t, X, Lt=self.Lt)

        # -----------------------------------------------------------------
        # Execute mesh movement
        self._move_mesh(displacement=displacement)

    def runtime_process(self, u):
        pass

    def setup(self):
        """Create FEM structures.

        - Mesh
        - Function Spaces
        - x-coordinates
        - Algebraic Solver
        """

        L0 = self.domain[self.L0]

        mesh = fenics.IntervalMesh(self.domain[self.NX], 0.0, L0)
        V = fenics.FunctionSpace(mesh, "P", self.degrees)
        dofmap = V.dofmap()

        # Galerkin projection
        u = fenics.TrialFunction(V=V)
        v = fenics.TestFunction(V=V)

        self.mesh = mesh
        self.V = V
        self.dofmap = dofmap
        self.u = u
        self.v = v
        self.ale_x = deepcopy(mesh.coordinates())

        # Create mappings for (M)DEIM
        self.build_cell_to_dofs()
        self.build_dofs_to_cells()
        self.find_dirichlet_entries()

        if self.filename is not None:
            self.file = fenics.File(self.filename)

        self.algebraic_solver = self.create_algebraic_solver()

        self.is_setup = True

    def assemble_stiffness_topology(self):
        """Assemble stiffness matrix for a ALE problem.

        Parameters
        ----------
        mu : dict
        t : float
        entries : list
        """

        # ---------------------------------------------------------------------
        # Weak Formulation
        # ---------------------------------------------------------------------
        dot, dx, grad = fenics.dot, fenics.dx, fenics.grad
        u, v = self.u, self.v

        Ah = -u.dx(0) * v * dx + dot(grad(u), grad(v)) * dx

        bc = self.define_homogeneous_dirichlet_bc()
        Ah_mat = self.assemble_operator(Ah, bc)

        return Ah_mat

    def find_dirichlet_entries(self):
        """Find Dirichlet BCs entries in matrices and vectors.

        This is necessary for the (M)DEIM procedure.
        """

        # ---------------------------------------------------------------------
        # Assemble the mass matrix
        Mh = OneDimensionalSolver.assemble_mass(self)
        Ah = self.assemble_stiffness_topology()

        Kh = (Mh + Ah) / 2
        Kh = bilinear_to_csr(Kh)
        Kh = eliminate_zeros(Kh)

        # ---------------------------------------------------------------------
        # Find the values equal to one
        rows, cols, values = get_nonzero_entries(Kh)
        mask_ones = np.isclose(values, self.DIRICHLET_ENTRY)
        dirichlet_dofs = [(dof,) for dof in rows[mask_ones]]
        dirichlet_entries = list(
            zip(
                rows[mask_ones],
                cols[mask_ones],
            )
        )

        self.entries_dirichlet = dirichlet_entries
        self.dofs_dirichlet = dirichlet_dofs

    def update_parametrization(self, new):
        """Update parameter vector.

        Parameters
        ----------
        new : dict
            New parameter vector.
        """
        self.mu = new.copy()

    def create_algebraic_solver(self):
        """Create KrylovSolver with default parameters.

        Returns
        -------
        solver : fenics.KrylovSolver
        """

        solver = fenics.KrylovSolver("gmres", "ilu")

        prm = solver.parameters
        prm["absolute_tolerance"] = 1e-10
        prm["relative_tolerance"] = 1e-10
        prm["maximum_iterations"] = int(1e7)

        return solver

    def create_lifting_operator(self, mu, t, L, only_g=False):
        """Create lifting function for the boundary conditions.

        Parameters
        ----------
        mu : dict
        t : float
        L : float

        Returns
        -------
        g : fenics.Expression
        dg_dt : fenics.Expression
        grad_g : fenics.Expression
        """

        dirichlet = self.dirichlet

        b0 = dirichlet[self.B0]
        bL = dirichlet[self.BL]

        db0_dt = dirichlet[self.DB0_DT]
        dbL_dt = dirichlet[self.DBL_DT]

        g = self._compute_linear_interpolation(left=b0, right=bL, mu=mu, t=t, L=L)

        if only_g:
            return g

        # Compute moving boundary effect
        if self.dLt_dt:

            L0 = self.domain[self.L0]

            dLt_dt = self.dLt_dt(t=t, **mu)
            dLt_dt *= L0
            dg_dt = self._compute_linear_interpolation(
                left=db0_dt, right=dbL_dt, mu=mu, t=t, L=L, dLt_dt=dLt_dt
            )

            moving_boundary = self._compute_moving_boundary_effect(
                left=b0, right=bL, mu=mu, t=t, L=L, dLt_dt=dLt_dt
            )

            dg_dt += moving_boundary
        else:
            dg_dt = self._compute_linear_interpolation(
                left=db0_dt, right=dbL_dt, mu=mu, t=t, L=L, dLt_dt=0.0
            )

        grad_g = self._create_lifting_gradient_expression(
            left=b0,
            right=bL,
            mu=mu,
            t=t,
            L=L,
        )

        return g, dg_dt, grad_g

    @staticmethod
    def _compute_linear_interpolation(left, right, mu, t, L, dLt_dt=0.0):
        """Compute linear interpolation for a one-dimensional mesh.

        Parameters
        ----------
        left : fenics.Expression
        right : fenics.Expression
        L : float

        Returns
        -------
        f : fenics.Expression
            Linear interpolation of boundary values (left/right).
        """

        f = fenics.Expression(
            f"({right}) * (x[0] / L) + ({left}) * (L - x[0]) / L",
            degree=2,
            L=L,
            dLt_dt=dLt_dt,
            t=t,
            **mu,
        )

        return f

    @staticmethod
    def _compute_moving_boundary_effect(left, right, mu, t, L, dLt_dt):
        """Compute the moving boundary effect for the linear interpolation function
        in a one-dimensional mesh.

        Parameters
        ----------
        left : fenics.Expression
        right : fenics.Expression
        L : float
        dL_dt : float

        Returns
        -------
        f : fenics.Expression
            Linear interpolation of boundary values (left/right).
        """

        f = fenics.Expression(
            f"({left} - {right}) * (x[0] / L) * (dLt_dt / L)",
            degree=2,
            L=L,
            dLt_dt=dLt_dt,
            t=t,
            **mu,
        )

        return f

    @staticmethod
    def _create_lifting_gradient_expression(left, right, mu, t, L):

        f = fenics.Expression(f"({right} - {left}) / L", degree=2, t=t, L=L, **mu)

        return f

    @staticmethod
    def assemble_operator(weak, bcs):
        """Assemble weak form into algebraic operator
        and apply boundary conditions.

        Parameters
        ----------
        weak : ufl.form.Form
            Weak form UFL description.
        bcs : fenics.DirichletBC
            Dirichlet boundary conditions.

        Returns
        -------
        operator : dolfin.cpp.la.Vector or dolfin.cpp.la.Matrix
            Assembled operator.
        """

        operator = fenics.assemble(weak)
        bcs.apply(operator)

        return operator

    def assemble_local(self, form, entries):
        """Assemble weak form for specific entries.

        Parameters
        ----------
        weak : ufl.form.Form
            Weak form UFL description.
        entries : list
            Entries required to integrate.
        """

        # Get mappings
        cell_to_dofs = self.cell_to_dofs

        # ---------------------------------------------------------------------
        # Detect if we have a vector (I,) or matrix (I,J)
        # ---------------------------------------------------------------------
        is_vector = len(entries[0]) == 1

        entry_integral = []
        for entry in entries:

            # -----------------------------------------------------------------
            # Apply Dirichlet BC if entry is Dirichlet
            # -----------------------------------------------------------------
            # Matrix
            if entry in self.entries_dirichlet:
                contribution = self.DIRICHLET_ENTRY
            # Vector
            elif entry in self.dofs_dirichlet:
                contribution = self.DIRICHLET_VALUE

            # -----------------------------------------------------------------
            # Integrate local cells to obtain contribution
            # -----------------------------------------------------------------
            else:
                cells_to_cover = self.find_cells_to_cover(entry)

                contribution = 0.0
                for cell in cells_to_cover:

                    # Unpack data
                    idx, coords, cell_cpp = cell

                    # Create local element coordinates map
                    dofs_local2global = cell_to_dofs[cell]

                    if is_vector:
                        map_to_local = dofs_local2global
                    else:
                        # Create element matrix coordinates
                        map_to_local = list(
                            product(dofs_local2global, dofs_local2global)
                        )

                    # ---------------------------------------------------------
                    # Assemble local operator
                    # ---------------------------------------------------------
                    if entry in map_to_local:
                        local = fenics.assemble_local(form, cell_cpp)
                        local = local.flatten()

                        # -----------------------------------------------------
                        # Select the entry with the contribution
                        # -----------------------------------------------------
                        local_idx = map_to_local.index(entry)
                        _contribution = local[local_idx]
                        contribution += _contribution
                    else:
                        continue

            entry_integral.append(contribution)

        # Return in the same format
        weak_vec = np.array(entry_integral)

        return weak_vec

    def find_cells_to_cover(self, entry):
        """Find cells covered by the DOFs to integrate locally.

        Parameters
        ----------
        entry : tuple
        dof_to_cells : mapping
            [description]

        Returns
        -------
        cells_to_cover : set of Cells
        """
        dof_to_cells = self.dof_to_cells

        # Select cells spanned by the basis function linked to the DOF
        cells_to_cover = []
        for dof in entry:
            cells_to_cover.extend(dof_to_cells[dof])

        # Remove duplicates
        cells_to_cover = set(cells_to_cover)

        return cells_to_cover

    @abstractmethod
    def assemble_stiffness(self, mu=None, t=None, entries=None):
        """Assemble stiffness matrix terms.

        Parameters
        ----------
        mu : dict
        t : float
        entries : list, optional
            Local integration, by default None
        """
        pass

    def assemble_convection(self, mu=None, t=None, entries=None):
        """Assemble stiffness matrix terms.

        Parameters
        ----------
        mu : dict
        t : float
        entries : list, optional
            Local integration, by default None
        """
        pass

    def assemble_mass(self, mu=None, t=None, entries=None):

        # Extract names to have a clean implementation
        dx = fenics.dx
        u, v = self.u, self.v

        Mh = u * v * dx

        if entries:
            Mh_mat = self.assemble_local(form=Mh, entries=entries)
        else:
            bc = self.define_homogeneous_dirichlet_bc()
            Mh_mat = self.assemble_operator(Mh, bc)

        return Mh_mat

    @abstractmethod
    def assemble_forcing(self, mu, t, entries=None):
        """Assemble forcing terms.

        Parameters
        ----------
        mu : dict
        t : float
        entries : list, optional
            Local integration, by default None
        """
        pass

    @abstractmethod
    def assemble_lifting(self, mu, t, entries=None):
        """Assemble lifting terms to enforce boundary conditions.

        Parameters
        ----------
        mu : dict
        t : float
        dofs : list, optional
            Local integration, by default None
        """
        pass

    def solve(self):
        """Integrate problem in time."""

        V = self.V

        # Create solution with u = span(V)
        uh = fenics.Function(V)
        uc_h = fenics.Function(V)  # uc = hmgns + lift

        t = 0.0
        mu = self.mu

        # ---------------------------------------------------------------------
        g0 = self.create_lifting_operator(mu=mu, t=0.0, L=self.L, only_g=True)
        # Obtain initial solution in the mesh
        # TODO: am I missing the initial condition of the lifting?
        u0 = self.u0
        if self.project_u0:
            u_n = fenics.project(u0, V)
        else:
            u_n = self.interpolate_func(u0, V, mu=mu, t=0.0)
        g0 = self.interpolate_func(g0, V, mu=mu, t=0.0)
        u_n.assign(u_n - g0)

        if self.BDF_SCHEME == BDF.TWO:
            u_n1 = fenics.Function(V)
        else:
            u_n1 = None

        # Time domain step size
        dt = self.dt

        # Prepare iteration
        snapshots = dict()
        solutions = dict()

        timesteps = []
        domain_x = []
        solver = self.algebraic_solver

        # --------------------------------------------------------------------
        # Prepare structures to contrast with exact solution
        # --------------------------------------------------------------------
        if self.exact_solution is not None:
            ue = fenics.Expression(self.exact_solution, degree=2, t=t, **mu)
            errors = dict()
            exact = dict()
        else:
            ue = None

        ts = range(self.domain["nt"])
        for timestep in tqdm(
            ts,
            desc="(FOM) Time integration",
            leave=False,
            miniters=100,
            mininterval=5.0,
        ):

            bdf = 1.0
            if (self.BDF_SCHEME == BDF.TWO) & (timestep > 0):
                bdf = 1.5

            # Update time
            t += dt

            timesteps.append(t)

            # -----------------------------------------------------------------
            # Assemble algebraic problem
            # -----------------------------------------------------------------
            # LHS
            Mh_mat, Kh_mat = self.assemble_system(mu, t, bdf, u_n, u_n1)

            # RHS
            bh_vec = self.assemble_system_rhs(mu, t, Mh_mat, u_n, u_n1)

            # -----------------------------------------------------------------
            # Solve problem
            # -----------------------------------------------------------------
            U = uh.vector()
            solver.solve(Kh_mat, U, bh_vec)

            # Update solution
            if self.BDF_SCHEME == BDF.TWO:
                u_n1.assign(u_n)
            u_n.assign(uh)

            # Prepare lifting function
            self.move_mesh(mu=mu, t=t)
            g = self.create_lifting_operator(mu=mu, t=t, L=self.L, only_g=True)
            domain_x.append(self.x)
            self.move_mesh(back=True)

            # Interpolate lifting and add to build actual solution
            gh = self.interpolate_func(g, V, mu=mu, t=t)
            uc_h.assign(uh + gh)

            if self.RUNTIME_PROCESS:
                self.runtime_process(u=uc_h)

            # -----------------------------------------------------------------
            # Collect solutions
            # -----------------------------------------------------------------
            snapshots[t] = uh.copy(deepcopy=True)
            solutions[t] = uc_h.copy(deepcopy=True)

            # -----------------------------------------------------------------
            # Compute error with exact solution
            # -----------------------------------------------------------------
            if ue is not None:
                ue.t = t
                ue_h = self.interpolate_func(ue, V, mu=mu, t=t)
                exact[t] = ue_h.copy(deepcopy=True)

                error = self._compute_error(u=uc_h, ue=ue_h)
                errors[t] = error

        # ---------------------------------------------------------------------
        # Save results
        solutions = SolutionsStorage(
            ts=timesteps,
            mu=mu,
            domain=np.hstack(domain_x),
            fom=self.dict_to_array(solutions),
            snapshots=self.dict_to_array(snapshots),
        )

        self.solutions = solutions

        if self.Lt:
            self.domain_x = np.hstack(domain_x)
        else:
            self.domain_x = self.x

        if ue is not None:
            self._exact = self.dict_to_array(exact)
            self.errors = errors
            self.exact = exact

    @abstractmethod
    def assemble_system_rhs(self, mu, Mh_mat, t, u_n, u_n1):
        pass

    @abstractmethod
    def assemble_system(self, mu, t, u_n=None):
        pass

    def interpolate_func(self, g, V, mu=None, t=None):
        """Interpolate function in the V space.

        Parameters
        ----------
        g : dolfin.function.function.Function
        V : FunctionSpace
        mu : dict, optional
        t : float, optional

        Returns
        -------
        gh : dolfin.function.function.Function
        """
        gh = fenics.interpolate(g, V)
        return gh

    def define_homogeneous_dirichlet_bc(self):
        """Define homogeneous boundary conditions.

        Returns
        -------
        bc : fenics.DirichletBC
        """

        V = self.V

        # Create boundary conditions
        def boundary(x, on_boundary):
            return on_boundary

        # snapshots boundary conditions
        zero_dirichlet = 0.0
        bc = fenics.DirichletBC(V, zero_dirichlet, boundary)

        return bc

    def _compute_error(self, u, ue, norm_type="L2"):
        """Compute vector error (u - ue).

        Parameters
        ----------
        u : dolfin.function.function.Function
            Numerical solution
        ue : dolfin.function.function.Function
            Solution to compare with.
        norm_type : str, optional
            Vector norm type, by default "L2"
            Accepted values are ["L2", "H1", "max"]

        Returns
        -------
        error : float
            Numerical error
        """

        # Compute maximum norm
        if norm_type == "max":

            mesh = self.mesh
            ue_vertex = ue.compute_vertex_values(mesh=mesh)
            u_vertex = u.compute_vertex_values(mesh=mesh)

            error = np.max(np.abs(ue_vertex - u_vertex))

        else:

            error = fenics.errornorm(
                u=ue,
                uh=u,
                norm_type=norm_type,
                degree_rise=0,
            )

        return error

    def plot_solution(self, pics=6, save=None):
        """Plot solution in space and time.

        Parameters
        ----------
        pics : int, optional
            Number of snapshots to show, by default 6
        """

        plt.figure()

        num = len(self.timesteps) // pics

        texts = []
        for t in range(0, self.domain_x.shape[1], num):

            x = self.domain_x[:, t]
            y = self.scale_solutions * self._solutions[:, t]
            plt.plot(x, y, c="b")

            _t = self.timesteps[t]
            _t = np.round(_t, 1)
            # texts.append(plt.text(x=1.1 * max(x), y=max(y), s=f"$t={_t}$"))

        adjust_text(texts)
        plt.grid()
        plt.xlabel("$x$")
        plt.ylabel("$u(x,t)$")
        plt.title("Solution")

        if save:
            plt.savefig(save, **FIG_KWARGS)
            plt.close()
        else:
            plt.show()

    def plot_errors(self, save=None, log=False, new=True, label=None):
        """Plot errors in time.

        Parameters
        ----------
        pics : int, optional
            Number of snapshots to show, by default 6
        """

        if new:
            plt.figure()

        errors = np.array(list(self.errors.values()))
        if log:
            errors = np.log10(errors)
        plt.plot(self.timesteps, errors, label=label)

        plt.grid()
        plt.xlabel("$t$")
        plt.ylabel("L2 norm")
        plt.title("Errors")

        if save:
            plt.savefig(save, **FIG_KWARGS)
        # else:
        #     plt.show()

    def plot_snapshots(self, pics=6, save=None):
        """Plot solution in space and time.

        Parameters
        ----------
        pics : int, optional
            Number of snapshots to show, by default 6
        """

        plt.figure()

        num = len(self.timesteps) // pics

        texts = []
        for t in range(0, self.domain_x.shape[1], num):

            x = self.domain_x[:, t]
            y = self._snapshots[:, t]
            plt.plot(x, y, c="b")

            _t = self.timesteps[t]
            _t = np.round(_t, 1)
            # texts.append(plt.text(x=1.1 * max(x), y=max(y), s=f"$t={_t}$"))

        adjust_text(texts)
        plt.grid()
        plt.xlabel("$x$")
        plt.ylabel("$\\hat{u}(x,t)$")
        plt.title("Snapshots")
        if save:
            plt.savefig(save, **FIG_KWARGS)
        else:
            plt.show()
