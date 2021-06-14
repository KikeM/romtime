from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from itertools import product

import fenics
import matplotlib.pyplot as plt
import numpy as np
from romtime.utils import bilinear_to_csr, function_to_array
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
    def _move_mesh(self, mu, t, entries=None):

        self.move_mesh(mu, t)
        operator = assemble(self, mu, t, entries)
        self.move_mesh(back=True)

        return operator

    return _move_mesh


class OneDimensionalSolver(ABC):

    DIRICHLET_ENTRY = 1.0
    DIRICHLET_VALUE = 0.0

    NX = "nx"
    NT = "nt"
    L0 = "L0"
    T = "T"

    B0 = "b0"
    BL = "bL"

    DB0_DT = "db0_dt"
    DBL_DT = "dbL_dt"

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
        poly_type=None,
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

        self.poly_type = poly_type
        self.degrees = degrees

        self.project_u0 = project_u0

        self._scale = None  # Private variable to be used by the moving mesh method

        # Structures to collect information about the exact solution
        self.exact_solution = exact_solution
        self.exact = None

        # FEM structures
        self.timesteps = None
        self.algebraic_solver = None

        # Mappings for local integration
        self.cell_to_dofs = None
        self.dof_to_cells = None
        self.entries_dirichlet = None
        self.dofs_dirichlet = None

        # Snapshots Collection
        self.solutions = None  # Actual solutions
        self.snapshots = None  # Homogeneous solutions (for ROM)
        self.liftings = None

        self.is_setup = False

    @property
    def x(self):
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

    def _move_mesh(self, scale=None, back=False):
        """Move mesh according to scale factor.

        Parameters
        ----------
        scale : float, optional
            Coefficient by which we should scale the Interval, by default None
        back : bool, optional
            Rescale the mesh back to the original size, by default False
        """

        if back == True:
            self.mesh.scale(1.0 / self._scale)
        elif back == False:
            self.mesh.scale(scale)
            self._scale = scale

    def move_mesh(self, mu=None, t=None, back=False):
        """Move mesh according to Lt(mu, t).

        Parameters
        ----------
        mu : dict, optional if back is True
        t : float, optional if back is True
        back : bool, optional
            Rescale the mesh back to the original size, by default False
        """

        # If we need to move the mesh back
        if back == True:
            return self._move_mesh(back=back)

        L_t = self.Lt(t=t, **mu)

        self._move_mesh(scale=L_t)

    def setup(self):
        """Create FEM structures.

        - Mesh
        - Function Spaces
        - x-coordinates
        - Algebraic Solver
        """

        L0 = self.domain[self.L0]

        mesh = fenics.IntervalMesh(self.domain[self.NX], 0.0, L0)
        V = fenics.FunctionSpace(mesh, self.poly_type, self.degrees)
        dofmap = V.dofmap()

        # Galerkin projection
        u = fenics.TrialFunction(V=V)
        v = fenics.TestFunction(V=V)

        self.mesh = mesh
        self.V = V
        self.dofmap = dofmap
        self.u = u
        self.v = v

        # Create mappings for (M)DEIM
        self.build_cell_to_dofs()
        self.build_dofs_to_cells()
        self.find_dirichlet_entries()

        if self.filename is not None:
            self.file = fenics.File(self.filename)

        self.algebraic_solver = self.create_algebraic_solver()

        self.is_setup = True

    def find_dirichlet_entries(self):
        """Find Dirichlet BCs entries in matrices and vectors.

        This is necessary for the (M)DEIM procedure.
        """

        # Assemble the mass matrix
        Mh = OneDimensionalSolver.assemble_mass(self)
        Mh = bilinear_to_csr(Mh)
        Mh.eliminate_zeros()

        # Find the values equal to one
        rows, cols, values = get_nonzero_entries(Mh)
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
        prm["absolute_tolerance"] = 1e-7
        prm["relative_tolerance"] = 1e-4
        prm["maximum_iterations"] = 1000

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

    def define_bdf1_forcing(self, u_n):
        """Define Backwards Difference Order 1 forcing terms.

        Parameters
        ----------
        u_n : fenics.Function
            Previous timestep.

        Returns
        -------
        term : ufl.form.Form
        """

        # Extract names to have a clean implementation
        dx = fenics.dx
        v = self.v

        term = u_n * v * dx

        return term

    def define_bdf2_forcing(self, u_n, u_n1):
        """Define Backwards Difference Order 2 forcing terms.

        Parameters
        ----------
        u_n : fenics.Function
            Previous timestep, u^{n}.

        u_n1 : fenics.Function
            Second previous timestep, u^{n-1}.

        Returns
        -------
        term : ufl.form.Form
        """

        # Extract names to have a clean implementation
        dx = fenics.dx
        v = self.v

        raise NotImplementedError("BDF-2 method to be implemented.")

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
    def assemble_stiffness(self, mu, t, entries=None):
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

        # Obtain initial solution in the mesh
        u0 = self.u0
        if self.project_u0:
            u_n = fenics.project(u0, V)
        else:
            u_n = self.interpolate_func(u0, V, mu=mu, t=t)

        # Time domain step size
        dt = self.domain[self.T] / self.domain[self.NT]
        self.dt = dt

        # Prepare iteration
        snapshots = dict()
        solutions = dict()
        liftings = dict()

        timesteps = [0.0]
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
        for timestep in tqdm(ts, desc="(FOM) Time integration", leave=False):

            # Update time
            t += dt

            # Prepare lifting function
            if self.Lt:
                self.move_mesh(mu=mu, t=t)
                g = self.create_lifting_operator(mu=mu, t=t, L=self.L, only_g=True)

                domain_x.append(self.x)
                self.move_mesh(back=True)
            else:
                g = self.create_lifting_operator(mu=mu, t=t, L=self.L, only_g=True)

            timesteps.append(t)

            # -----------------------------------------------------------------
            # Assemble algebraic problem
            # -----------------------------------------------------------------
            # LHS
            Mh_mat = self.assemble_mass(mu=mu, t=t)
            Ah_mat = self.assemble_stiffness(mu=mu, t=t)
            Kh_mat = Mh_mat + dt * Ah_mat

            # RHS
            fh_vec = self.assemble_forcing(mu=mu, t=t)
            fgh_vec = self.assemble_lifting(mu=mu, t=t)

            bdf_1 = Mh_mat * u_n.vector()
            bh_vec = bdf_1 + dt * (fh_vec + fgh_vec)

            # -----------------------------------------------------------------
            # Solve problem
            # -----------------------------------------------------------------
            U = uh.vector()
            solver.solve(Kh_mat, U, bh_vec)

            # Update solution
            u_n.assign(uh)

            # Interpolate lifting and add to build actual solution
            gh = self.interpolate_func(g, V, mu=mu, t=t)
            uc_h.assign(uh + gh)

            # -----------------------------------------------------------------
            # Collect solutions
            # -----------------------------------------------------------------
            snapshots[t] = uh.copy(deepcopy=True)
            solutions[t] = uc_h.copy(deepcopy=True)
            liftings[t] = gh.copy(deepcopy=True)

            # -----------------------------------------------------------------
            # Compute error with exact solution
            # -----------------------------------------------------------------
            if ue is not None:
                ue.t = t
                ue_h = self.interpolate_func(ue, V, mu=mu, t=t)
                exact[t] = ue_h.copy(deepcopy=True)

                error = self._compute_error(u=uc_h, ue=ue_h)
                errors[t] = error

        # Save results
        self.timesteps = timesteps
        self.solutions = solutions
        self.liftings = liftings
        self.snapshots = snapshots

        # Collect snapshots as actual arrays
        self._snapshots = self.dict_to_array(snapshots)
        self._solutions = self.dict_to_array(solutions)
        self._exact = self.dict_to_array(exact)

        if self.Lt:
            self.domain_x = np.hstack(domain_x)
        else:
            self.domain_x = self.x

        if ue is not None:
            self.errors = errors
            self.exact = exact

    @staticmethod
    def dict_to_array(my_dict):
        return np.array(
            [function_to_array(element) for element in list(my_dict.values())]
        ).T

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
        zero_dirichlet = fenics.Constant(0.0)
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
