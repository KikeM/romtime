from abc import ABC, abstractmethod
from collections import defaultdict

import fenics
import numpy as np
from tqdm import tqdm

from romtime.utils import function_to_array
from itertools import product


class OneDimensionalSolver(ABC):
    def __init__(
        self,
        domain=None,
        dirichlet=None,
        parameters=None,
        forcing_term=None,
        u0=None,
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

        self.poly_type = poly_type
        self.degrees = degrees

        self.project_u0 = project_u0

        # Structures to collect information about the exact solution
        self.exact_solution = exact_solution
        self.exact = None

        # FEM structures
        self.x = None  # Node distribution
        self.timesteps = None
        self.algebraic_solver = None

        # Mappings for local integration
        self.cell_to_dofs = None
        self.dof_to_cells = None

        # Snapshots Collection
        self.solutions = None  # Actual solutions
        self.snapshots = None  # Homogeneous solutions (for ROM)
        self.liftings = None

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

    def setup(self):
        """Create FEM structures.

        - Mesh
        - Function Spaces
        - x-coordinates
        - Algebraic Solver
        """

        L = self.domain["L"].values().item()

        mesh = fenics.IntervalMesh(self.domain["nx"], 0.0, L)
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
        self.x = V.tabulate_dof_coordinates()

        # Create mappings for (M)DEIM
        self.build_cell_to_dofs()
        self.build_dofs_to_cells()

        if self.filename is not None:
            self.file = fenics.File(self.filename)

        self.algebraic_solver = self.create_algebraic_solver()

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

    def create_lifting_operator(self, mu, t, L):
        """Create lifting function for the boundary conditions.

        Parameters
        ----------
        b0 : fenics.Expression
            Function value at x = 0.
        bL : fenics.Expression
            Function value at x = L(t).

        Returns
        -------
        g : fenics.Expression
        dg_dt : fenics.Expression
        grad_g : fenics.Expression
        """

        dirichlet = self.dirichlet

        b0 = dirichlet["b0"]
        bL = dirichlet["bL"]

        db0_dt = dirichlet["db0_dt"]
        dbL_dt = dirichlet["dbL_dt"]

        g = self._compute_linear_interpolation(left=b0, right=bL, mu=mu, t=t, L=L)
        dg_dt = self._compute_linear_interpolation(
            left=db0_dt, right=dbL_dt, mu=mu, t=t, L=L
        )
        grad_g = self._create_lifting_gradient_expression(
            left=b0, right=bL, mu=mu, t=t, L=L
        )

        return g, dg_dt, grad_g

    @staticmethod
    def _compute_linear_interpolation(left, right, mu, t, L):
        """Compute linear interpolation for a one-dimensional mesh.

        Parameters
        ----------
        left : fenics.Expression
        right : fenics.Expression
        L : [type]
            [description]

        Returns
        -------
        f : fenics.Expression
            Linear interpolation of boundary values (left/right).
        """

        f = fenics.Expression(
            f"{right} * (x[0] / L) + {left} * (L - x[0]) / L",
            degree=2,
            L=L,
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
        """Assemble a weak form for some specific local entries.

        Parameters
        ----------
        weak : ufl.form.Form
            Weak form UFL description.
        entries : list
            Entries required to integrate.
        """

        # Get mappings
        dof_to_cells = self.dof_to_cells
        cell_to_dofs = self.cell_to_dofs

        is_vector = len(entries[0]) == 1

        entry_integral = []
        for entry in entries:

            # Select cells spanned by the basis function linked to the DOF
            cells_to_cover = []
            for dof in entry:
                cells_to_cover.extend(dof_to_cells[dof])

            # Remove duplicates
            cells_to_cover = set(cells_to_cover)

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
                    map_to_local = list(product(dofs_local2global, dofs_local2global))

                if entry in map_to_local:
                    # Assemble local operator
                    local = fenics.assemble_local(form, cell_cpp)
                    local = local.flatten()

                    # Select the entry with the contribution
                    local_idx = map_to_local.index(entry)
                    _contribution = local[local_idx]
                    contribution += _contribution
                else:
                    continue

            entry_integral.append(contribution)

        # Return in the same format
        weak_vec = np.array(entry_integral)

        return weak_vec

    @abstractmethod
    def assemble_stiffness(self, mu, t, dofs=None):
        """Assemble stiffness matrix terms.

        Parameters
        ----------
        mu : dict
        t : float
        dofs : list, optional
            Local integration, by default None
        """
        pass

    @abstractmethod
    def assemble_mass(self, mu, t, dofs=None):
        """Assemble mass matrix.

        Parameters
        ----------
        mu : dict
        t : float
        dofs : list, optional
            Local integration, by default None
        """
        pass

    @abstractmethod
    def assemble_forcing(self, mu, t, dofs=None):
        """Assemble forcing terms.

        Parameters
        ----------
        mu : dict
        t : float
        dofs : list, optional
            Local integration, by default None
        """
        pass

    @abstractmethod
    def assemble_lifting(self, mu, t, dofs=None):
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

        # Obtain initial solution in the mesh
        u0 = self.u0
        if self.project_u0:
            u_n = fenics.project(u0, V)
        else:
            u_n = fenics.interpolate(u0, V)

        # Time domain step size
        dt = self.domain["T"] / self.domain["nt"]
        self.dt = dt

        # Time step discretization
        prev = self.define_bdf1_forcing(u_n=u_n)

        # Homogeneous boundary conditions
        bc = self.define_homogeneous_dirichlet_bc()

        # Prepare iteration
        snapshots = dict()
        solutions = dict()
        liftings = dict()

        timesteps = [0.0]
        solver = self.algebraic_solver
        mu = self.mu

        #######################################################################
        # Prepare structures to contrast with exact solution
        #######################################################################
        if self.exact_solution is not None:
            ue = fenics.Expression(self.exact_solution, degree=2, t=0.0, **mu)
            errors = dict()
            exact = dict()
        else:
            ue = None

        #  Prepare lifting function
        g, _, _ = self.create_lifting_operator(mu=mu, t=0.0, L=self.domain["L"])

        t = 0.0
        for timestep in tqdm(
            range(self.domain["nt"]), desc="(FOM) Time integration", leave=False
        ):

            # Update time
            t += dt
            g.t = t

            timesteps.append(t)

            ###################################################################
            # Assemble algebraic problem
            ###################################################################
            # LHS
            Mh_mat = self.assemble_mass(mu=mu, t=t)
            Ah_mat = self.assemble_stiffness(mu=mu, t=t)
            Kh_mat = Mh_mat + dt * Ah_mat

            # RHS
            fh_vec = self.assemble_forcing(mu=mu, t=t)
            fgh_vec = self.assemble_lifting(mu=mu, t=t)

            bdf_1 = self.assemble_operator(prev, bc)
            bh_vec = bdf_1 + dt * (fh_vec + fgh_vec)

            ###################################################################
            # Solve problem
            ###################################################################
            U = uh.vector()
            solver.solve(Kh_mat, U, bh_vec)

            # Update solution
            u_n.assign(uh)

            # Interpolate lifting and add to build actual solution
            gh = fenics.interpolate(g, V)
            uc_h.assign(uh + gh)

            ###################################################################
            # Collect solutions
            ###################################################################
            snapshots[t] = uh.copy(deepcopy=True)
            solutions[t] = uc_h.copy(deepcopy=True)
            liftings[t] = gh.copy(deepcopy=True)

            ###################################################################
            # Compute error with exact solution
            ###################################################################
            if ue is not None:
                ue.t = t
                ue_h = fenics.interpolate(ue, V)
                exact[t] = ue_h.copy(deepcopy=True)

                error = self._compute_error(u=uc_h, ue=ue_h)
                errors[t] = error

        # Save results
        self.timesteps = timesteps
        self.solutions = solutions
        self.liftings = liftings
        self.snapshots = snapshots

        # Collect snapshots as actual arrays
        self._snapshots = np.array(
            [function_to_array(element) for element in list(snapshots.values())]
        ).T

        if ue is not None:
            self.errors = errors
            self.exact = exact

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
