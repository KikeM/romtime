import fenics
import numpy as np
from tqdm import tqdm

from romtime.utils import function_to_array


class OneDimensionalHeatEquationSolver:
    def __init__(
        self,
        domain: dict,
        dirichlet: dict,
        parameters: dict,
        forcing_term,
        u0,
        filename="output.pvd",
        poly_type="P",
        degrees=1,
        project_u0=False,
        exact_solution=None,
    ) -> None:

        self.filename = filename
        self.file = None

        self.domain = domain.copy()
        self.dirichlet = dirichlet.copy()
        self.parameters = parameters.copy()
        self.forcing_term = forcing_term
        self.u0 = u0

        self.poly_type = poly_type
        self.degrees = degrees

        self.project_u0 = project_u0

        self.exact_solution = exact_solution

        self.algebraic_solver = None
        self.weak_formulation = None

        # FEM structures
        self.x = None  # Node distribution
        self.timesteps = None
        self.alpha = None  # Nonlinear diffusion coefficient

        self.a = None  # Stiffness matrix
        self.F = None  # Rhs

        self.g = None  # Lifting function
        self.dg_dt = None  # Lifting function time derivative
        self.f_g = None  #  Lifting forcing term

        self.solutions = None
        self.snapshots = None  # Homogeneous solutions
        self.liftings = None
        self.exact = None

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

        # Galerkin projection
        u = fenics.TrialFunction(V=V)
        v = fenics.TestFunction(V=V)

        self.mesh = mesh
        self.V = V
        self.u = u
        self.v = v
        self.x = V.tabulate_dof_coordinates()

        if self.filename is not None:
            self.file = fenics.File(self.filename)

        self.algebraic_solver = self.create_algebraic_solver()

    def update_parameters(self, new):

        self.parameters = new.copy()

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

    def create_diffusion_coefficient(self, mu):
        """Create non-linear diffusion term.

        \alpha(x) = \alpha_0 (1 + \varepsilon x^2)

        Returns
        -------
        alpha : fenics.Expression
        """

        alpha_0 = mu["alpha_0"]
        epsilon = mu["epsilon"]

        alpha = fenics.Expression(
            "alpha_0 * (1.0 + epsilon * x[0] * x[0])",
            degree=2,
            alpha_0=alpha_0,
            epsilon=epsilon,
        )

        return alpha

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
        """Assemble weak form into algebraic operator.

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

    def assemble_stiffness(self, mu, t):

        # Extract names to have a clean implementation
        dot, dx, grad = fenics.dot, fenics.dx, fenics.grad
        u, v = self.u, self.v

        alpha = self.create_diffusion_coefficient(mu)
        Ah = alpha * dot(grad(u), grad(v)) * dx

        bc = self.define_homogeneous_dirichlet_bc()
        Ah_mat = self.assemble_operator(Ah, bc)

        return Ah_mat

    def assemble_mass(self, mu, t):

        # Extract names to have a clean implementation
        dx = fenics.dx
        u, v = self.u, self.v

        Mh = u * v * dx

        bc = self.define_homogeneous_dirichlet_bc()
        Mh_mat = self.assemble_operator(Mh, bc)

        return Mh_mat

    def assemble_forcing(self, mu, t):

        # Extract names to have a clean implementation
        dx = fenics.dx
        v = self.v
        forcing_term = self.forcing_term
        f = fenics.Expression(forcing_term, degree=2, t=t, **mu)

        fh = f * v * dx

        bc = self.define_homogeneous_dirichlet_bc()
        fh_vec = self.assemble_operator(fh, bc)

        return fh_vec

    def assemble_lifting(self, mu, t):

        alpha = self.create_diffusion_coefficient(mu)

        # Extract names to have a clean implementation
        dot, dx, grad = fenics.dot, fenics.dx, fenics.grad
        v = self.v

        L = self.domain["L"]
        _, dg_dt, grad_g = self.create_lifting_operator(mu=mu, t=t, L=L)

        # Small hack to handle fenics internals
        grad_g_vec = fenics.as_vector((grad_g,))

        # lifting = -(dg_dt * v + dot(f_g_vec, grad(v))) * dx
        fgh_time = -(dg_dt * v) * dx
        fgh_grad = -(alpha * dot(grad_g_vec, grad(v))) * dx

        bc = self.define_homogeneous_dirichlet_bc()

        fgh_time_vec = self.assemble_operator(fgh_time, bc)
        fgh_grad_vec = self.assemble_operator(fgh_grad, bc)

        return fgh_time_vec, fgh_grad_vec

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

        # Start iteration
        snapshots = dict()
        solutions = dict()
        liftings = dict()

        if self.exact_solution is not None:
            ue = fenics.Expression(
                self.exact_solution, degree=2, t=0.0, **self.parameters
            )
            errors = dict()
            exact = dict()
        else:
            ue = None

        timesteps = [0.0]
        solver = self.algebraic_solver

        g, _, _ = self.create_lifting_operator(
            mu=self.parameters, t=0.0, L=self.domain["L"]
        )

        mu = self.parameters

        t = 0.0
        for timestep in tqdm(range(self.domain["nt"]), leave=False):

            # Update time
            t += dt
            g.t = t

            timesteps.append(t)

            ########################
            # Assemble linear system
            ########################
            Mh_mat = self.assemble_mass(mu=mu, t=t)
            Ah_mat = self.assemble_stiffness(mu=mu, t=t)
            Kh_mat = Mh_mat + dt * Ah_mat

            fh_vec = self.assemble_forcing(mu=mu, t=t)
            fgh_time, fgh_grad = self.assemble_lifting(mu=mu, t=t)
            fgh_vec = fgh_time + fgh_grad
            prev_vec = self.assemble_operator(prev, bc)
            bh_vec = prev_vec + dt * (fh_vec + fgh_vec)

            ###############
            # Solve problem
            ###############
            U = uh.vector()
            solver.solve(Kh_mat, U, bh_vec)

            # Update solution
            u_n.assign(uh)

            # Interpolate lifting and add to build actual solution
            gh = fenics.interpolate(g, V)
            uc_h.assign(uh + gh)

            # Collect solutions
            snapshots[t] = uh.copy(deepcopy=True)
            solutions[t] = uc_h.copy(deepcopy=True)
            liftings[t] = gh.copy(deepcopy=True)

            # Compute error with exact solution
            if ue is not None:
                ue.t = t
                ue_h = fenics.interpolate(ue, V)
                exact[t] = ue_h.copy(deepcopy=True)

                error = self._compute_error(u=uc_h, ue=ue_h)
                errors[t] = error

        self.timesteps = timesteps

        self.solutions = solutions
        self.snapshots = snapshots
        self.liftings = liftings

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
