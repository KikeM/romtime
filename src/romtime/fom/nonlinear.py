import fenics
import numpy as np
from romtime.utils import function_to_array, plot, plt

from .base import OneDimensionalSolver, move_mesh


class OneDimensionalBurgersConventions:

    A0 = "a0"
    DELTA = "delta"
    GAMMA = "gamma"


class OneDimensionalBurgers(OneDimensionalSolver):
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
        Lt=None,
        dLt_dt=None,
    ) -> None:

        super().__init__(
            domain=domain,
            dirichlet=dirichlet,
            parameters=parameters,
            forcing_term=forcing_term,
            u0=u0,
            filename=filename,
            poly_type=poly_type,
            degrees=degrees,
            project_u0=project_u0,
            exact_solution=exact_solution,
            Lt=Lt,
            dLt_dt=dLt_dt,
        )

        # FEM structures
        self.alpha = None  # Nonlinear diffusion coefficient

        #  Derived results
        self.mc = None  # Mass conservation in time
        self.outflow = None

    @property
    def scale_solutions(self):
        return self.mu[OneDimensionalBurgersConventions.A0]

    @property
    def nonlinear_coefficient(self):
        a0 = self.mu[OneDimensionalBurgersConventions.A0]
        gamma = self.mu[OneDimensionalBurgersConventions.GAMMA]
        A = (gamma + 1.0) / 2.0

        coeff = A * a0
        return coeff

    @staticmethod
    def _compute_linear_interpolation(right, mu, t, L, dLt_dt=0.0):
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
            f"({right}) * (x[0] / L)",
            degree=2,
            L=L,
            dLt_dt=dLt_dt,
            t=t,
            **mu,
        )

        return f

    @staticmethod
    def _create_lifting_gradient_expression(right, mu, t, L):

        f = fenics.Expression(f"({right}) / L", degree=2, t=t, L=L, **mu)

        return f

    def create_lifting_operator(self, mu, t, L, only_g=False):
        """Create lifting function for the boundary condition.

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
        bL = dirichlet[self.BL]
        dbL_dt = dirichlet[self.DBL_DT]

        g = self._compute_linear_interpolation(right=bL, mu=mu, t=t, L=L)

        if only_g:
            return g

        # Compute moving boundary effect
        if self.dLt_dt:

            L0 = self.domain[self.L0]

            dLt_dt = self.dLt_dt(t=t, **mu)
            dLt_dt *= L0
            dg_dt = self._compute_linear_interpolation(
                right=dbL_dt, mu=mu, t=t, L=L, dLt_dt=dLt_dt
            )

        else:
            dg_dt = self._compute_linear_interpolation(
                right=dbL_dt, mu=mu, t=t, L=L, dLt_dt=0.0
            )

        grad_g = self._create_lifting_gradient_expression(
            right=bL,
            mu=mu,
            t=t,
            L=L,
        )

        return g, dg_dt, grad_g

    def define_homogeneous_dirichlet_bc(self):
        """Define right-only homogeneous boundary conditions.

        Returns
        -------
        bc : fenics.DirichletBC
        """
        TOL = 1e-14

        V = self.V

        # Create boundary conditions
        def boundary_R(x, on_boundary):
            is_0 = fenics.near(x[0], 0, TOL)
            is_R = on_boundary & (not is_0)
            return is_R

        # snapshots boundary conditions
        zero_dirichlet = fenics.Constant(0.0)
        bc = fenics.DirichletBC(V, zero_dirichlet, boundary_R)

        return bc

    def create_diffusion_coefficient(self, mu=None):
        """Create non-linear diffusion term.

        \\alpha(x) = \\alpha_0 (1 + \\varepsilon x^2)

        Returns
        -------
        alpha : fenics.Expression
        """

        alpha = mu["alpha"]
        alpha = fenics.Expression("alpha", degree=1, alpha=alpha)

        return alpha

    def compute_mesh_velocity(self, mu, t):
        """Compute mesh velocity.

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        w : fenics.Expression
        """
        dLt_dt = self.dLt_dt(t=t, **mu)
        Lt = self.Lt(t=t, **mu)

        w = fenics.Expression(
            "x[0] * dLt_dt / Lt",
            degree=2,
            dLt_dt=dLt_dt,
            Lt=Lt,
            **mu,
        )

        return w

    def interpolate_func(self, g, V, mu, t):
        """Interpolate function in the V space moving the mesh accordingly.

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

        self.move_mesh(mu=mu, t=t)
        gh = fenics.interpolate(g, V)
        self.move_mesh(back=True)
        return gh

    def assemble_system(self, mu, t, u_n=None):

        Mh_mat = self.assemble_mass(mu=mu, t=t)
        Ah_mat = self.assemble_stiffness(mu=mu, t=t)
        Ch_mat = self.assemble_nonlinear(mu=mu, t=t, u_n=u_n)
        Chat_mat = self.assemble_convection_lifting(mu=mu, t=t)
        Bh_mat = self.assemble_convection(mu=mu, t=t)

        dt = self.dt

        Kh_mat = Mh_mat + dt * (Ah_mat + Bh_mat + Ch_mat + Chat_mat)

        return Mh_mat, Kh_mat

    def assemble_system_rhs(self, mu, t, u_n, Mh_mat):

        #  fh_vec = self.assemble_forcing(mu=mu, t=t)
        fgh_vec = self.assemble_lifting(mu=mu, t=t)

        bdf_1 = Mh_mat * u_n.vector()

        dt = self.dt
        bh_vec = bdf_1 + dt * fgh_vec

        return bh_vec

    # -------------------------------------------------------------------------
    # LHS terms
    @move_mesh
    def assemble_mass(self, mu, t, entries=None):
        return super().assemble_mass(mu=mu, t=t, entries=entries)

    @move_mesh
    def assemble_stiffness(self, mu, t, entries=None):

        # ---------------------------------------------------------------------
        # Weak Formulation
        # ---------------------------------------------------------------------
        dot, dx, grad = fenics.dot, fenics.dx, fenics.grad
        u, v = self.u, self.v

        alpha = self.create_diffusion_coefficient(mu)
        Ah = alpha * dot(grad(u), grad(v)) * dx

        if entries:
            Ah_mat = self.assemble_local(form=Ah, entries=entries)
        else:
            bc = self.define_homogeneous_dirichlet_bc()
            Ah_mat = self.assemble_operator(Ah, bc)

        return Ah_mat

    @move_mesh
    def assemble_nonlinear(self, mu, t, entries=None, u_n=None):

        # ---------------------------------------------------------------------
        # Weak Formulation
        # ---------------------------------------------------------------------
        u, v, dx = self.u, self.v, fenics.dx

        b0 = self.nonlinear_coefficient

        Ch = b0 * u_n * u.dx(0) * v * dx

        if entries:
            Ch_mat = self.assemble_local(form=Ch, entries=entries)
        else:
            bc = self.define_homogeneous_dirichlet_bc()
            Ch_mat = self.assemble_operator(Ch, bc)

        return Ch_mat

    @move_mesh
    def assemble_convection_lifting(self, mu, t, entries=None):

        # ---------------------------------------------------------------------
        # Weak Formulation
        u, v, dx = self.u, self.v, fenics.dx

        L = self.L
        g, _, grad_g = self.create_lifting_operator(mu=mu, t=t, L=L)

        b0 = self.nonlinear_coefficient
        s1 = g * u.dx(0)
        s2 = grad_g * u
        Ch = b0 * (s1 + s2) * v * dx

        if entries:
            Ch_mat = self.assemble_local(form=Ch, entries=entries)
        else:
            bc = self.define_homogeneous_dirichlet_bc()
            Ch_mat = self.assemble_operator(Ch, bc)

        return Ch_mat

    @move_mesh
    def assemble_convection(self, mu, t, entries=None):
        """Force convection due to mesh velocity and speed of sound propagation.

        Parameters
        ----------
        mu : dict
        t : float
        entries : list, optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """

        # ---------------------------------------------------------------------
        # Weak Formulation
        u, v, dx = self.u, self.v, fenics.dx

        a0 = mu[OneDimensionalBurgersConventions.A0]
        w = self.compute_mesh_velocity(mu=mu, t=t)

        Ch = -(a0 + w) * u.dx(0) * v * dx

        if entries:
            Ch_mat = self.assemble_local(form=Ch, entries=entries)
        else:
            bc = self.define_homogeneous_dirichlet_bc()
            Ch_mat = self.assemble_operator(Ch, bc)

        return Ch_mat

    @move_mesh
    def assemble_forcing(self, mu, t, entries=None):
        """Assemble forcing vector.

        Parameters
        ----------
        mu : dict
        t : float
        entries : list of tuples, optional
            Local entries to assemble when using DEIM techniques, by default None

        Returns
        -------
        """

        # ---------------------------------------------------------------------
        # Weak Formulation
        # ---------------------------------------------------------------------
        dx = fenics.dx
        v = self.v
        forcing_term = self.forcing_term
        f = fenics.Expression(forcing_term, degree=2, t=t, **mu)

        fh = f * v * dx

        if entries:
            fh_vec = self.assemble_local(form=fh, entries=entries)
        else:
            bc = self.define_homogeneous_dirichlet_bc()
            fh_vec = self.assemble_operator(fh, bc)

        return fh_vec

    @move_mesh
    def assemble_lifting(self, mu, t, entries=None):
        """Assemble lifting vector.

        Parameters
        ----------
        mu : dict
        t : float
        entries : list of tuples, optional
            Local entries to assemble when using DEIM techniques, by default None

        Returns
        -------
        """

        # ---------------------------------------------------------------------
        # Weak Formulation
        # ---------------------------------------------------------------------
        v, dx = self.v, fenics.dx
        L = self.L
        g, dg_dt, grad_g = self.create_lifting_operator(mu=mu, t=t, L=L)

        # ---------------------------------------------------------------------
        # Inertia
        time_derivative = dg_dt * v * dx

        # ---------------------------------------------------------------------
        # Inertia nonlinear
        b0 = self.nonlinear_coefficient
        nonlinear = b0 * g * grad_g * v * dx

        # ---------------------------------------------------------------------
        # Convection
        w = self.compute_mesh_velocity(mu=mu, t=t)
        a0 = self.mu[OneDimensionalBurgersConventions.A0]
        convection = (a0 + w) * grad_g * v * dx

        # ---------------------------------------------------------------------
        # Diffusion
        alpha = self.create_diffusion_coefficient(mu)
        diffusion = alpha * grad_g * v.dx(0) * dx

        # ---------------------------------------------------------------------
        # Aggregate effects
        fgh = -(time_derivative + nonlinear + diffusion) + convection

        # ---------------------------------------------------------------------
        # Assembly
        # ---------------------------------------------------------------------
        if entries:
            fgh_vec = self.assemble_local(form=fgh, entries=entries)
        else:
            bc = self.define_homogeneous_dirichlet_bc()
            fgh_vec = self.assemble_operator(fgh, bc)

        return fgh_vec

    @staticmethod
    def compute_rho(u, gamma):

        # Velocity scaling
        A = gamma - 1.0
        A /= 2.0

        # Exponent
        exp = 2.0 / (gamma - 1)

        rho = (1.0 - A * u) ** (exp)

        return rho

    @staticmethod
    def compute_p(u, gamma):

        # Velocity scaling
        A = gamma - 1.0
        A /= 2.0

        # Exponent
        exp = 2.0 * gamma / (gamma - 1)

        pressure = (1.0 - A * u) ** (exp)

        return pressure

    def compute_mass_conservation(self, figure=False):

        gamma = self.mu[OneDimensionalBurgersConventions.GAMMA]

        I = []
        outflow = []
        dx = fenics.dx
        for t, u in self.solutions.items():

            # -----------------------------------------------------------------
            # Integral computation
            self.move_mesh(mu=self.mu, t=t)
            rho = self.compute_rho(u, gamma)
            integrand = rho * dx
            integral = fenics.assemble(integrand)
            self.move_mesh(back=True)

            I.append(integral)

            # -----------------------------------------------------------------
            # Outflow computation
            rho0 = self.compute_rho(u(0), gamma=gamma)
            out = rho0 * u(0)
            outflow.append(out)

        I = np.array(I)
        outflow = np.array(outflow)

        # ---------------------------------------------------------------------
        # Compute integral time derivative
        dt = self.dt
        Iprime = np.gradient(I, dt, edge_order=2)

        a0 = self.mu[OneDimensionalBurgersConventions.A0]
        outflow *= a0

        mc = Iprime - outflow

        self.mc = mc
        self.outflow = outflow

        if figure:
            ts = list(self.solutions.keys())
            plot(
                x=ts,
                A=Iprime,
                show=False,
                plot_kwargs=dict(label="$\\frac{d}{dt} \\int \\rho dx$"),
            )
            plot(
                x=ts,
                A=outflow,
                show=False,
                plot_kwargs=dict(label="Outflow $(\\rho(0,t)u(0,t))$"),
            )
            plot(
                x=ts,
                A=mc,
                show=False,
                plot_kwargs=dict(
                    linestyle="--",
                    label="Mass Conservation",
                ),
            )
            plt.legend()
            plt.xlabel("t (s)")
            plt.title(f"dt = {dt}, nx = {self.domain[self.NX]}")
            plt.show()

        return (mc, outflow, I, Iprime)
