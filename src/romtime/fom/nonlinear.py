from functools import partial

import fenics
import numpy as np
from romtime.conventions import FIG_KWARGS
from romtime.utils import array_to_function, function_to_array, plot, plt

from .base import OneDimensionalSolver, move_mesh


class OneDimensionalBurgersConventions:

    A0 = "a0"
    DELTA = "delta"
    GAMMA = "gamma"
    ALPHA = "alpha"


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
        probe_locations=[0.0, 0.5],
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

        self.probe_location = probe_locations
        self.probes = None

        #  Derived results
        self.mc = None  # Mass conservation in time
        self.outflow = None

    @property
    def scale_solutions(self):
        return self.mu[OneDimensionalBurgersConventions.A0]

    @staticmethod
    def nonlinear_coefficient(mu):
        a0 = mu[OneDimensionalBurgersConventions.A0]
        gamma = mu[OneDimensionalBurgersConventions.GAMMA]
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

    def setup(self):
        super().setup()

        self.probe_location = [0.0, 0.5]

        num_probs = len(self.probe_location) + 1
        self.probes = dict()
        for idx in range(num_probs):
            self.probes[idx] = list()

    def runtime_process(self, u):

        num_probs = len(self.probe_location)
        for idx in range(num_probs):
            loc = self.probe_location[idx]
            self.probes[idx].append(u(loc))

        # Probe at the piston movement
        idx_L = idx + 1
        loc = self.L - 10.0 * fenics.DOLFIN_EPS
        self.probes[idx_L].append(u(loc))

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

    def assemble_system(self, mu, t, u_n=None, bdf=1.0):

        Mh_mat = self.assemble_mass(mu=mu, t=t)
        Ah_mat = self.assemble_stiffness(mu=mu, t=t)
        Ch_mat = self.assemble_nonlinear(mu=mu, t=t, u_n=u_n)
        Chat_mat = self.assemble_nonlinear_lifting(mu=mu, t=t)
        Bh_mat = self.assemble_convection(mu=mu, t=t)

        dt = self.dt

        Kh_mat = bdf * Mh_mat + dt * (Ah_mat + Bh_mat + Ch_mat + Chat_mat)

        return Mh_mat, Kh_mat

    def assemble_system_rhs(self, mu, t, Mh_mat, u_n, u_n1=None):

        #  fh_vec = self.assemble_forcing(mu=mu, t=t)
        fgh_vec = self.assemble_lifting(mu=mu, t=t)

        # BDF-1
        if u_n1 is None:
            bdf = Mh_mat * u_n.vector()
        # BDF-2
        else:
            u_sum = fenics.Function(self.V)
            u_sum.assign(2.0 * u_n - 0.5 * u_n1)
            bdf = Mh_mat * u_sum.vector()

        dt = self.dt
        bh_vec = bdf + dt * fgh_vec

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

        if isinstance(u_n, (np.ndarray)):
            u_n = array_to_function(u_n, self.V)

        b0 = self.nonlinear_coefficient(mu)
        Ch = b0 * u_n * u.dx(0) * v * dx

        if entries:
            Ch_mat = self.assemble_local(form=Ch, entries=entries)
        else:
            bc = self.define_homogeneous_dirichlet_bc()
            Ch_mat = self.assemble_operator(Ch, bc)

        return Ch_mat

    @move_mesh
    def assemble_nonlinear_lifting(self, mu, t, entries=None):

        # ---------------------------------------------------------------------
        # Weak Formulation
        u, v, dx = self.u, self.v, fenics.dx

        L = self.L
        g, _, grad_g = self.create_lifting_operator(mu=mu, t=t, L=L)

        b0 = self.nonlinear_coefficient(mu)
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
        b0 = self.nonlinear_coefficient(mu)
        nonlinear = b0 * g * grad_g * v * dx

        # ---------------------------------------------------------------------
        # Convection
        w = self.compute_mesh_velocity(mu=mu, t=t)
        a0 = mu[OneDimensionalBurgersConventions.A0]
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

    def assemble_rhs(self, mu, t, entries=None):
        """Assemble algebraic problem RHS.

        Parameters
        ----------
        mu : dict
        t : float
        entries : list of tuples, optional
            Local entries to assemble when using DEIM techniques, by default None

        Returns
        -------
        """

        fgh = self.assemble_lifting(mu=mu, t=t, entries=entries)

        return fgh

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

    def compute_mass_conservation(
        self, mu, ts, solutions, figure=False, save=None, show=True, title=None
    ):

        gamma = mu[OneDimensionalBurgersConventions.GAMMA]

        # This is to prevent many ifs during time loop
        if isinstance(solutions[0], np.ndarray):
            to_coeff = partial(array_to_function, V=self.V)
        else:
            to_coeff = lambda x: x

        mass = []
        outflow = []
        dx = fenics.dx
        for t, u in zip(ts, solutions):

            # -----------------------------------------------------------------
            # Integral computation
            self.move_mesh(mu=mu, t=t)
            u = to_coeff(u)
            rho = self.compute_rho(u, gamma)
            integrand = rho * dx
            integral = fenics.assemble(integrand)
            self.move_mesh(back=True)

            mass.append(integral)

            # -----------------------------------------------------------------
            # Outflow computation
            rho0 = self.compute_rho(u(0), gamma=gamma)
            out = rho0 * u(0)
            outflow.append(out)

        # Convert to numpy arrays
        mass = np.array(mass)
        outflow = np.array(outflow)

        # ---------------------------------------------------------------------
        # Compute integral time derivative (2nd order)
        dt = self.dt
        mass_change = np.gradient(mass, dt, edge_order=2)

        # ---------------------------------------------------------------------
        # Compute piston outflow
        a0 = mu[OneDimensionalBurgersConventions.A0]
        outflow *= a0

        # ---------------------------------------------------------------------
        # Compute mass conservation equation
        mc = mass_change - outflow

        self.mc = mc
        self.outflow = outflow

        if figure:

            plt.figure()

            plot(
                x=ts,
                A=mass_change,
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
            if title is None:
                title = f"dt = {dt}, nx = {self.domain[self.NX]}"
            plt.title(title)

            if show:
                plt.show()
            if save:
                plt.savefig(save + ".png", **FIG_KWARGS)
                plt.close()

        return mc, outflow, mass, mass_change

    def plot_probes(self, show=True, save=None):

        ts = self.timesteps[1:]

        probes = self.probes
        locations = self.probe_location
        locations.append("L")

        fig, axes = plt.subplots(
            nrows=len(locations),
            ncols=1,
            sharex=True,
            sharey=True,
            gridspec_kw={"hspace": 0.20},
        )

        axes = axes.flatten()

        for idx_probe in probes.keys():
            values = np.array(probes[idx_probe])
            values *= self.scale_solutions
            loc = locations[idx_probe]
            label = f"x={loc}"

            ax = axes[idx_probe]
            ax.plot(ts, values)
            ax.grid(True)
            # ax.set_title(label)
            ax.set_ylabel(f"$u({loc}, t)$")

        plt.xlabel("t (s)")

        if show:
            plt.show()
        if save:
            plt.savefig(save + ".png", **FIG_KWARGS)
            plt.close()
