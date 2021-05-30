import fenics

from .base import OneDimensionalSolver


class HeatEquationSolver(OneDimensionalSolver):
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

        super().__init__(
            domain,
            dirichlet,
            parameters,
            forcing_term,
            u0,
            filename,
            poly_type,
            degrees,
            project_u0,
            exact_solution,
        )

        # FEM structures
        self.alpha = None  # Nonlinear diffusion coefficient

    def create_diffusion_coefficient(self, mu):
        """Create non-linear diffusion term.

        \\alpha(x) = \\alpha_0 (1 + \\varepsilon x^2)

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

        L = self.domain["L"]
        _, dg_dt, grad_g = self.create_lifting_operator(mu=mu, t=t, L=L)

        # Small hack to handle fenics internals
        grad_g_vec = fenics.as_vector((grad_g,))
        alpha = self.create_diffusion_coefficient(mu)

        # ---------------------------------------------------------------------
        # Weak Formulation
        # ---------------------------------------------------------------------
        dot, dx, grad = fenics.dot, fenics.dx, fenics.grad
        v = self.v

        fgh = -(dg_dt * v + alpha * dot(grad_g_vec, grad(v))) * dx

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

        fh = self.assemble_forcing(mu=mu, t=t, entries=entries)
        fgh = self.assemble_lifting(mu=mu, t=t, entries=entries)

        return fh + fgh
