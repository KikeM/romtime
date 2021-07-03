import fenics
import numpy as np
from romtime.fom import OneDimensionalSolver, move_mesh


class MockSolver(OneDimensionalSolver):
    def __init__(
        self,
        domain,
        dirichlet,
        forcing_term,
        Lt=None,
        degrees=1,
    ) -> None:
        super().__init__(
            domain=domain,
            dirichlet=dirichlet,
            poly_type="P",
            degrees=degrees,
            forcing_term=forcing_term,
            Lt=Lt,
        )

    def assemble_system(self, mu, t):
        pass

    def assemble_system_rhs(self, mu, t):
        pass

    def create_diffusion_coefficient(self, mu, t):
        """Create non-linear diffusion term.

        \\alpha(x) = \\alpha_0 (1 + \\varepsilon x^2)

        Returns
        -------
        alpha : fenics.Expression
        """

        alpha_0 = mu["alpha_0"]

        alpha = fenics.Expression(
            "alpha_0 * (1.0 + t*t)",
            degree=2,
            alpha_0=alpha_0,
            t=t,
        )

        return alpha

    def assemble_stiffness(self, mu, t, entries=None):

        # ---------------------------------------------------------------------
        # Weak Formulation
        # ---------------------------------------------------------------------
        dot, dx, grad = fenics.dot, fenics.dx, fenics.grad
        u, v = self.u, self.v

        alpha = self.create_diffusion_coefficient(mu=mu, t=t)
        Ah = alpha * dot(grad(u), grad(v)) * dx

        if entries:
            Ah_mat = self.assemble_local(Ah, entries=entries)
        else:
            bc = self.define_homogeneous_dirichlet_bc()
            Ah_mat = self.assemble_operator(Ah, bc)

        return Ah_mat

    def assemble_forcing(self, mu, t, entries=None):
        """Assemble test forcing term

        Parameters
        ----------
        mu : dict
        t : float

        Returns
        -------
        dolfin.cpp.la.Vector
        """
        # Extract names to have a clean implementation
        dx = fenics.dx
        v = self.v
        forcing_term = self.forcing_term
        f = fenics.Expression(forcing_term, degree=2, t=t, **mu)

        # Weak form
        fh = f * v * dx

        #  Select between local assembly or global one
        if entries:
            fh_vec = self.assemble_local(form=fh, entries=entries)
        else:
            bc = self.define_homogeneous_dirichlet_bc()
            fh_vec = self.assemble_operator(weak=fh, bcs=bc)

        return fh_vec

    def assemble_lifting(self, mu, t):
        pass


class MockSolverMoving(MockSolver):
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

    @move_mesh
    def assemble_stiffness(self, mu, t, entries=None):
        return super().assemble_stiffness(mu, t, entries=entries)
