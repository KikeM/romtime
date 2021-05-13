import fenics
from fenics import dot, grad, dx
from functools import partial


import numpy as np


class HeatEquationSolver:
    def __init__(
        self,
        nx,
        ny,
        f,
        u_D,
        T,
        num_steps,
        poly_type="P",
        degrees=1,
        project_u0=False,
    ) -> None:

        self.nx = nx
        self.ny = ny
        self.f = f
        self.u_D = u_D

        self.T = T
        self.num_steps = num_steps

        self.poly_type = poly_type
        self.degrees = degrees

        self.project_u0 = project_u0

        self.a = None
        self.F = None

    def setup(self):

        mesh = fenics.UnitSquareMesh(nx=self.nx, ny=self.nx)
        V = fenics.FunctionSpace(mesh, self.poly_type, self.degrees)

        # Galerkin projection
        u = fenics.TrialFunction(V=V)
        v = fenics.TestFunction(V=V)

        self.mesh = mesh
        self.V = V
        self.u = u
        self.v = v

        V = self.V

        self.mesh = mesh
        self.V = V
        self.u = u
        self.v = v

    def solve(self):

        V = self.V

        # Create solution with u = span(V)
        uh = fenics.Function(V)

        # Obtain initial solution in the mesh
        u_D = self.u_D
        if self.project_u0 == True:
            u_n = fenics.project(u_D, V)
        else:
            u_n = fenics.interpolate(u_D, V)

        # Time domain step size
        dt = self.T / self.num_steps

        # Build variational problem in space
        u, v, f = self.u, self.v, self.f
        F = (u * v + dt * dot(grad(u), grad(v)) - (u_n + dt * f) * v) * dx
        a = fenics.lhs(F)
        L = fenics.rhs(F)

        # Create boundary conditions
        # TODO: introduce lifting function
        def boundary(x, on_boundary):
            return on_boundary

        bc = fenics.DirichletBC(V, u_D, boundary)

        # Fix boundary conditions
        solve = partial(fenics.solve, bcs=bc)

        t = 0
        errors = dict()
        for n in range(self.num_steps):

            # Update time
            t += dt
            u_D.t = t

            # Solve for new value
            solve(a == L, uh)

            # Compute error
            ue = fenics.interpolate(u_D, V)
            error = self._compute_error(u=uh, ue=ue)
            errors[t] = error

            # Update solution
            u_n.assign(uh)

        self.errors = errors

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

            error = fenics.errornorm(u=ue, uh=u, norm_type=norm_type)

        return error