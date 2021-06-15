import fenics
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from romtime.conventions import Domain, RomParameters
from romtime.deim import MatrixDiscreteEmpiricalInterpolation
from romtime.fom import HeatEquationSolver, move_mesh
from romtime.parameters import get_uniform_dist
from romtime.problems.mfp1 import define_mfp1_problem
from romtime.rom.base import Reductor

DIFFUSION = "diffusion"
CONVECTION = "convection"
BOTH = "both"


class MockSolverALE(HeatEquationSolver):
    def __init__(
        self,
        domain: dict,
        dirichlet: dict,
        parameters: dict,
        forcing_term: str,
        u0,
        filename=None,
        poly_type="P",
        degrees=1,
        project_u0=False,
        exact_solution=None,
        Lt=None,
        dLt_dt=None,
        which=None,
    ) -> None:

        self.WHICH = which

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

    def compute_mesh_velocity(self, mu, t):

        dLt_dt = self.dLt_dt(t=t, **mu)
        Lt = self.Lt(t=t, **mu)

        w = fenics.Expression("x[0] * dLt_dt / Lt", degree=1, dLt_dt=dLt_dt, Lt=Lt)

        return w

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

        if self.WHICH == BOTH:
            Ah = -u.dx(0) * v * dx + dot(grad(u), grad(v)) * dx
        elif self.WHICH == DIFFUSION:
            Ah = dot(grad(u), grad(v)) * dx
        elif self.WHICH == CONVECTION:
            Ah = -u.dx(0) * v * dx

        bc = self.define_homogeneous_dirichlet_bc()
        Ah_mat = self.assemble_operator(Ah, bc)

        return Ah_mat

    @move_mesh
    def assemble_stiffness(self, mu, t, entries=None):
        """Assemble stiffness matrix for a ALE problem.

        Parameters
        ----------
        mu : dict
        t : float
        entries : [type], optional
            [description], by default None
        """

        # ---------------------------------------------------------------------
        # Weak Formulation
        # ---------------------------------------------------------------------
        dot, dx, grad = fenics.dot, fenics.dx, fenics.grad
        u, v = self.u, self.v

        w = self.compute_mesh_velocity(mu=mu, t=t)
        alpha = self.create_diffusion_coefficient(mu)

        if self.WHICH == BOTH:
            Ah = -w * u.dx(0) * v * dx + alpha * dot(grad(u), grad(v)) * dx
        elif self.WHICH == DIFFUSION:
            Ah = alpha * dot(grad(u), grad(v)) * dx
        elif self.WHICH == CONVECTION:
            Ah = -w * u.dx(0) * v * dx

        if entries:
            Ah_mat = self.assemble_local(form=Ah, entries=entries)
        else:
            bc = self.define_homogeneous_dirichlet_bc()
            Ah_mat = self.assemble_operator(Ah, bc)

        return Ah_mat


@pytest.mark.parametrize("which", [CONVECTION, DIFFUSION])
def test_mdeim_ale_stiffness(which):

    domain = dict(
        L0=2.0,
        nx=500,
        nt=250,
        T=10.0,
    )

    # Compute omega variables
    n_min = 0.5
    n_max = 0.8

    tf = domain["T"]
    omegas = [(1.0 / tf) * np.arcsin(1.0 - n) for n in (n_min, n_max)]
    omega_max = max(omegas)
    omega_min = min(omegas)

    grid = {
        "delta": get_uniform_dist(min=0.01, max=5.0),
        "beta": get_uniform_dist(min=0.05, max=0.1),
        "alpha_0": get_uniform_dist(min=0.01, max=2.0),
        "omega": get_uniform_dist(min=omega_min, max=omega_max),
    }

    _, boundary_conditions, forcing_term, u0, ue, Lt, dLt_dt = define_mfp1_problem()

    fom = MockSolverALE(
        domain=domain,
        dirichlet=boundary_conditions,
        parameters=None,
        forcing_term=forcing_term,
        u0=u0,
        Lt=Lt,
        dLt_dt=dLt_dt,
        which=which,
    )

    fom.setup()

    # (M)DEIM parametrization
    tf, nt = domain[Domain.T], domain[Domain.NT]
    ts = np.linspace(tf / nt, tf, nt)

    mdeim_params = {
        RomParameters.TS: ts,
        RomParameters.NUM_SNAPSHOTS: None,
        RomParameters.NUM_MU: 2,
        RomParameters.NUM_TIME: 2,
    }

    mdeim = MatrixDiscreteEmpiricalInterpolation(
        assemble=fom.assemble_stiffness,
        name="Stiffness (ALE)",
        grid=grid,
        tree_walk_params=mdeim_params,
    )

    reductor = Reductor(grid=grid)
    rnd = np.random.RandomState(0)
    mu_space = reductor.build_sampling_space(num=10, rnd=rnd)
    mu_space = list(mu_space)

    mdeim.setup(rnd=rnd)
    mdeim.run(mu_space=mu_space)
    mdeim.evaluate(ts, mu_space=mu_space)

    mdeim.create_errors_summary()

    result = mdeim.summary_errors

    if which == CONVECTION:
        expected = pd.DataFrame(
            {
                "mean": {
                    0: 4.067919714383879e-18,
                    1: 2.860381570616804e-18,
                    2: 4.146111997501708e-18,
                    3: 6.787873897293087e-18,
                    4: 3.661210824955406e-18,
                    5: 4.3341609229696664e-18,
                    6: 3.7678748910599596e-18,
                    7: 4.0432954342970854e-18,
                    8: 4.919830984677875e-18,
                    9: 4.86332991015263e-18,
                },
                "median": {
                    0: 3.998663229968674e-18,
                    1: 2.8560924686563044e-18,
                    2: 4.127155925063022e-18,
                    3: 6.732949098849319e-18,
                    4: 3.642897533265148e-18,
                    5: 4.3978651250563926e-18,
                    6: 3.7652066547887535e-18,
                    7: 4.026568298662315e-18,
                    8: 4.964405456696337e-18,
                    9: 4.873637795410932e-18,
                },
                "max": {
                    0: 4.6206780856631024e-18,
                    1: 3.1451780658542135e-18,
                    2: 4.701049016891742e-18,
                    3: 7.270292336682435e-18,
                    4: 4.084072928552312e-18,
                    5: 4.853137732292008e-18,
                    6: 4.021780940979192e-18,
                    7: 4.3491682424354645e-18,
                    8: 5.287391280770844e-18,
                    9: 5.139578479677387e-18,
                },
                "min": {
                    0: 3.638973459527289e-18,
                    1: 2.4693686125467854e-18,
                    2: 3.838059636540121e-18,
                    3: 6.346881101011284e-18,
                    4: 3.359095297119872e-18,
                    5: 3.678183710297739e-18,
                    6: 3.516622144841333e-18,
                    7: 3.663927977476139e-18,
                    8: 4.2964626566855976e-18,
                    9: 4.5259351970025626e-18,
                },
            }
        )
    elif which == DIFFUSION:
        expected = pd.DataFrame(
            {
                "mean": {
                    0: 1.3561722447716433e-11,
                    1: 1.0267883204080527e-11,
                    2: 1.9470524015440092e-11,
                    3: 1.3828949940027551e-11,
                    4: 6.713737351411585e-12,
                    5: 2.7004642780375507e-11,
                    6: 8.096224526953213e-12,
                    7: 1.792994066883958e-11,
                    8: 1.8625156303494276e-11,
                    9: 2.563166727180761e-11,
                },
                "median": {
                    0: 1.3682124220919097e-11,
                    1: 1.0180640020780692e-11,
                    2: 1.857861803720626e-11,
                    3: 1.3902869203312794e-11,
                    4: 6.7207436842651e-12,
                    5: 2.695334977820879e-11,
                    6: 8.00982455421739e-12,
                    7: 1.7337189706868043e-11,
                    8: 1.7958902563123075e-11,
                    9: 2.5294497962516985e-11,
                },
                "max": {
                    0: 1.5383526425229874e-11,
                    1: 1.2159237908824454e-11,
                    2: 2.6768179103471296e-11,
                    3: 1.5144815392212343e-11,
                    4: 6.884859966256192e-12,
                    5: 3.681333458907268e-11,
                    6: 9.013381525251758e-12,
                    7: 2.3781794235466573e-11,
                    8: 2.3974893609399702e-11,
                    9: 3.435289689868974e-11,
                },
                "min": {
                    0: 1.064290945705619e-11,
                    1: 8.60539935703987e-12,
                    2: 1.4325111844070667e-11,
                    3: 1.206381822205726e-11,
                    4: 6.571969212287795e-12,
                    5: 1.955395253393507e-11,
                    6: 7.203924794136626e-12,
                    7: 1.3771528297292846e-11,
                    8: 1.348504755409947e-11,
                    9: 1.95937530960456e-11,
                },
            }
        )

    assert_frame_equal(left=expected, right=result)
