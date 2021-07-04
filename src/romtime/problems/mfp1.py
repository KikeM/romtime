from collections import defaultdict

import fenics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from romtime.conventions import FIG_KWARGS, OperatorType, RomParameters, Stage
from romtime.deim import (
    DiscreteEmpiricalInterpolation,
    MatrixDiscreteEmpiricalInterpolation,
    mdeim,
)
from romtime.fom import HeatEquationMovingSolver, HeatEquationSolver
from romtime.rom import RomConstructor, RomConstructorMoving
from tqdm import tqdm


def define_mfp1_problem(L=None, nx=None, tf=None, nt=None):

    domain = {
        HeatEquationSolver.L0: L,
        HeatEquationSolver.T: tf,
        HeatEquationSolver.NX: nx,
        HeatEquationSolver.NT: nt,
    }

    #  Boundary conditions
    b0 = "(1.0 - exp(- beta * t))"
    bL = "(1.0 - exp(- beta * t)) * (1.0 + delta*delta * L * L)"

    # TODO: This could be computed with sympy
    # but I don't need this overhead for the moment
    db0_dt = "beta * exp(- beta * t)"
    dbL_dt = "beta * exp(- beta * t) * (1.0 + delta * delta * L * L) + (2.0 * (1.0 - exp(- beta * t)) * (delta * delta) * L) * dLt_dt"

    boundary_conditions = {"b0": b0, "bL": bL, "db0_dt": db0_dt, "dbL_dt": dbL_dt}

    # Forcing term
    forcing_term = """beta * exp(- beta * t) * (1.0 + delta * delta * x[0] * x[0]) - 2.0 * delta * delta * alpha_0 * (1.0 - exp(- beta * t))"""

    # Initial condition
    u0 = fenics.Constant(0.0)

    # Exact solution
    ue = "(1.0 - exp(-beta * t)) * (1.0 + delta * delta * x[0] * x[0])"

    def Lt(omega, t, **kwargs):
        """Mesh scaling function.

        Parameters
        ----------
        omega : float
            Boundary frequency
        t : float

        Returns
        -------
        float
        """
        return 1.0 - np.sin(omega * t)

    def dLt_dt(omega, t, **kwargs):
        """Mesh scaling function time derivative.

        Parameters
        ----------
        omega : float
            Boundary frequency
        t : float

        Returns
        -------
        float
        """
        return -omega * np.cos(omega * t)

    return domain, boundary_conditions, forcing_term, u0, ue, Lt, dLt_dt


class HyperReducedOrderModelFixed:

    FIG_KWARGS = FIG_KWARGS

    def __init__(
        self,
        grid: dict,
        fom_params: dict,
        rom_params: dict,
        deim_params: dict,
        mdeim_params: dict,
        models: dict,
        rnd=None,
    ) -> None:

        self.grid = grid
        self.fom_params = fom_params
        self.rom_params = rom_params
        self.deim_params = deim_params
        self.mdeim_params = mdeim_params
        self.models = models
        self.online_params = None
        self.rnd = rnd

        self.fom = None
        self.rom = None
        self.deim_rhs = None
        self.mdeim_mass = None
        self.mdeim_stiffness = None
        self.mdeim_convection = None

        self.deim_runned = False
        self.rom_runned = False

        self.basis = None
        self.errors = None
        self.summary_basis = defaultdict(dict)
        self.summary_errors = defaultdict(dict)
        self.summary_sigmas = defaultdict(dict)
        self.summary_energy = defaultdict(dict)

    def __del__(self):

        del self.grid
        del self.fom_params
        del self.rom_params
        del self.deim_params
        del self.mdeim_params
        del self.online_params
        del self.rnd

        del self.fom
        del self.rom
        del self.deim_rhs
        del self.mdeim_mass
        del self.mdeim_stiffness
        del self.mdeim_convection

        del self.deim_runned
        del self.rom_runned

        del self.basis
        del self.errors
        del self.summary_basis
        del self.summary_errors
        del self.summary_sigmas
        del self.summary_energy

    def setup(self):
        """Setup up FOM and ROM structures."""

        # ---------------------------------------------------------------------
        # Full Order Model
        # ---------------------------------------------------------------------
        fom_params = self.fom_params
        fom = HeatEquationSolver(
            domain=fom_params.get("domain"),
            dirichlet=fom_params.get("dirichlet"),
            parameters=fom_params.get("parameters", None),
            forcing_term=fom_params.get("forcing_term"),
            u0=fom_params.get("u0"),
            exact_solution=fom_params.get("exact_solution", None),
        )

        fom.setup()

        rnd = self.rnd
        # ---------------------------------------------------------------------
        # Reduced Order Model
        # ---------------------------------------------------------------------
        rom = RomConstructor(fom=fom, grid=self.grid)
        rom.setup(rnd=rnd)

        self.rom = rom
        self.fom = fom

    def setup_hyperreduction(self):
        """Prepare objects to reduce algebraic operators."""

        fom = self.fom
        grid = self.grid
        rnd = self.rnd

        # ---------------------------------------------------------------------
        # DEIM
        # ---------------------------------------------------------------------
        deim_rhs = DiscreteEmpiricalInterpolation(
            name="RHS",
            assemble=fom.assemble_rhs,
            grid=grid,
            tree_walk_params=self.deim_params,
        )

        # ---------------------------------------------------------------------
        # MDEIM
        # ---------------------------------------------------------------------
        mdeim_mass = MatrixDiscreteEmpiricalInterpolation(
            name="Mass",
            assemble=fom.assemble_mass,
            grid=grid,
            tree_walk_params=self.mdeim_params,
        )

        mdeim_stiffness = MatrixDiscreteEmpiricalInterpolation(
            name="Stiffness",
            assemble=fom.assemble_stiffness,
            grid=grid,
            tree_walk_params=self.mdeim_params,
        )

        deim_rhs.setup(rnd=rnd)
        mdeim_mass.setup(rnd=rnd)
        mdeim_stiffness.setup(rnd=rnd)

        self.deim_rhs = deim_rhs
        self.mdeim_mass = mdeim_mass
        self.mdeim_stiffness = mdeim_stiffness

    def run_offline_rom(self, mu_space=None):
        """Generate Reduced Basis and project collateral basis (if any)."""

        rom = self.rom
        num_snapshots = self.rom_params[RomParameters.NUM_SNAPSHOTS]
        num_basis = self.rom_params.get(RomParameters.NUM_MU, None)

        TOL_TIME = RomParameters.TOL_TIME
        TOL_MU = RomParameters.TOL_MU

        tolerances = {
            TOL_TIME: self.rom_params.get(TOL_TIME, None),
            TOL_MU: self.rom_params.get(TOL_MU, None),
        }

        rom.build_reduced_basis(
            num_snapshots=num_snapshots,
            mu_space=mu_space,
            num_basis=num_basis,
            tolerances=tolerances,
        )

        self.basis = rom.basis

        # Project the operators
        if self.deim_runned:
            rom.project_reductors()

    def run_offline_hyperreduction(self, mu_space=None, evaluate=True):
        """Generate collateral basis for algebraic operators."""

        mdeim_stiffness = self.mdeim_stiffness
        mdeim_mass = self.mdeim_mass
        deim_rhs = self.deim_rhs

        if self.models[OperatorType.STIFFNESS]:
            self._run_deim(
                object=mdeim_stiffness,
                which=OperatorType.STIFFNESS,
                evaluate=evaluate,
                mu_space=mu_space,
            )
        if self.models[OperatorType.MASS]:
            self._run_deim(
                object=mdeim_mass,
                which=OperatorType.MASS,
                evaluate=evaluate,
                mu_space=mu_space,
            )
        if self.models[OperatorType.RHS]:
            self._run_deim(
                object=deim_rhs,
                which=OperatorType.RHS,
                evaluate=evaluate,
                mu_space=mu_space,
            )

        self.deim_runned = True

    def solve(self, mu, step):
        """Solve ROM for a given parameter.

        Parameters
        ----------
        mu : dict
        step : str
            Stage.OFFLINE
            Stage.ONLINE
            Stage.VALIDATION
        """
        self.rom.solve(mu, step)

    def evaluate_online(self, params=None, mu_space=None):
        """Online evaluation for a random set of parameters.

        Parameters
        ----------
        params : dict
            Online evaluation parametrization.
        """

        self.online_params = params

        rom = self.rom

        if params:
            rnd2 = params.get("rnd", None)
            num = params["num"]
            space = rom.build_sampling_space(num=num, rnd=rnd2)
        elif mu_space:
            space = mu_space

        for mu in tqdm(space, desc="(HROM) Online evaluation", leave=True):
            rom.solve(mu=mu, step=Stage.ONLINE)

        self.errors = rom.errors_rom

    def generate_summary(self):
        """Generate reduction summaries."""

        summary_basis = self.summary_basis
        summary_sig = self.summary_sigmas
        summary_energy = self.summary_energy

        rom = self.rom
        BASIS_WALK = rom.BASIS_AFTER_WALK
        BASIS_FINAL = rom.BASIS_FINAL
        SPECTRUM_MU = rom.SPECTRUM_MU
        ENERGY_MU = rom.ENERGY_MU
        OFFLINE = Stage.OFFLINE

        off_report = rom.report[OFFLINE]
        summary_basis["reduced-basis"][BASIS_WALK] = off_report[BASIS_WALK]
        summary_basis["reduced-basis"][BASIS_FINAL] = off_report[BASIS_FINAL]
        summary_sig["reduced-basis"][SPECTRUM_MU] = off_report[SPECTRUM_MU]
        summary_energy["reduced-basis"][ENERGY_MU] = off_report[ENERGY_MU]

        if self.mdeim_mass:
            off_report = self.mdeim_mass.report[OFFLINE]
            summary_basis["mdeim-mass"][BASIS_WALK] = off_report[BASIS_WALK]
            summary_basis["mdeim-mass"][BASIS_FINAL] = off_report[BASIS_FINAL]
            summary_sig["mdeim-mass"][SPECTRUM_MU] = off_report[SPECTRUM_MU]
            summary_energy["mdeim-mass"][ENERGY_MU] = off_report[ENERGY_MU]
        if self.mdeim_stiffness:
            off_report = self.mdeim_stiffness.report[OFFLINE]
            summary_basis["mdeim-stiffness"][BASIS_WALK] = off_report[BASIS_WALK]
            summary_basis["mdeim-stiffness"][BASIS_FINAL] = off_report[BASIS_FINAL]
            summary_sig["mdeim-stiffness"][SPECTRUM_MU] = off_report[SPECTRUM_MU]
            summary_energy["mdeim-stiffness"][ENERGY_MU] = off_report[ENERGY_MU]
        if self.mdeim_convection:
            off_report = self.mdeim_convection.report[OFFLINE]
            summary_basis["mdeim-convection"][BASIS_WALK] = off_report[BASIS_WALK]
            summary_basis["mdeim-convection"][BASIS_FINAL] = off_report[BASIS_FINAL]
            summary_sig["mdeim-convection"][SPECTRUM_MU] = off_report[SPECTRUM_MU]
            summary_energy["mdeim-convection"][ENERGY_MU] = off_report[ENERGY_MU]
        if self.deim_rhs:
            off_report = self.deim_rhs.report[OFFLINE]
            summary_basis["deim-rhs"][BASIS_WALK] = off_report[BASIS_WALK]
            summary_basis["deim-rhs"][BASIS_FINAL] = off_report[BASIS_FINAL]
            summary_sig["deim-rhs"][SPECTRUM_MU] = off_report[SPECTRUM_MU]
            summary_energy["deim-rhs"][ENERGY_MU] = off_report[ENERGY_MU]

        self.summary_basis = pd.DataFrame(summary_basis).T

        # Integration errors
        summary_errors = defaultdict(dict)
        for idx, error in rom.errors.items():
            summary_errors[idx]["mean"] = np.mean(error)
            summary_errors[idx]["median"] = np.median(error)
            summary_errors[idx]["max"] = np.max(error)
            summary_errors[idx]["min"] = np.min(error)

        self.summary_errors = pd.DataFrame(summary_errors).T

    def plot_spectrums(self, save=None, new=True):

        if new:
            plt.figure()

        SPECTRUM_MU = self.rom.SPECTRUM_MU

        summary_sigmas = self.summary_sigmas

        for element, sigmas_dict in summary_sigmas.items():
            sigma = sigmas_dict[SPECTRUM_MU]
            sigma = np.log10(sigma)
            plt.plot(sigma, label=element)

        plt.xlabel("Up to n-th basis element")
        plt.ylabel("$\\log (\\sigma)$")
        plt.title("Spectrum decay in the parameter space")
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(save + ".png", **self.FIG_KWARGS)

        plt.show()

    def plot_energy(self, save=None):

        ENERGY_MU = self.rom.ENERGY_MU

        summary_energy = self.summary_energy

        for element, energy in summary_energy.items():
            sigma = energy[ENERGY_MU]
            plt.plot(sigma, label=element)

        plt.xlabel("Up to n-th basis element")
        plt.ylabel("Ratio")
        plt.title("Total POD Energy Ratio")
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(save + ".png", **self.FIG_KWARGS)

        plt.show()

    def plot_errors(self, save=None, new=True, label=None, show=True):

        if new:
            plt.figure()

        rom = self.rom

        for idx, error in rom.errors.items():
            error = np.log10(error)
            plt.plot(rom.timesteps[1:], error, linewidth=1.0, alpha=0.85, label=label)

        plt.xlabel("t")
        plt.ylabel("log10 Error (L2)")
        plt.title("Online Errors")
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(save + ".png", **self.FIG_KWARGS)

        if show:
            plt.show()

    def _run_deim(
        self,
        object: DiscreteEmpiricalInterpolation,
        which: str,
        mu_space: list,
        evaluate=False,
    ):

        # Build collateral basis
        object.run(mu_space=mu_space)

        # Online evaluation
        if evaluate:
            n_online = self.deim_params.get(RomParameters.NUM_ONLINE, None)
            timesteps = self.deim_params["ts"]

            object.evaluate(ts=timesteps, num=n_online, mu_space=mu_space)

        # Include the reduction for the algebraic operators
        rom = self.rom
        rom.add_hyper_reductor(reductor=object, which=which)


class HyperReducedOrderModelMoving(HyperReducedOrderModelFixed):
    def __init__(
        self,
        grid: dict,
        fom_params: dict,
        rom_params: dict,
        deim_params: dict,
        mdeim_params: dict,
        models: dict,
        rnd,
    ) -> None:

        super().__init__(
            grid=grid,
            fom_params=fom_params,
            rom_params=rom_params,
            deim_params=deim_params,
            mdeim_params=mdeim_params,
            models=models,
            rnd=rnd,
        )

        self.mdeim_convection = None
        self.deim_moving_runned = False

    def setup(self):
        """Setup up FOM and ROM structures."""

        # ---------------------------------------------------------------------
        # Full Order Model
        # ---------------------------------------------------------------------
        fom_params = self.fom_params
        fom = HeatEquationMovingSolver(
            domain=fom_params.get("domain"),
            dirichlet=fom_params.get("dirichlet"),
            parameters=fom_params.get("parameters", None),
            forcing_term=fom_params.get("forcing_term"),
            u0=fom_params.get("u0"),
            exact_solution=fom_params.get("exact_solution", None),
            Lt=fom_params.get("Lt"),
            dLt_dt=fom_params.get("dLt_dt"),
        )

        fom.setup()

        rnd = self.rnd
        # ---------------------------------------------------------------------
        # Reduced Order Model
        # ---------------------------------------------------------------------
        rom = RomConstructorMoving(fom=fom, grid=self.grid)
        rom.setup(rnd=rnd)

        self.rom = rom
        self.fom = fom

    def setup_hyperreduction(self):
        """Prepare objects to reduce algebraic operators."""

        super().setup_hyperreduction()

        fom = self.fom
        grid = self.grid
        rnd = self.rnd

        # ---------------------------------------------------------------------
        # MDEIM
        # ---------------------------------------------------------------------
        mdeim_convection = MatrixDiscreteEmpiricalInterpolation(
            name=OperatorType.CONVECTION,
            assemble=fom.assemble_convection,
            grid=grid,
            tree_walk_params=self.mdeim_params,
        )

        mdeim_convection.setup(rnd=rnd)

        self.mdeim_convection = mdeim_convection

    def run_offline_hyperreduction(self, mu_space=None, evaluate=True):
        """Generate collateral basis for algebraic operators."""

        super().run_offline_hyperreduction(mu_space=mu_space, evaluate=evaluate)

        mdeim_convection = self.mdeim_convection

        if self.models[OperatorType.CONVECTION]:
            self._run_deim(
                object=mdeim_convection,
                mu_space=mu_space,
                evaluate=evaluate,
                which=OperatorType.CONVECTION,
            )

        self.deim_moving_runned = True
