from collections import defaultdict

import fenics
import numpy as np
import pandas as pd
from romtime.deim import (
    DiscreteEmpiricalInterpolation,
    MatrixDiscreteEmpiricalInterpolation,
)
from romtime.fom import HeatEquationSolver
from romtime.rom import RomConstructor

import matplotlib.pyplot as plt


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
    ue = "(1.0 - exp(-beta*t)) * (1.0 + delta*delta * x[0]*x[0])"

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


class HyperReducedOrderModel:

    FIG_KWARGS = dict(dpi=300, bbox_inches="tight")

    def __init__(
        self,
        grid: dict,
        fom_params: dict,
        rom_params: dict,
        deim_params: dict,
        mdeim_params: dict,
        rnd=None,
    ) -> None:

        self.grid = grid
        self.fom_params = fom_params
        self.rom_params = rom_params
        self.deim_params = deim_params
        self.mdeim_params = mdeim_params
        self.online_params = None
        self.rnd = rnd

        self.fom = None
        self.rom = None
        self.deim_rhs = None
        self.mdeim_mass = None
        self.mdeim_stiffness = None

        self.deim_runned = False
        self.rom_runned = False

        self.basis = None
        self.errors = None
        self.summary_basis = defaultdict(dict)
        self.summary_errors = defaultdict(dict)
        self.summary_sigmas = defaultdict(dict)

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
        mdeim_stiffness = MatrixDiscreteEmpiricalInterpolation(
            name="Stiffness",
            assemble=fom.assemble_stiffness,
            grid=grid,
            tree_walk_params=self.mdeim_params,
        )

        mdeim_mass = MatrixDiscreteEmpiricalInterpolation(
            name="Mass",
            assemble=fom.assemble_mass,
            grid=grid,
            tree_walk_params=self.mdeim_params,
        )

        deim_rhs.setup(rnd=rnd)
        mdeim_mass.setup(rnd=rnd)
        mdeim_stiffness.setup(rnd=rnd)

        self.mdeim_mass = mdeim_mass
        self.mdeim_stiffness = mdeim_stiffness
        self.deim_rhs = deim_rhs

    def run_offline_rom(self):
        """Generate Reduced Basis and project collateral basis (if any)."""

        rom = self.rom
        num_snapshots = self.rom_params["num_snapshots"]

        rom.build_reduced_basis(num_snapshots=num_snapshots)

        self.basis = rom.basis

        # Project the operators
        if self.deim_runned:
            rom.project_reductors()

    def run_offline_hyperreduction(self):

        mdeim_stiffness = self.mdeim_stiffness
        deim_rhs = self.deim_rhs
        mdeim_mass = self.mdeim_mass

        mdeim_stiffness.run()
        deim_rhs.run()
        mdeim_mass.run()

        self.deim_runned = True

        # Include the reduction for the algebraic operators
        rom = self.rom
        rom.add_hyper_reductor(reductor=deim_rhs, which=rom.RHS)
        rom.add_hyper_reductor(reductor=mdeim_stiffness, which=rom.STIFFNESS)
        rom.add_hyper_reductor(reductor=mdeim_mass, which=rom.MASS)

    def solve(self, mu, step):
        """Solve ROM for a given parameter.

        Parameters
        ----------
        mu : dict
        step : str
            RomConstructor.OFFLINE
            RomConstructor.ONLINE
            RomConstructor.VALIDATION
        """
        rom = self.rom
        rom.solve(mu, step)

    def evaluate_online(self, params: dict):
        """Online evaluation for a random set of parameters.

        Parameters
        ----------
        params : dict
            Online evaluation parametrization.
        """

        self.online_params = params

        rom = self.rom

        rnd2 = params.get("rnd", None)
        num = params["num"]
        sampler = rom.build_sampling_space(num=num, rnd=rnd2)

        for mu in sampler:
            rom.solve(mu=mu, step=rom.ONLINE)

        self.errors = rom.errors_rom

    def generate_summary(self):

        summary_basis = self.summary_basis
        summary_sig = self.summary_sigmas

        rom = self.rom
        BASIS_WALK = rom.BASIS_AFTER_WALK
        BASIS_FINAL = rom.BASIS_FINAL
        SPECTRUM_MU = rom.SPECTRUM_MU
        OFFLINE = rom.OFFLINE

        off_report = rom.report[OFFLINE]
        summary_basis["reduced-basis"][BASIS_WALK] = off_report[BASIS_WALK]
        summary_basis["reduced-basis"][BASIS_FINAL] = off_report[BASIS_FINAL]
        summary_sig["reduced-basis"][SPECTRUM_MU] = off_report[SPECTRUM_MU]

        if self.mdeim_mass:
            off_report = self.mdeim_mass.report[OFFLINE]
            summary_basis["mdeim-mass"][BASIS_WALK] = off_report[BASIS_WALK]
            summary_basis["mdeim-mass"][BASIS_FINAL] = off_report[BASIS_FINAL]
            summary_sig["mdeim-mass"][SPECTRUM_MU] = off_report[SPECTRUM_MU]
        if self.mdeim_stiffness:
            off_report = self.mdeim_stiffness.report[OFFLINE]
            summary_basis["mdeim-stiffness"][BASIS_WALK] = off_report[BASIS_WALK]
            summary_basis["mdeim-stiffness"][BASIS_FINAL] = off_report[BASIS_FINAL]
            summary_sig["mdeim-stiffness"][SPECTRUM_MU] = off_report[SPECTRUM_MU]
        if self.deim_rhs:
            off_report = self.deim_rhs.report[OFFLINE]
            summary_basis["deim-rhs"][BASIS_WALK] = off_report[BASIS_WALK]
            summary_basis["deim-rhs"][BASIS_FINAL] = off_report[BASIS_FINAL]
            summary_sig["deim-rhs"][SPECTRUM_MU] = off_report[SPECTRUM_MU]

        self.summary_basis = pd.DataFrame(summary_basis).T

        # Integration errors
        summary_errors = defaultdict(dict)
        for idx, error in rom.errors.items():
            summary_errors[idx]["mean"] = np.mean(error)
            summary_errors[idx]["median"] = np.median(error)
            summary_errors[idx]["max"] = np.max(error)
            summary_errors[idx]["min"] = np.min(error)

        self.summary_errors = pd.DataFrame(summary_errors).T

    def plot_spectrums(self, save=None):

        SPECTRUM_MU = self.rom.SPECTRUM_MU

        summary_sigmas = self.summary_sigmas

        for element, sigmas_dict in summary_sigmas.items():
            sigma = sigmas_dict[SPECTRUM_MU]
            sigma = np.log10(sigma)
            plt.plot(sigma, label=element)

        plt.xlabel("idx")
        plt.ylabel("$\\log (\\sigma)$")
        plt.title("Spectrum decay in the parameter space")
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(save + ".png", **self.FIG_KWARGS)

        plt.show()

    def plot_errors(self, save=None):

        rom = self.rom

        for idx, error in rom.errors.items():
            error = np.log10(error)
            plt.plot(rom.timesteps[1:], error, c="k", linewidth=1.0, alpha=0.85)

        plt.xlabel("t")
        plt.ylabel("log Error (L2)")
        plt.title("Online Phase Accuracy")
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(save + ".png", **self.FIG_KWARGS)
        plt.show()
