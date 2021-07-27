import ujson
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from romtime.conventions import FIG_KWARGS, OperatorType, RomParameters, Stage
from romtime.deim import (
    DiscreteEmpiricalInterpolation,
    MatrixDiscreteEmpiricalInterpolation,
)
from romtime.deim.nonlinear import MatrixDiscreteEmpiricalInterpolationNonlinear
from romtime.fom import HeatEquationMovingSolver, HeatEquationSolver
from romtime.fom.nonlinear import OneDimensionalBurgers
from romtime.rom import RomConstructor, RomConstructorMoving
from romtime.rom.rom import RomConstructorNonlinear
from tqdm import tqdm

import pickle


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

        self.fom = OneDimensionalBurgers
        self.rom = RomConstructorNonlinear
        self.deim_rhs = None
        self.mdeim_mass = None
        self.mdeim_stiffness = None
        self.mdeim_convection = None
        self.mdeim_nonlinear = None
        self.mdeim_nonlinear_lifting = None

        self.deim_runned = False
        self.rom_runned = False

        self.basis = None
        self.errors = dict()
        self.summary_basis = defaultdict(dict)
        self._summary_basis = defaultdict(dict)
        self.summary_errors = defaultdict(dict)
        self.summary_errors_deim = defaultdict(dict)
        self.mu_space_deim = dict()
        self.summary_sigmas = defaultdict(dict)
        self.summary_energy = defaultdict(dict)

        # For validation purposes
        self.validation_solutions = None

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
        del self.mdeim_nonlinear
        del self.mdeim_nonlinear_lifting

        del self.deim_runned
        del self.rom_runned

        del self.basis
        del self.errors
        del self._summary_basis
        del self.summary_basis
        del self.summary_errors
        del self.summary_sigmas
        del self.summary_energy
        del self.summary_errors_deim

        del self.validation_solutions

    @property
    def mu_space(self):
        return self.rom.mu_space

    def dump_mu_space(self, path=None):
        """Write mu space to a json file.

        Parameters
        ----------
        path : str or Path-like, optional
        """
        if path is None:
            path = "mu_space.json"

        with open(path, mode="w") as fp:
            ujson.dump(self.rom.mu_space, fp)

    def dump_mu_space_deim(self, path=None):

        if path is None:
            path = "mu_space_deim.json"

        with open(path, mode="w") as fp:
            ujson.dump(self.mu_space_deim, fp)

    def dump_reduced_basis(self, path=None):

        if path is None:
            path = "rb_basis.pkl"

        with open(path, mode="wb") as fp:
            pickle.dump(self.basis, fp)

    def dump_validation_fom(self, path=None):

        if path is None:
            path = "validation_solutions.pkl"

        with open(path, mode="wb") as fp:
            pickle.dump(self.validation_solutions, fp)

    def dump_errors(self, which, path=None):

        if path is None:
            path = Path(".")

        errors = self.errors

        is_present = which in errors
        if is_present == False:
            Warning("These errors ({which}) have not been computed yet.")
            return None
        else:
            pd.DataFrame(errors[which]).to_csv(path / f"errors_{which}.csv")

    def dump_errors_deim(self, path=None):

        if path is None:
            path = Path(".")

        errors_deim = self.summary_errors_deim

        for operator, errors in errors_deim.items():
            pd.DataFrame(errors).to_csv(path / f"errors_deim_{operator.lower()}.csv")

    def dump_setup(self, path):

        if path is None:
            path = "setup.json"

        out = dict()
        out["fom_params"] = self.fom_params["domain"]
        out["mu_space"] = self.fom_params["grid_params"]
        out["rom_params"] = self.rom_params
        out["deim_params"] = self.deim_params
        out["mdeim_params"] = self.mdeim_params
        out["online_params"] = self.online_params

        with open(path, mode="w") as fp:
            ujson.dump(out, fp)

    def load_validation_fom(self, path=None):

        if path is None:
            path = "validation_solutions.pkl"

        with open(path, mode="rb") as fp:
            self.validation_solutions = pickle.load(fp)

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

        #  ---------------------------------------------------------------------
        #  Build basis
        # fom_solutions = rom.build_reduced_basis(
        #     num_snapshots=num_snapshots,
        #     mu_space=mu_space,
        #     num_basis=num_basis,
        #     tolerances=tolerances,
        # )

        # self.basis = rom.basis
        # self.validation_solutions = fom_solutions

        #  ---------------------------------------------------------------------
        #  Read basis
        print("Loading RB Basis from disk ...")
        with open("rb_basis.pkl", mode="rb") as fp:
            basis = pickle.load(fp)

        self.basis = basis.copy()
        rom.basis = basis.copy()

        with open("mu_space.json", mode="r") as fp:
            mu_space = ujson.load(fp)

        mu_space[Stage.ONLINE] = []
        mu_space[Stage.VALIDATION] = []
        rom.mu_space = mu_space.copy()

        self.load_validation_fom()

        # Project the operators
        if self.deim_runned:
            rom.project_reductors()

    def run_offline_hyperreduction(self, mu_space=None, evaluate=True):
        """Generate collateral basis for algebraic operators."""

        mdeim_stiffness = self.mdeim_stiffness
        mdeim_mass = self.mdeim_mass
        deim_rhs = self.deim_rhs

        if self.models[OperatorType.STIFFNESS]:
            self._run_mdeim(
                object=mdeim_stiffness,
                which=OperatorType.STIFFNESS,
                evaluate=evaluate,
                mu_space=mu_space,
            )
        if self.models[OperatorType.MASS]:
            self._run_mdeim(
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

    def evaluate_validation(self):
        """ROM evaluation for training set of parameters."""

        rom = self.rom
        space = rom.mu_space[Stage.OFFLINE]
        self._evaluate(which=Stage.VALIDATION, mu_space=space)

    def evaluate_online(self, params, rnd=None):
        """Online evaluation for a random set of parameters.

        Parameters
        ----------
        params : dict
            Online evaluation parametrization.
        """

        self.online_params = params
        rom = self.rom
        # ---------------------------------------------------------------------
        # Build random sampling space
        rnd2 = rnd
        num = params["num"]
        space = rom.build_sampling_space(num=num, rnd=rnd2)

        # ---------------------------------------------------------------------
        # Evaluate
        self._evaluate(which=Stage.ONLINE, mu_space=space)

    def _evaluate(self, which, mu_space=None):
        """Evaluation for a set of parameters."""

        fom = self.fom
        rom = self.rom

        if fom.exact_solution is None:
            rom_fom_errors = dict()

        compute_error = rom._compute_error
        desc = f"(HROM) {which.upper()} evaluation"
        for mu in tqdm(mu_space, desc=desc, leave=True):

            # -----------------------------------------------------------------
            # Solve ROM
            idx_mu = rom.solve(mu=mu, step=which)

            # -----------------------------------------------------------------
            # Compare against FOM
            if which == Stage.VALIDATION:
                uh_fom = self.validation_solutions[idx_mu]
            else:
                fom.setup()
                fom.update_parametrization(mu)
                fom.solve()
                uh_fom = fom._solutions

            uh_rom = rom._solution

            nt = uh_fom.shape[1]
            errors = [
                compute_error(uh_fom[:, idx], uh_rom[:, idx]) for idx in range(nt)
            ]
            errors = np.array(errors)
            rom_fom_errors.update({idx_mu: np.array(errors)})

            # -------------------------------------------------------------
            # Compute FOM solution
            if fom.RUNTIME_PROCESS & (which == Stage.ONLINE):
                name_probes = f"probes_{which}_fom_{idx_mu}"
                fom.plot_probes(show=False, save=name_probes)

            # Compute ROM solution
            # TODO: Pending! I have problems with function evaluations after I
            # use interpolate.

            # -------------------------------------------------------------
            # Compute mass conservation
            timesteps = rom.timesteps[1:]

            # ROM
            name_rom = f"mass_conservation_{which}_rom_{idx_mu}"
            rom_sols = uh_rom.T
            fom.compute_mass_conservation(
                mu=mu,
                ts=timesteps,
                solutions=rom_sols,
                figure=True,
                save=name_rom,
                show=False,
                title="ROM Solution",
            )

            # FOM
            name_fom = f"mass_conservation_{which}_fom_{idx_mu}"
            fom_sols = uh_fom.T
            fom.compute_mass_conservation(
                mu=mu,
                ts=timesteps,
                solutions=fom_sols,
                figure=True,
                save=name_fom,
                show=False,
                title="FOM Solution",
            )

        if fom.exact_solution is None:
            self.errors[which] = rom_fom_errors
        else:
            self.errors[which] = rom.errors_rom

    def generate_summary(self):
        """Generate reduction summaries."""

        summary_basis = self._summary_basis
        summary_sig = self.summary_sigmas
        summary_energy = self.summary_energy
        summary_errors_deim = self.summary_errors_deim
        mu_space_deim = self.mu_space_deim

        rom = self.rom
        BASIS_WALK = rom.BASIS_AFTER_WALK
        BASIS_FINAL = rom.BASIS_FINAL
        SPECTRUM_MU = rom.SPECTRUM_MU
        ENERGY_MU = rom.ENERGY_MU
        OFFLINE = Stage.OFFLINE

        report = rom.report[OFFLINE]
        summary_basis["reduced-basis"][BASIS_WALK] = report[BASIS_WALK]
        summary_basis["reduced-basis"][BASIS_FINAL] = report[BASIS_FINAL]
        summary_sig["reduced-basis"][SPECTRUM_MU] = report[SPECTRUM_MU]
        summary_energy["reduced-basis"][ENERGY_MU] = report[ENERGY_MU]

        if self.mdeim_mass:
            report = self.mdeim_mass.report[OFFLINE]
            MASS = OperatorType.MASS
            summary_basis[MASS][BASIS_WALK] = report[BASIS_WALK]
            summary_basis[MASS][BASIS_FINAL] = report[BASIS_FINAL]
            summary_sig[MASS][SPECTRUM_MU] = report[SPECTRUM_MU]
            summary_energy[MASS][ENERGY_MU] = report[ENERGY_MU]
            summary_errors_deim[MASS] = self.mdeim_mass.errors_rom.copy()
            mu_space_deim[MASS] = self.mdeim_mass.mu_space
        if self.mdeim_stiffness:
            report = self.mdeim_stiffness.report[OFFLINE]
            STIFFNESS = OperatorType.STIFFNESS
            summary_basis[STIFFNESS][BASIS_WALK] = report[BASIS_WALK]
            summary_basis[STIFFNESS][BASIS_FINAL] = report[BASIS_FINAL]
            summary_sig[STIFFNESS][SPECTRUM_MU] = report[SPECTRUM_MU]
            summary_energy[STIFFNESS][ENERGY_MU] = report[ENERGY_MU]
            summary_errors_deim[STIFFNESS] = self.mdeim_stiffness.errors_rom.copy()
            mu_space_deim[STIFFNESS] = self.mdeim_stiffness.mu_space
        if self.mdeim_convection:
            report = self.mdeim_convection.report[OFFLINE]
            CONVECTION = OperatorType.CONVECTION
            summary_basis[CONVECTION][BASIS_WALK] = report[BASIS_WALK]
            summary_basis[CONVECTION][BASIS_FINAL] = report[BASIS_FINAL]
            summary_sig[CONVECTION][SPECTRUM_MU] = report[SPECTRUM_MU]
            summary_energy[CONVECTION][ENERGY_MU] = report[ENERGY_MU]
            summary_errors_deim[CONVECTION] = self.mdeim_convection.errors_rom.copy()
            mu_space_deim[CONVECTION] = self.mdeim_convection.mu_space
        if self.mdeim_nonlinear:
            report = self.mdeim_nonlinear.report[OFFLINE]
            NONLINEAR = OperatorType.NONLINEAR
            summary_basis[NONLINEAR][BASIS_WALK] = report[BASIS_WALK]
            summary_basis[NONLINEAR][BASIS_FINAL] = report[BASIS_FINAL]
            summary_sig[NONLINEAR][SPECTRUM_MU] = report[SPECTRUM_MU]
            summary_energy[NONLINEAR][ENERGY_MU] = report[ENERGY_MU]
            summary_errors_deim[NONLINEAR] = self.mdeim_nonlinear.errors_rom.copy()
            mu_space_deim[NONLINEAR] = self.mdeim_nonlinear.mu_space
        if self.mdeim_nonlinear_lifting:
            report = self.mdeim_nonlinear_lifting.report[OFFLINE]
            NONLINEAR_LIFTING = OperatorType.NONLINEAR_LIFTING
            summary_basis[NONLINEAR_LIFTING][BASIS_WALK] = report[BASIS_WALK]
            summary_basis[NONLINEAR_LIFTING][BASIS_FINAL] = report[BASIS_FINAL]
            summary_sig[NONLINEAR_LIFTING][SPECTRUM_MU] = report[SPECTRUM_MU]
            summary_energy[NONLINEAR_LIFTING][ENERGY_MU] = report[ENERGY_MU]
            summary_errors_deim[
                NONLINEAR_LIFTING
            ] = self.mdeim_nonlinear_lifting.errors_rom.copy()
            mu_space_deim[NONLINEAR_LIFTING] = self.mdeim_nonlinear_lifting.mu_space
        if self.deim_rhs:
            report = self.deim_rhs.report[OFFLINE]
            RHS = OperatorType.RHS
            summary_basis[RHS][BASIS_WALK] = report[BASIS_WALK]
            summary_basis[RHS][BASIS_FINAL] = report[BASIS_FINAL]
            summary_sig[RHS][SPECTRUM_MU] = report[SPECTRUM_MU]
            summary_energy[RHS][ENERGY_MU] = report[ENERGY_MU]
            summary_errors_deim[RHS] = self.deim_rhs.errors_rom.copy()
            mu_space_deim[RHS] = self.deim_rhs.mu_space

        self.summary_basis = pd.DataFrame(summary_basis).T

        # Integration errors
        summary_errors = defaultdict(dict)
        for idx, error in rom.errors.items():
            summary_errors[idx]["mean"] = np.mean(error)
            summary_errors[idx]["median"] = np.median(error)
            summary_errors[idx]["max"] = np.max(error)
            summary_errors[idx]["min"] = np.min(error)

        self.summary_errors = pd.DataFrame(summary_errors).T

    def plot_spectrums(self, save=None, new=True, show=True):
        """Plot reduction spectrums.

        Parameters
        ----------
        save : str, optional
            Figure name, by default None
        new : bool, optional
            Create new figure, by default True
        """

        if new:
            plt.figure()

        SPECTRUM_MU = self.rom.SPECTRUM_MU

        summary_sigmas = self.summary_sigmas

        for element, sigmas_dict in summary_sigmas.items():
            sigma = sigmas_dict[SPECTRUM_MU]
            if sigma is None:
                continue
            sigma = np.log10(sigma)
            element = element.title()
            plt.plot(sigma, label=element)

        plt.xlabel("Up to n-th basis element")
        plt.ylabel("$\\log (\\sigma)$")
        plt.title("Spectrum decay in the parameter space")
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(save + ".png", **self.FIG_KWARGS)

        if show:
            plt.show()

    def plot_energy(self, save=None, show=True):

        ENERGY_MU = self.rom.ENERGY_MU

        summary_energy = self.summary_energy

        plt.figure()

        for element, energy in summary_energy.items():
            sigma = energy[ENERGY_MU]
            element = element.title()
            plt.plot(sigma, label=element)

        plt.xlabel("Up to n-th basis element")
        plt.ylabel("Ratio")
        plt.title("Total POD Energy Ratio")
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(save + ".png", **self.FIG_KWARGS)
        if show:
            plt.show()

    def plot_errors(
        self,
        which=Stage.ONLINE,
        save=None,
        new=True,
        label=None,
        show=True,
    ):

        if new:
            plt.figure()

        rom = self.rom

        errors = self.errors[which]

        for idx, error in errors.items():
            error = np.log10(error)
            plt.plot(rom.timesteps[1:], error, linewidth=1.0, alpha=0.85, label=label)

        plt.xlabel("t")
        plt.ylabel("log10 Error (L2)")
        plt.title(f"{which.capitalize()} Errors")
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()
        if save:
            plt.savefig(save + ".png", **self.FIG_KWARGS)
            plt.close()

    def _run_deim(
        self,
        object: DiscreteEmpiricalInterpolation,
        which: str,
        mu_space: list,
        is_mdeim=False,
        evaluate=False,
    ):
        """Run DEIM offline phase.

        Parameters
        ----------
        object : MatrixDiscreteEmpiricalInterpolation
        which : str
        mu_space : list
            If this is provided, the same collection is used
            for the offline phase and online evaluation.
        is_mdeim : bool, optional
            Which parameters to select, by default True
        evaluate : bool, optional
            Whether to run online evaluation stage, by default False
        """

        # Build collateral basis
        object.run(mu_space=mu_space)

        object.dump_basis()

        if is_mdeim:
            params = self.mdeim_params
        else:
            params = self.deim_params

        # Online evaluation
        if evaluate:
            n_online = params.get(RomParameters.NUM_ONLINE, None)
            timesteps = params["ts"]

            object.evaluate(ts=timesteps, num=n_online, mu_space=mu_space)

        # Include the reduction for the algebraic operators
        rom = self.rom
        rom.add_hyper_reductor(reductor=object, which=which)

    def _run_mdeim(
        self,
        object: MatrixDiscreteEmpiricalInterpolation,
        which: str,
        mu_space: list,
        evaluate=False,
    ):
        """Run MDEIM offline phase.

        Parameters
        ----------
        object : MatrixDiscreteEmpiricalInterpolation
        which : str
        mu_space : list
            If this is provided, the same collection is used
            for the offline phase and online evaluation.
        evaluate : bool, optional
            , by default False
        """
        self._run_deim(
            object=object,
            which=which,
            mu_space=mu_space,
            evaluate=evaluate,
            is_mdeim=True,
        )


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


class HyperReducedPiston(HyperReducedOrderModelFixed):
    def __init__(
        self,
        grid: dict,
        fom_params: dict,
        rom_params: dict,
        deim_params: dict,
        mdeim_params: dict,
        mdeim_nonlinear_params: dict,
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

        self.mdeim_nonlinear_params = mdeim_nonlinear_params.copy()

        self.mdeim_convection = None
        self.deim_moving_runned = False

    def setup(self):
        """Setup up FOM and ROM structures."""

        # ---------------------------------------------------------------------
        # Full Order Model
        # ---------------------------------------------------------------------
        fom_params = self.fom_params
        fom = OneDimensionalBurgers(
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
        rom = RomConstructorNonlinear(fom=fom, grid=self.grid)
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

        mdeim_nonlinear = MatrixDiscreteEmpiricalInterpolationNonlinear(
            name=OperatorType.NONLINEAR,
            assemble=fom.assemble_nonlinear,
            grid=grid,
            tree_walk_params=self.mdeim_nonlinear_params,
        )

        mdeim_nonlinear_lifting = MatrixDiscreteEmpiricalInterpolation(
            name=OperatorType.NONLINEAR_LIFTING,
            assemble=fom.assemble_nonlinear_lifting,
            grid=grid,
            tree_walk_params=self.mdeim_params,
        )

        mdeim_convection.setup(rnd=rnd)
        mdeim_nonlinear_lifting.setup(rnd=rnd)
        mdeim_nonlinear.setup(rnd=rnd, V=fom.V)

        self.mdeim_convection = mdeim_convection
        self.mdeim_nonlinear = mdeim_nonlinear
        self.mdeim_nonlinear_lifting = mdeim_nonlinear_lifting

    def run_offline_hyperreduction(self, mu_space=None, u_n=None, evaluate=True):
        """Generate collateral basis for algebraic operators."""

        super().run_offline_hyperreduction(mu_space=mu_space, evaluate=evaluate)

        mdeim_convection = self.mdeim_convection
        mdeim_nonlinear = self.mdeim_nonlinear
        mdeim_nonlinear_lifting = self.mdeim_nonlinear_lifting

        if u_n is None:
            u_n = self.basis

        if self.models[OperatorType.CONVECTION]:
            self._run_mdeim(
                object=mdeim_convection,
                mu_space=mu_space,
                evaluate=evaluate,
                which=OperatorType.CONVECTION,
            )
        if self.models[OperatorType.NONLINEAR_LIFTING]:
            self._run_mdeim(
                object=mdeim_nonlinear_lifting,
                mu_space=mu_space,
                evaluate=evaluate,
                which=OperatorType.NONLINEAR_LIFTING,
            )
        if self.models[OperatorType.NONLINEAR]:
            self._run_mdeim_nonlinear(
                object=mdeim_nonlinear,
                mu_space=mu_space,
                evaluate=evaluate,
                which=OperatorType.NONLINEAR,
                u_n=u_n,
            )

        self.deim_moving_runned = True

    def _run_mdeim_nonlinear(
        self,
        object: MatrixDiscreteEmpiricalInterpolationNonlinear,
        u_n,
        which: str,
        mu_space: list,
        evaluate=False,
    ):
        """Generate N-MDEIM, feeding it with the FOM reduced basis.

        Parameters
        ----------
        object : MatrixDiscreteEmpiricalInterpolationNonlinear
        u_n : np.array
        which : str
        mu_space : list
        evaluate : bool, optional
            , by default False
        """

        # Build collateral basis
        object.run(u_n=u_n, mu_space=mu_space)

        object.dump_basis()

        # Online evaluation
        if evaluate:
            n_online = self.mdeim_params.get(RomParameters.NUM_ONLINE, None)
            timesteps = self.mdeim_params["ts"]

            object.evaluate(ts=timesteps, num=n_online, mu_space=mu_space)

        # Include the reduction for the algebraic operators
        rom = self.rom
        rom.add_hyper_reductor(reductor=object, which=which)
