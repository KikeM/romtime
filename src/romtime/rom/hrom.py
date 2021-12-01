from collections import defaultdict
from functools import partial
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ujson
from romtime.conventions import (
    FIG_KWARGS,
    Errors,
    OperatorType,
    ProbeLocations,
    ProblemType,
    RomParameters,
    Stage,
    StorageNames,
    Treewalk,
    TreewalkNonlinear,
)
from romtime.deim import (
    DiscreteEmpiricalInterpolation,
    MatrixDiscreteEmpiricalInterpolation,
)
from romtime.deim.nonlinear import MatrixDiscreteEmpiricalInterpolationNonlinear
from romtime.fom import HeatEquationMovingSolver, HeatEquationSolver
from romtime.fom.nonlinear import OneDimensionalBurgers
from romtime.rom import RomConstructor, RomConstructorMoving
from romtime.rom.rom import RomConstructorNonlinear
from romtime.utils import (
    compute_rom_difference,
    dump_csv,
    dump_json,
    dump_pickle,
    read_json,
    read_pickle,
)
from tqdm import tqdm

sns.set_theme(context="paper", palette="viridis")


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
        self.srom = RomConstructorNonlinear
        self.deim_rhs = None
        self.mdeim_mass = None
        self.mdeim_stiffness = None
        self.mdeim_convection = None
        self.mdeim_nonlinear = None
        self.mdeim_trilinear = None
        self.mdeim_trilinear_lifting = None

        self.deim_runned = False
        self.rom_runned = False

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
        del self.mdeim_trilinear
        del self.mdeim_trilinear_lifting

        del self.deim_runned
        del self.rom_runned

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

    @property
    def basis(self):
        """Reduced Order Basis (V)."""
        return self.rom.basis

    def dump_mu_space(self, path=None):
        """Write ROM parameter space to a json file."""
        if path is None:
            path = StorageNames.MU_SPACE

        dump_json(path, self.mu_space)

    def dump_mu_space_deim(self, path=None):
        """Write N-(M)DEIM parameter space to a json file."""
        if path is None:
            path = StorageNames.MU_SPACE_DEIM

        dump_json(path, self.mu_space_deim)

    def dump_reduced_basis(self, path=None):
        """Pickle ROM and S-ROM basis vectors."""

        if path is None:
            path_rom = StorageNames.ROM
            path_srom = StorageNames.SROM

        dump_pickle(path_rom, self.basis)
        dump_pickle(path_srom, self.srom.basis)

    def dump_nonlinear_basis(self, path=None):
        """Pickle nonlinear term basis."""

        dump_pickle(
            f"basis_fom_n-mdeim_{OperatorType.TRILINEAR}.pkl", self.srom.basis_nonlinear
        )

    def dump_validation_fom(self, path=None):
        """Pickle validation FOM solutions."""

        if path is None:
            path = StorageNames.VALIDATION_SOLUTIONS

        dump_pickle(path, self.validation_solutions)

    def dump_errors(self, which, path=None):

        if path is None:
            path = Path(".")

        errors = self.errors

        is_present = which in errors
        if is_present:
            pd.DataFrame(errors[which]).to_csv(path / f"errors_{which}.csv")
        else:
            raise Warning("These errors ({which}) have not been computed yet.")

    def dump_errors_deim(self, path=None):

        if path is None:
            path = Path(".")

        errors_deim = self.summary_errors_deim

        for operator, errors in errors_deim.items():
            df = pd.DataFrame(errors)
            has_data = not df.empty
            if has_data:
                file = path / f"errors_deim_{operator.lower()}.csv"
                df.to_csv(file)

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
            path = StorageNames.VALIDATION_SOLUTIONS

        try:
            self.validation_solutions = read_pickle(path)
        except FileNotFoundError:
            print("Validation solutions are not defined.")

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

    def project_reductors(self):
        """Project collateral basis unto the RB space."""

        rom = self.rom
        srom = self.srom

        rom.project_reductors()
        srom.project_reductors()

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

        num_snapshots = self.rom_params[RomParameters.NUM_SNAPSHOTS]
        num_basis = self.rom_params.get(RomParameters.NUM_MU, None)

        TOL_TIME = RomParameters.TOL_TIME
        TOL_MU = RomParameters.TOL_MU

        tolerances = {
            TOL_TIME: self.rom_params.get(TOL_TIME, None),
            TOL_MU: self.rom_params.get(TOL_MU, None),
        }

        # ---------------------------------------------------------------------
        # Build Reduced Basis (RB)
        # We build it with the S-ROM, because we are then going to remove modes
        srom = self.srom
        fom_solutions = srom.build_reduced_basis(
            num_snapshots=num_snapshots,
            mu_space=mu_space,
            num_basis=num_basis,
            tolerances=tolerances,
        )

        # Truncate basis
        n = self.rom_params[RomParameters.SROM_TRUNCATE]
        rom = srom.truncate(n=n)
        rom.name = "ROM"
        self.rom = rom

        self.validation_solutions = fom_solutions

        print(f"(ROM) N = {rom.N}")
        print(f"(SROM) N = {srom.N}")

    def start_from_existing_basis(self, deim=True, size=dict()):

        # ---------------------------------------------------------------------
        # For validation tests
        self.load_validation_fom()

        # ---------------------------------------------------------------------
        # Read ROB basis
        print("Loading RB Basis from disk ...")

        try:
            mu_space = read_json(StorageNames.MU_SPACE)
            mu_space[Stage.ONLINE] = list()
        except FileNotFoundError:
            print("No mu_space detected, creating on empty ...")
            mu_space = {
                Stage.OFFLINE: list(),
                Stage.ONLINE: list(),
                Stage.VALIDATION: list(),
            }

        # S-ROM
        srom = self.srom
        basis_srom = read_pickle(StorageNames.SROM)
        N_srom = self.rom_params[RomParameters.SROM_KEEP]
        print(f"Loaded basis size: {basis_srom.shape[1]}.")
        print(f"Required SROM size: {N_srom}.")
        if N_srom is not None:
            basis_srom = basis_srom[:, :N_srom]

        srom.load_from_basis(basis=basis_srom, mu_space=mu_space)

        del basis_srom

        # ROM
        self.rom = srom.truncate(self.rom_params[RomParameters.SROM_TRUNCATE])

        print(f"S-ROM size: {self.srom.basis.shape}")
        print(f"ROM size: {self.rom.basis.shape}")

        # ---------------------------------------------------------------------
        # Read DEIM basis
        if deim:
            self.deim_rhs.load_fom_basis(keep=size.get(OperatorType.RHS, None))
            self.mdeim_mass.load_fom_basis(keep=size.get(OperatorType.MASS, None))
            self.mdeim_stiffness.load_fom_basis(
                keep=size.get(OperatorType.STIFFNESS, None)
            )
            self.mdeim_convection.load_fom_basis(
                keep=size.get(OperatorType.CONVECTION, None)
            )
            self.mdeim_trilinear_lifting.load_fom_basis(
                keep=size.get(OperatorType.NONLINEAR_LIFTING, None)
            )

            N_mdeim_trilinear = self.rom_params[RomParameters.NMDEIM_SIZE]
            self.mdeim_trilinear.load_fom_basis(keep=N_mdeim_trilinear)

            # ---------------------------------------------------------------------
            # Connect (M)DEIM with ROM models
            # Include the reduction for the algebraic operators
            operators = [
                OperatorType.LIFTING,
                OperatorType.MASS,
                OperatorType.STIFFNESS,
                OperatorType.CONVECTION,
                OperatorType.NONLINEAR_LIFTING,
                OperatorType.TRILINEAR,
            ]

            deims = [
                self.deim_rhs,
                self.mdeim_mass,
                self.mdeim_stiffness,
                self.mdeim_convection,
                self.mdeim_trilinear_lifting,
                self.mdeim_trilinear,
            ]

            for reductor, which in zip(deims, operators):
                for _rom in [self.rom, self.srom]:
                    _rom.add_hyper_reductor(reductor=reductor, which=which)

    def run_offline_hyperreduction(self, mu_space=None, evaluate=True):
        """Generate collateral basis for linear algebraic operators."""

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
        """(H)ROM evaluation for training parameter set."""

        rom = self.rom
        space = rom.mu_space[Stage.OFFLINE]
        self._evaluate(which=Stage.VALIDATION, mu_space=space)

    def evaluate_online(self, params, rnd=None):
        """(H)ROM online evaluation for a random parameter set.

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

    def evaluate_deim(self):
        """Evaluate all DEIM models."""

        mu_space = self.mu_space[Stage.OFFLINE]
        self.evaluate_deim_model(object=self.deim_rhs, mu_space=mu_space)

        self.evaluate_deim_model(object=self.mdeim_mass, mu_space=mu_space)
        self.evaluate_deim_model(object=self.mdeim_stiffness, mu_space=mu_space)
        self.evaluate_deim_model(object=self.mdeim_convection, mu_space=mu_space)
        self.evaluate_deim_model(object=self.mdeim_trilinear_lifting, mu_space=mu_space)

        self.evaluate_deim_model(object=self.mdeim_trilinear, mu_space=mu_space)

    def _evaluate(self, which, estimate_error=True, mu_space=None):
        """(H)ROM evaluation for a set of parameters."""

        fom = self.fom
        rom = self.rom
        srom = self.srom

        if fom.exact_solution is None:
            rom_fom_errors = dict()

        mu_space = list(mu_space)

        print(f"Evaluation for {which}")
        pprint(mu_space)

        compute_error = rom._compute_error
        desc = f"(HROM) {which.upper()} evaluation"
        for mu in tqdm(mu_space, desc=desc, leave=True):

            # -----------------------------------------------------------------
            # Solve ROM
            idx_mu = rom.solve(mu=mu, step=which)

            name_rom_solutions = f"solutions_rom_{rom.N}_{which}_{idx_mu}"
            rom.solutions.to_pickle(name_rom_solutions)

            # -----------------------------------------------------------------
            # Compare against FOM
            if which == Stage.VALIDATION:
                # Recover from validation storage
                uh_fom = self.validation_solutions[idx_mu]
            else:
                # Online stage requires actual simulation
                fom.setup()
                fom.update_parametrization(mu)
                fom.solve()
                uh_fom = fom.solutions.fom

            name_fom_solutions = f"solutions_fom_{which}_{idx_mu}"
            fom.solutions.to_pickle(name_fom_solutions)

            # -----------------------------------------------------------------
            # Compute errors
            uh_rom = rom.solutions.fom

            nt = uh_fom.shape[1]
            _compute_error = lambda uh: [
                compute_error(uh_fom[:, idx], uh[:, idx]) for idx in range(nt)
            ]

            # Convert to array format
            errors_rom = _compute_error(uh_rom)
            errors_rom = np.array(errors_rom)

            # -----------------------------------------------------------------
            # Compute error estimator
            if estimate_error == True:

                # Solve SROM
                srom.solve(mu=mu, step=which)
                name_srom_solutions = f"solutions_srom_{srom.N}_{which}_{idx_mu}"
                srom.solutions.to_pickle(name_srom_solutions)

                uh_srom = srom.solutions.fom
                errors_srom = _compute_error(uh_srom)
                errors_srom = np.array(errors_srom)

                # Estimator
                uNs = rom.solutions.rom
                uNs_srom = srom.solutions.rom

                V_srom = srom.basis

                estimator = np.array([])
                for nt in range(uNs.shape[1]):
                    uN = uNs[:, nt]
                    uN_scf = uNs_srom[:, nt]
                    error = compute_rom_difference(uN=uN, uN_srom=uN_scf, V_srom=V_srom)
                    estimator = np.append(estimator, [error])
            else:
                estimator = np.array()
                errors_srom = np.array()

            # -----------------------------------------------------------------
            # Prepare errors payload
            payload_errors = {
                Errors.ESTIMATOR: estimator,
                Errors.ROM: errors_rom,
                Errors.SACRIFICIAL: errors_srom,
            }

            rom_fom_errors.update({idx_mu: payload_errors})

            # -----------------------------------------------------------------
            # Compute FOM and ROM solution
            if fom.RUNTIME_PROCESS & (which == Stage.ONLINE):
                name_probes = f"probes_{which}_fom_{idx_mu}.csv"
                probes = fom.save_probes(name=name_probes)

                piston = probes["L"].squeeze()
                piston.name = "piston"
                name_probes_comparison = (
                    f"probes_comparison_rom_{rom.N}_srom_{srom.N}_{which}_{idx_mu}.csv"
                )
                self.save_fom_rom_probes(
                    name=name_probes_comparison,
                    piston=piston,
                    fom=fom,
                    rom=rom,
                    srom=srom,
                )

            # -----------------------------------------------------------------
            # Compute mass conservation
            timesteps = rom.timesteps

            # ROM
            rom_sols = uh_rom.T
            output_rom = fom.compute_mass_conservation(
                mu=mu, ts=timesteps, solutions=rom_sols, which=ProblemType.ROM
            )
            name_rom = (
                f"mass_conservation_rom_{rom.N}_srom_{srom.N}_{which}_rom_{idx_mu}.csv"
            )
            dump_csv(name_rom, obj=output_rom)

            # FOM
            fom_sols = uh_fom.T
            output_fom = fom.compute_mass_conservation(
                mu=mu, ts=timesteps, solutions=fom_sols, which=ProblemType.FOM
            )
            name_fom = f"mass_conservation_{which}_fom_{idx_mu}.csv"
            dump_csv(name_fom, obj=output_fom)

        if fom.exact_solution is None:
            self.errors[which] = rom_fom_errors
        else:
            self.errors[which] = rom.errors_rom

    @staticmethod
    def compare_models(x, piston, ts, fom, rom, srom):

        x_fom = fom.solutions.compute_at(x=x)
        x_rom = rom.solutions.compute_at(x=x)
        x_srom = srom.solutions.compute_at(x=x)

        data = np.vstack([x_fom, x_rom, x_srom]).T

        columns = [ProblemType.FOM, ProblemType.ROM, ProblemType.SROM]
        df = pd.DataFrame(data=data, index=ts, columns=columns)

        df[ProbeLocations.PISTON] = piston

        return df

    def save_fom_rom_probes(self, name, piston, fom, rom, srom):
        """Compare models at the ouflow and piston halfway.

        Parameters
        ----------
        name : str
        piston : pandas.Series
        fom : OneDimensionalSolver-like
        rom : RomConstructor-like
        srom : RomConstructor-like

        Returns
        -------
        outflow : pandas.DataFrame
        half : pandas.DataFrame
        """

        ts = rom.solutions.ts

        compare_models = partial(
            self.compare_models,
            fom=fom,
            rom=rom,
            srom=srom,
            piston=piston,
            ts=ts,
        )

        outflow = compare_models(x=0.0)
        half = compare_models(x=0.5)

        name_outflow = "_".join([ProbeLocations.OUTFLOW, name])
        name_half = "_".join([ProbeLocations.MIDDLE, name])

        outflow.to_csv(name_outflow)
        half.to_csv(name_half)

        return outflow, half

    def generate_summary(self):
        """Generate reduction summaries."""

        summary_basis = self._summary_basis
        summary_sig = self.summary_sigmas
        summary_energy = self.summary_energy
        summary_errors_deim = self.summary_errors_deim
        mu_space_deim = self.mu_space_deim

        OFFLINE = Stage.OFFLINE
        # ---------------------------------------------------------------------
        # ROM summary
        rom = self.rom
        BASIS_WALK = Treewalk.BASIS_AFTER_WALK
        BASIS_FINAL = Treewalk.BASIS_FINAL
        SPECTRUM_MU = Treewalk.SPECTRUM_MU
        SPECTRUM_TIME = Treewalk.SPECTRUM_TIME
        ENERGY_MU = rom.ENERGY_MU

        report = rom.report[OFFLINE]
        REDUCED_BASIS = OperatorType.REDUCED_BASIS
        summary_basis[REDUCED_BASIS][BASIS_WALK] = report[BASIS_WALK]
        summary_basis[REDUCED_BASIS][BASIS_FINAL] = report[BASIS_FINAL]
        summary_sig[REDUCED_BASIS][SPECTRUM_MU] = report[SPECTRUM_MU]
        summary_sig[REDUCED_BASIS][SPECTRUM_TIME] = report[SPECTRUM_TIME]
        summary_energy[REDUCED_BASIS][ENERGY_MU] = report[ENERGY_MU]

        # ---------------------------------------------------------------------
        # Nonlinear operator summary
        N_BASIS_WALK = TreewalkNonlinear.BASIS_AFTER_WALK
        N_BASIS_FINAL = TreewalkNonlinear.BASIS_FINAL
        N_SPECTRUM_MU = TreewalkNonlinear.SPECTRUM_MU
        N_SPECTRUM_TIME = TreewalkNonlinear.SPECTRUM_TIME
        N_ENERGY_MU = TreewalkNonlinear.ENERGY_MU

        report = rom.report[OFFLINE]
        TRILINEAR = OperatorType.TRILINEAR
        summary_basis[TRILINEAR][BASIS_WALK] = report[N_BASIS_WALK]
        summary_basis[TRILINEAR][BASIS_FINAL] = report[N_BASIS_FINAL]
        summary_sig[TRILINEAR][SPECTRUM_MU] = report[N_SPECTRUM_MU]
        summary_sig[TRILINEAR][SPECTRUM_TIME] = report[N_SPECTRUM_TIME]
        summary_energy[TRILINEAR][ENERGY_MU] = report[N_ENERGY_MU]
        mu_space_deim[TRILINEAR] = self.mdeim_trilinear.mu_space
        summary_errors_deim[TRILINEAR] = self.mdeim_trilinear.errors_rom.copy()

        # ---------------------------------------------------------------------
        # Algebraic operators summary
        generate_operator_summary = partial(
            self.generate_operator_summary,
            basis=summary_basis,
            sigma=summary_sig,
            energy=summary_energy,
            errors_deim=summary_errors_deim,
            mu_space_deim=mu_space_deim,
        )

        algebraic_operators = [
            self.deim_rhs,
            self.mdeim_mass,
            self.mdeim_stiffness,
            # self.mdeim_nonlinear,
            self.mdeim_convection,
            self.mdeim_trilinear_lifting,
        ]
        for operator in algebraic_operators:
            if operator is not None:
                generate_operator_summary(operator)

        self.summary_basis = pd.DataFrame(summary_basis).T

        # ---------------------------------------------------------------------
        # Integration errors
        summary_errors = defaultdict(dict)
        for idx, error in rom.errors.items():
            summary_errors[idx]["mean"] = np.mean(error)
            summary_errors[idx]["median"] = np.median(error)
            summary_errors[idx]["max"] = np.max(error)
            summary_errors[idx]["min"] = np.min(error)

        self.summary_errors = pd.DataFrame(summary_errors).T

    @staticmethod
    def generate_operator_summary(
        operator,
        basis,
        sigma,
        energy,
        errors_deim,
        mu_space_deim,
    ):
        """Update summary dictionary with operator DEIM results.

        Parameters
        ----------
        operator : DiscreteEmpiricalInterpolationMethod-like
        basis : dict
        sigma : dict
        energy : dict
        errors_deim : dict
        mu_space_deim : dict
        """

        BASIS_WALK = Treewalk.BASIS_AFTER_WALK
        BASIS_FINAL = Treewalk.BASIS_FINAL
        SPECTRUM_MU = Treewalk.SPECTRUM_MU
        SPECTRUM_TIME = Treewalk.SPECTRUM_TIME
        ENERGY_MU = Treewalk.ENERGY_MU
        OFFLINE = Stage.OFFLINE
        NAME = operator.name
        report = operator.report[OFFLINE]

        # Treewalk results
        basis[NAME][BASIS_WALK] = report[BASIS_WALK]
        basis[NAME][BASIS_FINAL] = report[BASIS_FINAL]
        sigma[NAME][SPECTRUM_MU] = report[SPECTRUM_MU]
        sigma[NAME][SPECTRUM_TIME] = report[SPECTRUM_TIME]
        energy[NAME][ENERGY_MU] = report[ENERGY_MU]

        errors_deim[NAME] = operator.errors_rom.copy()
        mu_space_deim[NAME] = operator.mu_space

    def evaluate_deim_model(self, object, mu_space):
        """Evaluate individual DEIM model.

        Parameters
        ----------
        object : DiscreteEmpiricalInterpolationMethod-like
        mu_space : dict
        """
        params = object.tree_walk_params

        n_online = params.get(RomParameters.NUM_ONLINE, None)
        timesteps = params["ts"]

        object.evaluate(ts=timesteps, num=n_online, mu_space=mu_space)

    def evaluate_nmdeim_model(
        self,
        object: MatrixDiscreteEmpiricalInterpolationNonlinear,
        mu_space=None,
        funcs=None,
    ):
        """Evaluate N-MDEIM model.

        Parameters
        ----------
        object : MatrixDiscreteEmpiricalInterpolationNonlinear-like
        mu_space : dict
        """
        params = object.tree_walk_params

        n_online = params.get(RomParameters.NUM_ONLINE, None)
        timesteps = params["ts"]
        num_psi = params[RomParameters.NUM_PSI_NMDEIM]

        funcs = self.basis[:, :num_psi]

        object.evaluate(ts=timesteps, funcs=funcs, num=n_online, mu_space=mu_space)

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
        if is_mdeim:
            params = self.mdeim_params
        else:
            params = self.deim_params

        if mu_space is None:
            mu_space_offline = self.rom.build_sampling_space(
                num=params[RomParameters.NUM_SNAPSHOTS],
                rnd=params[RomParameters.RND],
            )
        else:
            mu_space_offline = mu_space

        # Build collateral basis
        object.run(mu_space=mu_space_offline)
        object.dump_fom_basis()

        # Online evaluation
        if evaluate:
            mu_space_online = self.rom.build_sampling_space(
                num=params[RomParameters.NUM_ONLINE],
                rnd=params[RomParameters.RND_ONLINE],
            )
            self.evaluate_deim_model(object=object, mu_space=mu_space_online)

        # Include the reduction for the algebraic operators
        for rom in [self.rom, self.srom]:
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
        rom = RomConstructorMoving(fom=fom, grid=self.grid, name="ROM")
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
        """Generate collateral basis for algebraic operators.

        - Linear
        - Convection
        """

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
        rom = RomConstructorNonlinear(fom=fom, grid=self.grid, name="ROM")
        rom.setup(rnd=rnd)

        # Extended S-ROM to compute error estimator
        srom = RomConstructorNonlinear(fom=fom, grid=self.grid, name="S-ROM")
        srom.setup(rnd=rnd)

        self.srom = srom
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

        mdeim_trilinear_lifting = MatrixDiscreteEmpiricalInterpolation(
            name=OperatorType.NONLINEAR_LIFTING,
            assemble=fom.assemble_nonlinear_lifting,
            grid=grid,
            tree_walk_params=self.mdeim_params,
        )

        mdeim_trilinear = MatrixDiscreteEmpiricalInterpolationNonlinear(
            name=OperatorType.TRILINEAR,
            assemble=fom.assemble_trilinear,
            grid=grid,
            tree_walk_params=self.mdeim_nonlinear_params,
        )

        mdeim_nonlinear = MatrixDiscreteEmpiricalInterpolationNonlinear(
            name=OperatorType.NONLINEAR,
            assemble=fom.assemble_nonlinear,
            grid=grid,
            tree_walk_params=self.mdeim_nonlinear_params,
        )

        mdeim_convection.setup(rnd=rnd)
        mdeim_trilinear_lifting.setup(rnd=rnd)
        mdeim_trilinear.setup(rnd=rnd, V=fom.V)
        mdeim_nonlinear.setup(rnd=rnd, V=fom.V)

        self.mdeim_convection = mdeim_convection
        self.mdeim_trilinear_lifting = mdeim_trilinear_lifting
        self.mdeim_trilinear = mdeim_trilinear
        self.mdeim_nonlinear = mdeim_nonlinear

    def run_offline_hyperreduction(self, mu_space=None, u_n=None, evaluate=True):
        """Generate collateral basis for algebraic operators.

        - Linear
        - Convection
        - Nonlinear
        """

        super().run_offline_hyperreduction(mu_space=mu_space, evaluate=evaluate)

        if self.models[OperatorType.CONVECTION]:

            mdeim_convection = self.mdeim_convection

            self._run_mdeim(
                object=mdeim_convection,
                mu_space=mu_space,
                evaluate=evaluate,
                which=OperatorType.CONVECTION,
            )

        if self.models[OperatorType.NONLINEAR_LIFTING]:

            mdeim_nonlinear_lifting = self.mdeim_trilinear_lifting

            self._run_mdeim(
                object=mdeim_nonlinear_lifting,
                mu_space=mu_space,
                evaluate=evaluate,
                which=OperatorType.NONLINEAR_LIFTING,
            )

        # ---------------------------------------------------------------------
        # Nonlinear operator
        if self.models[OperatorType.TRILINEAR]:

            mdeim_trilinear = self.mdeim_trilinear

            # Get basis from FOM
            basis_mdeim = self.srom.basis_nonlinear

            # Use ROB basis vectors
            if u_n is None:
                u_n = self.basis

            self._run_mdeim_nonlinear(
                object=mdeim_trilinear,
                mu_space=mu_space,
                evaluate=evaluate,
                which=OperatorType.TRILINEAR,
                u_n=u_n,
                basis=basis_mdeim,
            )

        self.deim_moving_runned = True

    def _run_mdeim_nonlinear(
        self,
        object: MatrixDiscreteEmpiricalInterpolationNonlinear,
        u_n,
        which: str,
        mu_space: list,
        evaluate=False,
        basis=None,
    ):
        """Generate N-MDEIM.

        Parameters
        ----------
        object : MatrixDiscreteEmpiricalInterpolationNonlinear
        u_n : np.array
        which : str
        mu_space : list
        evaluate : bool
            By default False.
        """

        params = self.mdeim_nonlinear_params

        # ---------------------------------------------------------------------
        # Build collateral basis
        # (A) Assembling operator with the reduced basis elements
        if basis is None:

            # Create sampling space
            if mu_space is None:
                mu_space_offline = self.rom.build_sampling_space(
                    num=params[RomParameters.NUM_SNAPSHOTS],
                    rnd=params[RomParameters.RND],
                )
            else:
                mu_space_offline = mu_space

            # Running offline stage
            object.run(u_n=u_n, mu_space=mu_space_offline)
            object.dump_fom_basis()

        # (B) Using the snapshots from the FOM model
        else:
            object.load_fom_basis(basis=basis)

        # ---------------------------------------------------------------------
        # Online evaluation
        if evaluate:
            mu_space_online = self.rom.build_sampling_space(
                num=params[RomParameters.NUM_ONLINE],
                rnd=params[RomParameters.RND_ONLINE],
            )
            self.evaluate_nmdeim_model(object=object, mu_space=mu_space_online)

        # ---------------------------------------------------------------------
        # Include the reduction for the algebraic operators
        for rom in [self.rom, self.srom]:
            rom.add_hyper_reductor(reductor=object, which=which)
