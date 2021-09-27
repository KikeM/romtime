FIG_KWARGS = dict(dpi=300, bbox_inches="tight")


class StorageNames:

    ROM = "basis_rom.pkl"
    SROM = "basis_srom.pkl"

    VALIDATION_SOLUTIONS = "validation_solutions.pkl"
    SETUP = "setup.json"
    MU_SPACE = "mu_space.json"
    MU_SPACE_DEIM = "mu_space_deim.json"


class ProblemType:

    FOM = "fom"
    ROM = "rom"


class MassConservation:

    WHICH = "which"
    TIMESTEPS = "timesteps"
    MASS = "mass"
    MASS_CHANGE = "mass_change"
    OUTFLOW = "outflow"


class Errors(ProblemType):

    SACRIFICIAL = "sacrificial"
    ESTIMATOR = "estimator"

    AVERAGE_ROM = "rom_average"
    AVERAGE_ESTIMATOR = "estimator_average"
    AVERAGE_SACRIFICIAL = "srom_average"


class BDF:

    ONE = "1"
    TWO = "2"


class OperatorType(ProblemType):

    CONVECTION = "convection"
    FORCING = "forcing"
    LIFTING = "lifting"
    MASS = "mass"
    NONLINEAR = "nonlinear"
    NONLINEAR_LIFTING = "nonlinear-lifting"
    REDUCED_BASIS = "reduced-basis"
    RHS = "rhs"
    STIFFNESS = "stiffness"


class Treewalk:

    BASIS_AFTER_WALK = "basis-shape-after-tree-walk"
    BASIS_FINAL = "basis-shape-final"
    BASIS_TIME = "basis-shape-time"
    ENERGY_MU = "energy-mu"
    ENERGY_TIME = "energy-time"
    SPECTRUM_MU = "spectrum-mu"
    SPECTRUM_TIME = "spectrum-time"


class EmpiricalInterpolation:

    DEIM = "DEIM"
    MDEIM = "MDEIM"
    NONLINEAR = "N-MDEIM"


class Stage:

    OFFLINE = "offline"
    VALIDATION = "validation"
    ONLINE = "online"


class BoundaryConditions:

    # Dirichlet values
    B0 = "b0"
    BL = "bL"

    # Time derivatives
    DB0_DT = "db0_dt"
    DBL_DT = "dbL_dt"


class Domain:

    NT = "nt"
    NX = "nx"
    T = "T"
    L0 = "L0"


class RomParameters:

    # Online phase
    NUM_ONLINE = "num_online"

    # S-ROM definition
    SROM_TRUNCATE = "srom_truncate"  # How many modes to remove
    SROM_KEEP = "srom_num"  # How many modes to keep

    # Offline phase
    NUM_BASIS = "num_phi"
    NUM_MU = "num_mu"
    NUM_SNAPSHOTS = "num_snapshots"
    NUM_TIME = "num_time"
    TOL_BASIS = "tol_phi"
    TOL_MU = "tol_mu"
    TOL_TIME = "tol_time"
    TS = "ts"


class PistonParameters:

    ALPHA = "alpha"
    DELTA = "delta"
    GAMMA = "gamma"
    OMEGA = "omega"
    A0 = "a0"

    FORCING = "forcing"