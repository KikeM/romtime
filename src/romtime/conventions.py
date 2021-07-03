FIG_KWARGS = dict(dpi=300, bbox_inches="tight")


class OperatorType:

    MASS = "mass"
    STIFFNESS = "stiffness"
    CONVECTION = "convection"
    FORCING = "forcing"
    LIFTING = "lifting"
    RHS = "rhs"

    FOM = "fom"
    ROM = "rom"


class Stage:

    OFFLINE = "offline"
    VALIDATION = "validation"
    ONLINE = "online"


class BoundaryConditions:

    # Dirichlet values
    B0 = "b0"
    BL = "bL"

    # Time-derivatives
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

    # Offline phase
    TS = "ts"
    NUM_SNAPSHOTS = "num_snapshots"
    NUM_MU = "num_mu"
    TOL_MU = "tol_mu"
    NUM_TIME = "num_time"
    TOL_TIME = "tol_time"