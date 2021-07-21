from .base import OneDimensionalSolver, move_mesh
from .heat import HeatEquationSolver, HeatEquationMovingSolver
from .nonlinear import OneDimensionalBurgers

__all__ = [
    "move_mesh",
    "OneDimensionalSolver",
    "HeatEquationSolver",
    "HeatEquationMovingSolver",
    "OneDimensionalBurgers",
]
