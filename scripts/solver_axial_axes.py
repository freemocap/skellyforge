"""Compatibility entry point; implementation moved with its workflow."""
import sys
from experiments.generators import solver_axial_axes as _implementation
sys.modules[__name__] = _implementation
