"""Compatibility entry point; implementation moved with its workflow."""
import sys
from experiments.generators import solver_lab_data as _implementation
sys.modules[__name__] = _implementation
