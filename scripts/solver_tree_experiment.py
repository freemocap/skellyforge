"""Compatibility import; generator lives under experiments."""
import sys
from experiments.generators import solver_tree_experiment as _implementation
sys.modules[__name__] = _implementation
