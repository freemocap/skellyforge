"""Compatibility import; generator lives under experiments."""
import sys
from experiments.generators import solver_chain_experiment as _implementation
sys.modules[__name__] = _implementation
