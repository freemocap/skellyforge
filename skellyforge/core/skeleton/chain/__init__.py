"""The ontology's chain layer: declared multi-segment paths over the joints.

Forward synthesis lives in ``synthesis.py`` and is imported directly by its
callers - keeping it out of this package's ``__init__`` avoids an import cycle
(synthesis needs the rest pose's forward pass, which needs the skeleton). The
IK solvers are pure position math with no skeleton dependency, so they export
cleanly from here.
"""

from skellyforge.core.skeleton.chain.fabrik_ik import (
    DEFAULT_FABRIK_MAX_ITERATIONS,
    DEFAULT_FABRIK_TOLERANCE_MM,
    FabrikSolution,
    solve_fabrik,
)
from skellyforge.core.skeleton.chain.kinematic_chain import KinematicChain
from skellyforge.core.skeleton.chain.two_bone_ik import (
    TwoBoneIkSolution,
    solve_two_bone_ik,
)

__all__ = [
    "DEFAULT_FABRIK_MAX_ITERATIONS",
    "DEFAULT_FABRIK_TOLERANCE_MM",
    "FabrikSolution",
    "KinematicChain",
    "TwoBoneIkSolution",
    "solve_fabrik",
    "solve_two_bone_ik",
]
