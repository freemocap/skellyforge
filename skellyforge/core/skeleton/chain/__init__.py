"""The ontology's chain layer: declared multi-segment paths over the joints.

Forward synthesis lives in ``synthesis.py`` and is imported directly by its
callers - keeping it out of this package's ``__init__`` avoids an import cycle
(synthesis needs the rest pose's forward pass, which needs the skeleton).
"""

from skellyforge.core.skeleton.chain.kinematic_chain import KinematicChain

__all__ = [
    "KinematicChain",
]
