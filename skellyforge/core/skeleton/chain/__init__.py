"""The ontology's chain layer: declared multi-segment paths over the joints.

Submodules are imported directly rather than re-exported here:
``synthesis`` needs the rest pose's forward pass and ``twist_backfill`` needs
roll resolution - both reach back to ``skeleton_definition``, so eager
re-exports would close an import cycle.
"""

from skellyforge.core.skeleton.chain.kinematic_chain import KinematicChain

__all__ = [
    "KinematicChain",
]
