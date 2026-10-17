"""The ontology's linkage layer: joints between segment pairs.

A joint is two segments joined at the landmark they share. Its static face is a
`JointDefinition` (authored in the skeleton's `joints:` YAML); its hydrated face
is a `JointPose` - the relative orientation plus its decomposition into named
angles under the joint's convention, carrying the provenance of what fed it.
"""

from skellyforge.core.skeleton.linkage.joint_definition import (
    EulerConvention,
    JointDefinition,
)
from skellyforge.core.skeleton.linkage.joint_pose import (
    JointInputProvenance,
    JointPose,
    compute_joint_poses,
    relative_orientation,
)

__all__ = [
    "EulerConvention",
    "JointDefinition",
    "JointInputProvenance",
    "JointPose",
    "compute_joint_poses",
    "relative_orientation",
]
