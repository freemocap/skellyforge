"""Shared type aliases: the array type beartype can check, and the name-string vocabulary.

The name aliases are `str` at runtime and carry no enforcement. They are here for the
reader: a signature saying `Mapping[LandmarkNameString, Point]` says which of the several
kinds of name it wants, where `Mapping[str, Point]` says only that it is a string. They
live in one file so the vocabulary is enumerable - the ontology's layers are keypoint ->
landmark -> segment -> linkage -> chain -> skeleton, and each layer's name type is below.
"""

from typing import Any

import numpy as np

# Written as an explicit ``np.ndarray[..., np.dtype[...]]`` subscription rather than
# ``numpy.typing.NDArray[np.float64]``: under numpy >= 2.5 ``NDArray`` is a PEP-695
# alias whose ``ScalarT`` typevar beartype cannot reduce to a concrete dtype, which
# makes beartype skip decorating every annotated function instead of checking it.
FloatArray = np.ndarray[Any, np.dtype[np.float64]]

LandmarkNameString = str
"""A landmark's canonical lowercase name, or one of its aliases."""

LandmarkDefinitionString = str
"""The medical prose naming exactly one point on the body."""

RigidBodySegmentName = str
"""A segment's canonical lowercase name, or one of its aliases."""

LinkageNameString = str
"""A linkage's name - two segments joined at a shared landmark."""

ChainNameString = str
"""A kinematic chain's name - a sequence of linkages."""

SkeletonNameString = str
"""A whole skeleton's name."""

BlendshapeName = str
"""One ARKit face blendshape's name. Kept camelCase, unlike every other name here,
because ARKit's are case-sensitive and the face bypasses the lowercasing loader."""
