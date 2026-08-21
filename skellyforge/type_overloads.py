from typing import Any

import numpy as np

# Written as an explicit ``np.ndarray[..., np.dtype[...]]`` subscription rather than
# ``numpy.typing.NDArray[np.float64]``: under numpy >= 2.5 ``NDArray`` is a PEP-695
# alias whose ``ScalarT`` typevar beartype cannot reduce to a concrete dtype, which
# makes beartype skip decorating every annotated function instead of checking it.
FloatArray = np.ndarray[Any, np.dtype[np.float64]]
LandmarkNameString = str
LandmarkDefinitionString = str
RigidBodySegmentName = str
