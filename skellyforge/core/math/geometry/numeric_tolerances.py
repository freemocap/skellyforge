"""Numeric tolerances shared across the geometry and kinematics math.

These were once six independent magic numbers spread over four files. They are now
derived from one base - the smallest magnitude treated as meaningfully nonzero - so the
relationships between them are visible rather than accidental.
"""

from __future__ import annotations

from typing import Final

# The smallest magnitude this package treats as meaningfully nonzero.
MINIMUM_VECTOR_NORM: Final[float] = 1e-9

# A "unit" vector's length may be off by an order of magnitude more than the minimum.
UNIT_LENGTH_TOLERANCE: Final[float] = 10.0 * MINIMUM_VECTOR_NORM

# Quaternions carry four components, so roundoff accumulates faster: their degenerate
# threshold is a decade tighter, and their unit check is the base minimum.
MINIMUM_QUATERNION_NORM: Final[float] = 0.1 * MINIMUM_VECTOR_NORM
UNIT_QUATERNION_TOLERANCE: Final[float] = MINIMUM_VECTOR_NORM

# An orthonormal triad is "unit length" to the same slack as any unit vector.
ORTHONORMALITY_TOLERANCE: Final[float] = UNIT_LENGTH_TOLERANCE

# Gram-Schmidt loses stability as the defining vectors approach collinear, so the sine
# threshold that guards it is three orders of magnitude looser than the vector minimum.
MINIMUM_SINE_BETWEEN_DEFINING_VECTORS: Final[float] = 1000.0 * MINIMUM_VECTOR_NORM
