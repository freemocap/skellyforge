"""Chirality of a coordinate system.

Right-handed is the assumption everywhere in this codebase. Left-handed frames are
buildable so that geometry authored here can be converted into a left-handed external
convention later, but every construction of one announces itself loudly.
"""

from __future__ import annotations

from enum import Enum


class LeftHandedCoordinateSystemWarning(UserWarning):
    """Emitted whenever a left-handed coordinate system is requested."""
    # print warning saying that the system is Left Handed, which may interfere with down stream operations that generally assume right handedness


class Handedness(Enum):
    """Chirality of a coordinate system, valued by the determinant it must have."""

    RIGHT_HANDED = 1
    LEFT_HANDED = -1

    @property
    def expected_determinant(self) -> float:
        """+1.0 for a right-handed frame, -1.0 for a left-handed one."""
        return float(self.value)
