"""The six subject-relative anatomical directions, and their canonical unit vectors.

Every coordinate-system convention is written as "which anatomical direction does each of
my +X/+Y/+Z axes point along". The directions are subject-relative, so RIGHT always means
the subject's right regardless of the coordinate frame.

The canonical frame these unit vectors are expressed in is the Blender frame
(+X right, +Y forward, +Z up), which is also the package's default convention. Expressing
every direction in one fixed frame is what makes any two conventions convertible through
that shared reference.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from skellyforge.type_overloads import FloatArray


class AnatomicalDirection(Enum):
    """One of the six subject-relative directions a coordinate axis can point along."""

    RIGHT = "right"
    LEFT = "left"
    FORWARD = "forward"
    BACKWARD = "backward"
    UP = "up"
    DOWN = "down"

    @property
    def unit_vector(self) -> FloatArray:
        """This direction as a unit vector in the canonical (Blender) frame."""
        return _UNIT_VECTORS_BY_DIRECTION[self]


def parse_anatomical_direction(*, label: str) -> AnatomicalDirection:
    """The direction named by label, failing loudly on anything unrecognized."""
    try:
        return AnatomicalDirection(label.lower())
    except ValueError as error:
        valid = ", ".join(sorted(direction.value for direction in AnatomicalDirection))
        raise ValueError(
            f"unknown anatomical direction {label!r} - expected one of {valid}"
        ) from error


def _as_readonly_unit_vector(values: list[float]) -> FloatArray:
    """A read-only float64 unit vector, so shared constants cannot be mutated in place."""
    array = np.array(values, dtype=np.float64)
    array.flags.writeable = False
    return array


_UNIT_VECTORS_BY_DIRECTION: dict[AnatomicalDirection, FloatArray] = {
    AnatomicalDirection.RIGHT: _as_readonly_unit_vector([1.0, 0.0, 0.0]),
    AnatomicalDirection.LEFT: _as_readonly_unit_vector([-1.0, 0.0, 0.0]),
    AnatomicalDirection.FORWARD: _as_readonly_unit_vector([0.0, 1.0, 0.0]),
    AnatomicalDirection.BACKWARD: _as_readonly_unit_vector([0.0, -1.0, 0.0]),
    AnatomicalDirection.UP: _as_readonly_unit_vector([0.0, 0.0, 1.0]),
    AnatomicalDirection.DOWN: _as_readonly_unit_vector([0.0, 0.0, -1.0]),
}
