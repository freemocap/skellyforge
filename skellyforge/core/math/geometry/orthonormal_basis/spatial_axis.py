"""The six signed cartesian directions, and the axis arithmetic built on them."""

from __future__ import annotations

from enum import Enum

AXIS_NAMES: tuple[str, str, str] = ("x", "y", "z")


class SpatialAxis(Enum):
    """A cartesian axis together with a direction along it, e.g. `NEGATIVE_Y`.

    Each member carries `index`, its position in a (..., 3) vector, and `sign`, its
    direction along that axis. Code that only cares which of the three cartesian axes is
    meant reads `index`; code that cares which way it points reads `sign`. Two members
    naming the same axis (`X` and `NEGATIVE_X`) share an `index` but are NOT identical,
    so compare `index` rather than the members themselves when asking "same axis?".

    Used as a defining direction, `SpatialAxis.NEGATIVE_Y` means "the vector from the
    origin point toward the named point IS the negative y direction", so the frame's y
    basis vector points the opposite way from that named point.
    """

    X = (0, 1)
    Y = (1, 1)
    Z = (2, 1)

    NEGATIVE_X = (0, -1)
    NEGATIVE_Y = (1, -1)
    NEGATIVE_Z = (2, -1)

    def __init__(self, index: int, sign: int) -> None:
        self.index: int = index
        self.sign: int = sign

    def __str__(self) -> str:
        return f"{'+' if self.sign > 0 else '-'}{AXIS_NAMES[self.index]}"

    def cyclic_sign_toward(self, other: SpatialAxis) -> int:
        """+1 when the axes are a cyclic pair (x, y), (y, z), (z, x), else -1.

        This is the sign that makes `sign * cross(self_vector, other_vector)` point along
        the remaining axis in the right-handed sense. Only the axes are consulted, not
        their directions - the callers fold their own signs into the vectors themselves.
        """
        if self.index == other.index:
            raise ValueError(f"Cannot order an axis against itself - both axes are {self}")
        return 1 if (self.index + 1) % 3 == other.index else -1

    def remaining_axis(self, other: SpatialAxis) -> SpatialAxis:
        """The positive axis that is neither `self`'s axis nor `other`'s."""
        if self.index == other.index:
            raise ValueError(f"Cannot find remaining axis - both axes are {self}")
        return SpatialAxis((3 - self.index - other.index, 1))
