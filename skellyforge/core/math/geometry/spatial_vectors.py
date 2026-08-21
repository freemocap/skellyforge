"""The affine algebra of 3D space: `Point`, `Displacement`, and `UnitVector`.

These three types wrap the same storage - a `(..., 3)` float64 array - but they are
deliberately NOT interchangeable, because they obey different algebras:

    Point        - Point        -> Displacement
    Point        + Displacement -> Point
    Displacement + Displacement -> Displacement
    Point        + Point        -> TypeError (meaningless)

A point is a location in space; a displacement is the difference between two locations.
Rotating a displacement just rotates it, while rotating a point requires centering it
first - which is the bug this distinction exists to catch. A `UnitVector` is a
displacement that is known to have unit length, so anything taking one may skip
normalizing and anything returning one has already done it.

The leading dimensions are free, so a single landmark is shape `(3,)`, a trajectory is
`(num_frames, 3)`, and a cloud of points per frame is `(num_frames, num_points, 3)`.
Batching is the normal case here, not a special one.

Every class validates in `__post_init__` and never mutates afterwards. Anything that
needs to coerce, convert, or normalize its input does so in a `from_*` classmethod,
*before* the frozen instance exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Self

import numpy as np

from skellyforge.type_overloads import FloatArray

MINIMUM_VECTOR_NORM: Final[float] = 1e-9
UNIT_LENGTH_TOLERANCE: Final[float] = 1e-8


@dataclass(frozen=True, slots=True, eq=False)
class Vector3Array:
    """Storage and validation shared by every `(..., 3)` spatial quantity.

    Never instantiate this directly - use `Point`, `Displacement`, or `UnitVector`,
    which are siblings rather than substitutes for one another.

    Attributes:
        array: `(..., 3)` float64 array of finite values. Must already be a float64
            ndarray; use `from_array` or `from_xyz` to build one from anything else.
    """

    array: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.array, np.ndarray):
            raise TypeError(
                f"{type(self).__name__} must wrap a numpy array - got "
                f"{type(self.array).__name__}. Use {type(self).__name__}.from_array()."
            )
        if self.array.dtype != np.float64:
            raise TypeError(
                f"{type(self).__name__} must wrap a float64 array - got dtype "
                f"{self.array.dtype}. Use {type(self).__name__}.from_array()."
            )
        if self.array.ndim < 1 or self.array.shape[-1] != 3:
            raise ValueError(
                f"{type(self).__name__} must have shape (..., 3) - got shape "
                f"{self.array.shape}"
            )
        if not np.all(np.isfinite(self.array)):
            raise ValueError(
                f"{type(self).__name__} contains non-finite (NaN/inf) values"
            )

    @classmethod
    def from_array(cls, *, values: object, name: str = "array") -> Self:
        """Coerce anything array-like into this type, converting before construction."""
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Cannot build a {cls.__name__} from `{name}` of type "
                f"{type(values).__name__}"
            ) from error
        return cls(array=array)

    @classmethod
    def from_xyz(cls, *, x: float, y: float, z: float) -> Self:
        """Build the single, unbatched `(3,)` case from three scalars."""
        return cls(array=np.array([x, y, z], dtype=np.float64))

    @property
    def x(self) -> FloatArray:
        """The `(...,)` x components."""
        return _as_float_array(values=self.array[..., 0])

    @property
    def y(self) -> FloatArray:
        """The `(...,)` y components."""
        return _as_float_array(values=self.array[..., 1])

    @property
    def z(self) -> FloatArray:
        """The `(...,)` z components."""
        return _as_float_array(values=self.array[..., 2])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The leading dimensions, i.e. everything but the trailing 3."""
        return self.array.shape[:-1]

    def expanded_at(self, *, axis: int) -> Self:
        """Insert a length-1 axis, for broadcasting one frame against many points."""
        return type(self)(array=np.expand_dims(self.array, axis=axis))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(batch_shape={self.batch_shape}, array={self.array})"


@dataclass(frozen=True, slots=True, eq=False)
class Point(Vector3Array):
    """A location in space, in whatever frame the surrounding code is working in.

    Points do not add. Subtracting two of them gives the `Displacement` between them,
    and adding a `Displacement` to one moves it somewhere else.
    """

    def __sub__(self, other: Point | Displacement) -> Displacement | Point:
        """`Point - Point` is the displacement between them; `Point - Displacement` moves it."""
        if isinstance(other, Point):
            return Displacement(array=self.array - other.array)
        if isinstance(other, Displacement):
            return Point(array=self.array - other.array)
        return NotImplemented

    def __add__(self, other: Displacement) -> Point:
        """Move this point along a displacement.

        Adding two points is meaningless and raises `TypeError`. The isinstance guard
        does that work rather than the type hint alone, so the algebra holds even when
        this module is imported somewhere beartype is not active.
        """
        if not isinstance(other, Displacement):
            return NotImplemented
        return Point(array=self.array + other.array)


@dataclass(frozen=True, slots=True, eq=False)
class Displacement(Vector3Array):
    """The difference between two points: a direction with a magnitude.

    Unlike a `Point`, a displacement has no location - it may be added, scaled, negated,
    and rotated on its own.
    """

    def __add__(self, other: Displacement) -> Displacement:
        if not isinstance(other, Displacement):
            return NotImplemented
        return Displacement(array=self.array + other.array)

    def __sub__(self, other: Displacement) -> Displacement:
        if not isinstance(other, Displacement):
            return NotImplemented
        return Displacement(array=self.array - other.array)

    def __neg__(self) -> Displacement:
        return Displacement(array=-self.array)

    def scaled_by(self, *, factors: float | FloatArray) -> Displacement:
        """Scale by a single number, or by one `(...,)` factor per batch element."""
        return Displacement(array=self.array * _as_broadcastable_factors(factors=factors))

    def dot(self, *, other: Displacement | UnitVector) -> FloatArray:
        """Row-wise dot product, returning shape `(...,)`."""
        return _row_wise_dot(first=self.array, second=other.array)

    def norm(self) -> FloatArray:
        """Row-wise length, returning shape `(...,)`."""
        return _as_float_array(values=np.linalg.norm(self.array, axis=-1))

    def normalized(self, *, description: str) -> UnitVector:
        """Scale to unit length, failing loudly on a degenerate (near-zero) displacement.

        Args:
            description: what this displacement is, quoted back in the error so that a
                degenerate frame names the points that caused it.
        """
        norms = np.linalg.norm(self.array, axis=-1, keepdims=True)
        if np.any(norms < MINIMUM_VECTOR_NORM):
            raise ValueError(
                f"Cannot normalize {description} - at least one displacement has norm "
                f"{float(norms.min()):.3e} < {MINIMUM_VECTOR_NORM:.1e} "
                "(the defining points are coincident)"
            )
        return UnitVector(array=self.array / norms)


@dataclass(frozen=True, slots=True, eq=False)
class UnitVector(Vector3Array):
    """A displacement that is known to be unit length - a pure direction.

    Construction verifies the length, so every `UnitVector` in the program is normalized
    by the time anything else sees it. Build one from a `Displacement` with
    `Displacement.normalized()` rather than constructing it directly.
    """

    def __post_init__(self) -> None:
        # Zero-argument super() is unavailable inside a slotted dataclass, because the
        # decorator rebuilds the class after the method's __class__ cell is bound.
        Vector3Array.__post_init__(self)
        norms = np.linalg.norm(self.array, axis=-1)
        worst_error = float(np.abs(norms - 1.0).max())
        if worst_error > UNIT_LENGTH_TOLERANCE:
            raise ValueError(
                f"UnitVector must have unit length - worst |‖v‖ - 1| = {worst_error:.3e} "
                f"> {UNIT_LENGTH_TOLERANCE:.1e}. Build one with Displacement.normalized()."
            )

    def __neg__(self) -> UnitVector:
        """The opposite direction, which is still unit length."""
        return UnitVector(array=-self.array)

    def dot(self, *, other: Displacement | UnitVector) -> FloatArray:
        """Row-wise dot product, returning shape `(...,)`."""
        return _row_wise_dot(first=self.array, second=other.array)

    def cross(self, *, other: UnitVector) -> UnitVector:
        """Cross product of two PERPENDICULAR unit vectors.

        The result is unit length only when the operands are perpendicular - its norm is
        the sine of the angle between them - so crossing two non-perpendicular directions
        fails loudly in `UnitVector.__post_init__` rather than quietly returning something
        shorter than unit length.
        """
        return UnitVector(array=_component_wise_cross(first=self.array, second=other.array))

    def scaled_by(self, *, factors: float | FloatArray) -> Displacement:
        """Give this direction a magnitude, producing a `Displacement`."""
        return Displacement(array=self.array * _as_broadcastable_factors(factors=factors))

    def as_displacement(self) -> Displacement:
        """This direction as a plain unit-length `Displacement`."""
        return Displacement(array=self.array)


def _as_float_array(*, values: object) -> FloatArray:
    """Keep per-batch results as arrays, so the unbatched case is a 0-d array.

    Numpy collapses a `(...,)`-shaped result to a bare `np.float64` scalar when there are
    no leading dimensions left. Keeping it a 0-d array means callers can index and
    broadcast it identically at every batch rank.
    """
    return np.asarray(values, dtype=np.float64)


def _as_broadcastable_factors(*, factors: float | FloatArray) -> float | FloatArray:
    """Give per-batch-element factors a trailing axis so they scale whole vectors."""
    if isinstance(factors, np.ndarray):
        return factors[..., np.newaxis]
    return factors


def _row_wise_dot(*, first: FloatArray, second: FloatArray) -> FloatArray:
    """Row-wise dot product of two `(..., 3)` arrays, returning shape `(...,)`.

    Unbatched `(3,)` inputs give a 0-d array rather than a numpy scalar, so that callers
    can index and broadcast the result the same way at every batch rank.
    """
    return _as_float_array(values=np.einsum("...i,...i->...", first, second))


def _component_wise_cross(*, first: FloatArray, second: FloatArray) -> FloatArray:
    """Cross product of two `(..., 3)` arrays, written out componentwise.

    Roughly 1.5x faster than `np.cross` over the large batches this runs on.
    """
    result = np.empty(shape=np.broadcast_shapes(first.shape, second.shape), dtype=np.float64)
    result[..., 0] = first[..., 1] * second[..., 2] - first[..., 2] * second[..., 1]
    result[..., 1] = first[..., 2] * second[..., 0] - first[..., 0] * second[..., 2]
    result[..., 2] = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    return result
