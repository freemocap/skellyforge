"""A fixed-capacity ring buffer of `(num_points, 3)` frames, for streaming point data.

One buffer serves all three ways this codebase consumes point data, because they differ
only in the length of the leading (time) axis that `Point` already carries:

    single frame (realtime)   `latest()`            -> Point of shape (3,)
    rolling window (stream)   `window_view()`       -> Point of shape (window, 3)
    full take (post hoc)      capacity >= num_frames, one window over everything

`window_by_name()` returns exactly the `Mapping[str, Point]` that
`calculate_orthonormal_basis` takes, so streaming and batch callers run the same code.

Layout
------
Storage is a single `(num_points, 2 * capacity, 3)` array and every appended frame is
written TWICE, at `cursor` and at `cursor + capacity`. That redundancy is what makes the
newest `capacity` samples always a CONTIGUOUS slice, so a window is a plain view: no
concatenate, no roll, no copy, and no per-frame cost that scales with the window length.
Time is the middle axis so that each point's window is contiguous in memory, which keeps
downstream einsum calls on their fast path. The cost is 2x memory - at capacity 1000 with
50 points, 2.4 MB.

Aliasing
--------
`window_view` and `latest` return views into storage that the NEXT `append` overwrites.
A retained view does not slide along with the buffer - it goes INCOHERENT, because the
incoming sample lands in the slot that view's oldest entry occupies. A view read when the
window held `[0, 1, 2]` reads back as `[99, 1, 2]` after appending `99`, which is neither
the old window nor the new one (`[1, 2, 99]`).

So: read the window AFTER each append and discard it before the next, which is what a
compute-then-discard frame loop does naturally. Use `window_copy` for anything that must
outlive the next append.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.type_overloads import FloatArray

NUMBER_OF_SPATIAL_DIMENSIONS: Final[int] = 3
_IMMUTABLE_FIELD_NAMES: Final[frozenset[str]] = frozenset({"point_names", "capacity"})


@dataclass(slots=True, eq=False)
class PointRingBuffer:
    """Fixed-capacity storage for a stream of named point positions.

    Appends are O(1) regardless of capacity, and windows are zero-copy views.

    This is the one MUTABLE type in the geometry module, and deliberately so: a ring
    buffer's whole job is writing into storage it already owns and advancing a cursor.
    Freezing it would mean allocating a fresh buffer per frame, which is exactly the
    O(window)-per-frame cost the layout exists to avoid. It is a container, not a value -
    the values it hands out (`Point`) are frozen as ever.

    Because it is mutable, `__post_init__` both validates AND allocates derived state,
    unlike the frozen value types where `__post_init__` only ever validates.

    Attributes:
        point_names: the points this buffer tracks, in the row order that `append`
            expects. Must be non-empty and free of duplicates.
        capacity: how many frames the rolling window holds. For a whole-take batch, set
            this to the number of frames.
    """

    point_names: tuple[str, ...]
    capacity: int
    _index_by_name: dict[str, int] = field(init=False, repr=False)
    _storage: FloatArray = field(init=False, repr=False)
    _cursor: int = field(init=False, repr=False, default=0)
    _number_of_appended_frames: int = field(init=False, repr=False, default=0)

    def __setattr__(self, name: str, value: object) -> None:
        """Refuse to rewrite the two fields the allocated storage was sized from.

        The buffer is mutable because a ring buffer's job is writing into storage it
        already owns - but `point_names` and `capacity` are not the mutable part. Changing
        either after `__post_init__` would leave the cursor, the index and the array
        describing three different buffers, and every read afterwards would be quietly
        wrong rather than loudly broken.
        """
        if name in _IMMUTABLE_FIELD_NAMES and hasattr(self, "_storage"):
            raise AttributeError(
                f"PointRingBuffer.{name} is fixed once the buffer has allocated its "
                "storage - build a new buffer instead of resizing this one"
            )
        # `object.__setattr__` rather than a zero-argument `super()`: the dataclass
        # decorator rebuilds a slotted class after this method's `__class__` cell is
        # bound, so the zero-argument form raises here.
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"capacity must be at least 1 - got {self.capacity}")
        if len(self.point_names) == 0:
            raise ValueError("A PointRingBuffer needs at least one point name")
        if len(set(self.point_names)) != len(self.point_names):
            duplicates = sorted(
                {name for name in self.point_names if self.point_names.count(name) > 1}
            )
            raise ValueError(f"point_names must be unique - repeated: {duplicates}")

        self._index_by_name = {name: index for index, name in enumerate(self.point_names)}
        self._storage = np.zeros(
            shape=(len(self.point_names), 2 * self.capacity, NUMBER_OF_SPATIAL_DIMENSIONS),
            dtype=np.float64,
        )

    @classmethod
    def for_point_names(cls, *, point_names: Sequence[str], capacity: int) -> PointRingBuffer:
        """Build a buffer from any sequence of names, which is stored as a tuple."""
        return cls(point_names=tuple(point_names), capacity=capacity)

    @property
    def number_of_appended_frames(self) -> int:
        """How many frames have ever been appended, including ones since overwritten."""
        return self._number_of_appended_frames

    @property
    def window_length(self) -> int:
        """How many frames the current window holds - `capacity` once filled."""
        return min(self._number_of_appended_frames, self.capacity)

    @property
    def is_full(self) -> bool:
        """Whether the window has reached capacity and is now overwriting old frames."""
        return self._number_of_appended_frames >= self.capacity

    def index_of(self, *, name: str) -> int:
        """The row index of a point name, for callers assembling frames to append."""
        if name not in self._index_by_name:
            raise KeyError(
                f"Point `{name}` is not tracked by this buffer - it holds "
                f"{list(self.point_names)}"
            )
        return self._index_by_name[name]

    def append(self, *, positions: Point) -> None:
        """Append one frame of positions, ordered to match `point_names`.

        Taking the whole frame as one `(num_points, 3)` array rather than a mapping keeps
        the append a single vectorized store, with no per-point Python loop in the
        streaming hot path.

        Args:
            positions: `(num_points, 3)` locations for this frame, row `i` being
                `point_names[i]`.
        """
        expected_shape = (len(self.point_names), NUMBER_OF_SPATIAL_DIMENSIONS)
        if positions.array.shape != expected_shape:
            raise ValueError(
                f"A frame must have shape {expected_shape} to match point_names - got "
                f"{positions.array.shape}"
            )
        self._storage[:, self._cursor, :] = positions.array
        self._storage[:, self._cursor + self.capacity, :] = positions.array
        self._cursor = (self._cursor + 1) % self.capacity
        self._number_of_appended_frames += 1

    def window_view(self, *, name: str) -> Point:
        """A zero-copy `(window_length, 3)` view of one point's window, oldest first.

        Valid only until the next `append`, which overwrites the memory it points at.
        """
        index = self.index_of(name=name)
        start = self._cursor + self.capacity - self._raise_if_empty()
        stop = self._cursor + self.capacity
        return Point.from_prevalidated_array(array=self._storage[index, start:stop, :])

    def window_copy(self, *, name: str) -> Point:
        """A `(window_length, 3)` copy of one point's window, safe to keep."""
        return Point.from_prevalidated_array(array=self.window_view(name=name).array.copy())

    def window_by_name(self) -> dict[str, Point]:
        """Every point's window as zero-copy views, keyed by name.

        This is the mapping `calculate_orthonormal_basis` takes, so a streaming caller
        feeds it the same way a batch caller does.
        """
        return {name: self.window_view(name=name) for name in self.point_names}

    def latest(self, *, name: str) -> Point:
        """A zero-copy `(3,)` view of one point's most recent sample."""
        self._raise_if_empty()
        index = self.index_of(name=name)
        return Point.from_prevalidated_array(
            array=self._storage[index, self._cursor + self.capacity - 1, :]
        )

    def latest_by_name(self) -> dict[str, Point]:
        """Every point's most recent sample as zero-copy views, keyed by name."""
        return {name: self.latest(name=name) for name in self.point_names}

    def _raise_if_empty(self) -> int:
        """The current window length, refusing to hand out a window of nothing."""
        window_length = self.window_length
        if window_length == 0:
            raise ValueError(
                "This PointRingBuffer is empty - append at least one frame before "
                "reading a window"
            )
        return window_length

    def __repr__(self) -> str:
        return (
            f"PointRingBuffer(point_names={list(self.point_names)}, "
            f"capacity={self.capacity}, window_length={self.window_length}, "
            f"number_of_appended_frames={self._number_of_appended_frames})"
        )
