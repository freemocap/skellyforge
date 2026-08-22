"""Tests for the streaming point ring buffer."""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.point_ring_buffer import PointRingBuffer
from skellyforge.core.math.geometry.spatial_vectors import Point


def _frame(*, value: float, number_of_points: int = 2) -> Point:
    return Point.from_array(values=np.full(shape=(number_of_points, 3), fill_value=value))


def _buffer(*, capacity: int = 3) -> PointRingBuffer:
    return PointRingBuffer(point_names=("a", "b"), capacity=capacity)


def test_window_fills_then_slides_oldest_first() -> None:
    buffer = _buffer(capacity=3)
    expected_windows = [[0.0], [0.0, 1.0], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
    for value, expected in enumerate(expected_windows):
        buffer.append(positions=_frame(value=float(value)))
        np.testing.assert_allclose(buffer.window_view(name="a").array[:, 0], expected)


def test_window_is_a_contiguous_zero_copy_view() -> None:
    buffer = _buffer(capacity=3)
    for value in range(5):
        buffer.append(positions=_frame(value=float(value)))
    window = buffer.window_view(name="a").array
    assert window.base is not None, "window should be a view, not a copy"
    assert window.flags["C_CONTIGUOUS"], "window must stay contiguous for fast einsum"


def test_a_view_aliases_storage_but_a_copy_does_not() -> None:
    buffer = _buffer(capacity=3)
    for value in range(3):
        buffer.append(positions=_frame(value=float(value)))
    view = buffer.window_view(name="a")
    copy = buffer.window_copy(name="a")
    np.testing.assert_allclose(view.array[:, 0], [0.0, 1.0, 2.0])

    buffer.append(positions=_frame(value=99.0))

    # A retained view does not slide - it goes INCOHERENT. The new sample lands in the
    # slot the view's oldest entry occupied, so the retained view is neither the window it
    # was nor the window it is now. Re-read the window after every append.
    np.testing.assert_allclose(view.array[:, 0], [99.0, 1.0, 2.0])
    np.testing.assert_allclose(buffer.window_view(name="a").array[:, 0], [1.0, 2.0, 99.0])
    # A copy is unaffected by later appends.
    np.testing.assert_allclose(copy.array[:, 0], [0.0, 1.0, 2.0])


def test_latest_returns_the_newest_sample() -> None:
    buffer = _buffer(capacity=3)
    for value in range(7):
        buffer.append(positions=_frame(value=float(value)))
    assert float(buffer.latest(name="a").x) == 6.0
    assert buffer.latest(name="a").batch_shape == ()


def test_window_and_latest_feed_the_geometry_solvers_unchanged() -> None:
    buffer = _buffer(capacity=4)
    for value in range(6):
        buffer.append(positions=_frame(value=float(value)))
    windows = buffer.window_by_name()
    assert set(windows) == {"a", "b"}
    assert windows["a"].batch_shape == (4,)
    assert buffer.latest_by_name()["b"].batch_shape == ()


def test_window_length_and_fullness_track_appends() -> None:
    buffer = _buffer(capacity=3)
    assert buffer.window_length == 0 and not buffer.is_full
    buffer.append(positions=_frame(value=0.0))
    assert buffer.window_length == 1 and not buffer.is_full
    for value in range(2):
        buffer.append(positions=_frame(value=float(value)))
    assert buffer.window_length == 3 and buffer.is_full
    buffer.append(positions=_frame(value=9.0))
    assert buffer.window_length == 3
    assert buffer.number_of_appended_frames == 4


def test_capacity_one_keeps_only_the_newest_frame() -> None:
    buffer = _buffer(capacity=1)
    for value in range(4):
        buffer.append(positions=_frame(value=float(value)))
    np.testing.assert_allclose(buffer.window_view(name="a").array[:, 0], [3.0])


def test_reading_an_empty_buffer_raises() -> None:
    buffer = _buffer(capacity=3)
    with pytest.raises(ValueError, match="empty"):
        buffer.window_view(name="a")
    with pytest.raises(ValueError, match="empty"):
        buffer.latest(name="a")


def test_unknown_point_name_raises() -> None:
    buffer = _buffer(capacity=3)
    buffer.append(positions=_frame(value=0.0))
    with pytest.raises(KeyError, match="not tracked"):
        buffer.window_view(name="nope")


def test_wrong_frame_shape_raises() -> None:
    buffer = _buffer(capacity=3)
    with pytest.raises(ValueError, match="must have shape"):
        buffer.append(positions=_frame(value=0.0, number_of_points=5))


def test_construction_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="capacity must be at least 1"):
        PointRingBuffer(point_names=("a",), capacity=0)
    with pytest.raises(ValueError, match="at least one point name"):
        PointRingBuffer(point_names=(), capacity=3)
    with pytest.raises(ValueError, match="unique"):
        PointRingBuffer(point_names=("a", "a"), capacity=3)


def test_buffer_is_mutable_by_design_but_hands_out_frozen_values() -> None:
    # The buffer is the one mutable type in the geometry module; the Points it yields are
    # frozen like every other value.
    buffer = _buffer(capacity=2)
    buffer.append(positions=_frame(value=1.0))
    with pytest.raises(Exception):
        buffer.latest(name="a").array = np.zeros(3)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("capacity", 5), ("point_names", ("a", "b", "c"))],
)
def test_the_fields_the_storage_was_sized_from_cannot_be_rewritten(
    field_name: str, value: object
) -> None:
    """Being a container does not make every field of it fair game.

    Rebinding either of these after construction would leave the cursor, the name index
    and the allocated array describing three different buffers, and every read afterwards
    would be quietly wrong rather than loudly broken.
    """
    buffer = _buffer(capacity=2)
    buffer.append(positions=_frame(value=1.0))
    with pytest.raises(AttributeError, match="fixed once the buffer"):
        setattr(buffer, field_name, value)


def test_for_point_names_accepts_any_sequence() -> None:
    buffer = PointRingBuffer.for_point_names(point_names=["a", "b"], capacity=2)
    assert buffer.point_names == ("a", "b")


def test_index_of_matches_point_name_order() -> None:
    buffer = _buffer(capacity=3)
    assert buffer.point_names == ("a", "b")
    assert buffer.index_of(name="b") == 1


def test_append_cost_does_not_grow_with_capacity() -> None:
    # The whole point of the doubled-buffer layout: appends are O(1), not O(window).
    import timeit

    def append_time(*, capacity: int) -> float:
        buffer = PointRingBuffer(point_names=("a", "b"), capacity=capacity)
        frame = _frame(value=1.0)
        for _ in range(capacity):
            buffer.append(positions=frame)
        return timeit.timeit(lambda: buffer.append(positions=frame), number=2000)

    small = append_time(capacity=10)
    large = append_time(capacity=10_000)
    assert large < small * 5, (
        f"append should not scale with capacity - 10 frames took {small:.4f}s, "
        f"10000 frames took {large:.4f}s"
    )
