"""The Kabsch degeneracy guard.

``align_point_sets_kabsch`` recovers a rotation only when the point set spans a
plane. A collinear set leaves the rotation about that line unconstrained, so the
solver must reject it (raise) instead of silently returning whatever
least-squares yields. This is the guard the pelvis root needs: its three
tracker-hydrated landmarks (hips_center = the midpoint of the two hips) are
exactly collinear every frame.
"""

import numpy as np
import pytest

from skellyforge.kinematics.coordinate_frame_ops import align_point_sets_kabsch


def _rotation_about_axis(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return c * np.eye(3) + s * cross + (1 - c) * np.outer(axis, axis)


def test_kabsch_recovers_rotation_from_planar_points():
    """Three non-collinear points (a plane) determine a unique rotation."""
    reference = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 80.0, 0.0]])
    rotation = _rotation_about_axis([0.2, 0.5, 0.84], np.deg2rad(41.0))
    live = reference @ rotation.T
    recovered = align_point_sets_kabsch(reference, live)
    assert np.allclose(recovered, rotation, atol=1e-9)


def test_kabsch_raises_on_exactly_collinear_points():
    """A collinear reference (and its rotated live image) is under-determined."""
    reference = np.array([[0.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 100.0, 0.0]])
    live = reference @ _rotation_about_axis([1.0, 0.0, 0.0], np.deg2rad(37.0)).T
    with pytest.raises(ValueError, match="collinear|degenerate"):
        align_point_sets_kabsch(reference, live)


def test_kabsch_raises_on_collinear_points_with_noise():
    """Near-collinear (the audit's 0.5 mm-noise case) is still rejected: the
    rotation about the near-line is still unconstrained."""
    reference = np.array([[0.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 100.0, 0.0]])
    # tiny perpendicular perturbation — far below the point spread
    reference = reference + np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    live = reference @ _rotation_about_axis([1.0, 0.0, 0.0], np.deg2rad(37.0)).T
    with pytest.raises(ValueError, match="collinear|degenerate"):
        align_point_sets_kabsch(reference, live)


def test_pelvis_style_midpoint_triple_is_rejected():
    """hips_center as the midpoint of the two hips is exactly collinear."""
    left_hip = np.array([0.0, 90.0, 0.0])
    right_hip = np.array([0.0, -90.0, 0.0])
    hips_center = 0.5 * (left_hip + right_hip)
    reference = np.stack([hips_center, left_hip, right_hip])
    live = reference @ _rotation_about_axis([0.0, 1.0, 0.0], np.deg2rad(20.0)).T
    with pytest.raises(ValueError, match="collinear|degenerate"):
        align_point_sets_kabsch(reference, live)
