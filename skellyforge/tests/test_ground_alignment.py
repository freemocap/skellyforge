"""Foot-support fitting must preserve scene geometry and reject unsupported floors."""

import numpy as np

from skellyforge.core.biomechanics.body_alignment import (
    AlignmentOutcome,
    BodyAlignmentResult,
)
from skellyforge.core.biomechanics.ground_alignment import (
    FootContactTrack,
    GroundAlignmentConfig,
    GroundAlignmentOutcome,
    estimate_ground_alignment,
)
from skellyforge.core.math.geometry.transform_math import Transform


def body_reference() -> BodyAlignmentResult:
    return BodyAlignmentResult(
        outcome=AlignmentOutcome.BODY_REFERENCE,
        transform=Transform.identity(),
        anchor_segment="skull",
        support_seconds=1.0,
        mean_quality=1.0,
        maximum_deviation_degrees=0.0,
    )


def contact_tracks(*, positions: list[list[float]]) -> tuple[FootContactTrack, ...]:
    return tuple(
        FootContactTrack(
            landmark_name=f"contact_{index}",
            timestamps_seconds=np.linspace(0, 1, 21),
            positions=np.tile(np.asarray(position, dtype=np.float64), (21, 1)),
            quality=np.ones(21),
        )
        for index, position in enumerate(positions)
    )


def ground_config() -> GroundAlignmentConfig:
    return GroundAlignmentConfig(
        maximum_speed=10.0, maximum_plane_distance=2.0, minimum_spread=5.0
    )


def test_tilted_support_recovers_plane_and_preserves_distances() -> None:
    contacts = contact_tracks(
        positions=[
            [-100, -100, 280],
            [100, -100, 320],
            [-100, 100, 280],
            [100, 100, 320],
        ]
    )
    result = estimate_ground_alignment(
        contacts=contacts, body_reference=body_reference(), config=ground_config()
    )
    assert result.outcome is GroundAlignmentOutcome.FOOT_SUPPORT
    assert result.transform is not None
    points = np.stack([track.positions[0] for track in contacts])
    rotation = result.transform.rotation.to_rotation_matrix()
    transformed = points @ rotation.T + result.transform.translation.array
    np.testing.assert_allclose(transformed[:, 2], 0, atol=1e-10)
    np.testing.assert_allclose(
        np.linalg.norm(transformed[1] - transformed[0]),
        np.linalg.norm(points[1] - points[0]),
    )
    # Whole-scene extrinsic composition must preserve the original camera coordinates.
    camera_rotation = np.eye(3)
    camera_translation = np.array([10.0, 20.0, 30.0])
    adjusted_rotation = camera_rotation @ rotation.T
    adjusted_translation = (
        camera_translation - adjusted_rotation @ result.transform.translation.array
    )
    np.testing.assert_allclose(
        transformed @ adjusted_rotation.T + adjusted_translation,
        points + camera_translation,
    )


def test_collinear_contacts_do_not_define_floor() -> None:
    result = estimate_ground_alignment(
        contacts=contact_tracks(positions=[[0, 0, 0], [100, 0, 0], [200, 0, 0]]),
        body_reference=body_reference(),
        config=ground_config(),
    )
    assert result.transform is None


def test_isolated_elevated_contact_does_not_tilt_plane() -> None:
    result = estimate_ground_alignment(
        contacts=contact_tracks(
            positions=[
                [-100, -100, 0],
                [100, -100, 0],
                [-100, 100, 0],
                [100, 100, 0],
                [0, 0, 500],
            ]
        ),
        body_reference=body_reference(),
        config=ground_config(),
    )
    assert result.outcome is GroundAlignmentOutcome.FOOT_SUPPORT
    assert result.transform is not None
    np.testing.assert_allclose(
        result.transform.rotation.to_rotation_matrix(), np.eye(3), atol=1e-10
    )


def test_motion_is_not_contact() -> None:
    contacts = contact_tracks(
        positions=[[-100, -100, 0], [100, -100, 0], [-100, 100, 0], [100, 100, 0]]
    )
    for track in contacts:
        track.positions[:, 0] += np.linspace(0, 1000, 21)
    result = estimate_ground_alignment(
        contacts=contacts, body_reference=body_reference(), config=ground_config()
    )
    assert result.outcome is GroundAlignmentOutcome.INSUFFICIENT_CONTACTS
