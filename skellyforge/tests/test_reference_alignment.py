"""The shared reference-selection policy preserves supplied geometry when required."""

import numpy as np
import pytest

from skellyforge.core.biomechanics.body_alignment import (
    BodyAlignmentConfig,
    BodyReferenceTrack,
)
from skellyforge.core.biomechanics.ground_alignment import GroundAlignmentConfig
from skellyforge.core.biomechanics.reference_alignment import (
    ReferenceAlignmentOutcome,
    ReferenceAlignmentRequest,
    estimate_reference_alignment,
)


@pytest.mark.parametrize(
    ("enabled", "ground", "outcome"),
    [
        (False, False, ReferenceAlignmentOutcome.DISABLED),
        (True, True, ReferenceAlignmentOutcome.EXPLICIT_GROUND),
        (False, True, ReferenceAlignmentOutcome.EXPLICIT_GROUND),
        (True, False, ReferenceAlignmentOutcome.BODY_REFERENCE),
    ],
)
def test_reference_policy(
    *, enabled: bool, ground: bool, outcome: ReferenceAlignmentOutcome
) -> None:
    track = BodyReferenceTrack(
        segment_name="skull",
        timestamps_seconds=np.linspace(0, 1, 11),
        world_from_body=np.tile(np.eye(3), (11, 1, 1)),
        origins=np.full((11, 3), 50.0),
        quality=np.ones(11),
    )
    result = estimate_reference_alignment(
        request=ReferenceAlignmentRequest(
            enabled=enabled,
            has_explicit_ground=ground,
            body_tracks=(track,),
            foot_contacts=(),
            body_config=BodyAlignmentConfig(),
            ground_config=GroundAlignmentConfig(
                maximum_speed=10.0, maximum_plane_distance=2.0, minimum_spread=5.0
            ),
        )
    )
    assert result.outcome is outcome
    expected = -50.0 if outcome is ReferenceAlignmentOutcome.BODY_REFERENCE else 0.0
    np.testing.assert_allclose(result.transform.translation.array, np.full(3, expected))
