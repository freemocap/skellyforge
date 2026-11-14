import numpy as np
import pytest

from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.definitions.human_skeleton.hand import human_hand_model


@pytest.mark.parametrize("side", ["left", "right"])
def test_standalone_hand_preserves_the_human_hand_rest_geometry(side):
    human = SkeletonDefinition.from_default_yaml()
    rest = RestPose.from_default_yaml(skeleton=human)
    hand, pose = human_hand_model(human=human, rest=rest, side=side)
    assert len(hand.segments) == 20
    assert len(hand.joints) == 19
    assert pose.root_segment_name == f"{side}_carpals"
    wrist = rest.segment_origins[pose.root_segment_name].array
    for name, landmark in hand.landmarks.items():
        assert landmark is human.landmarks[name]
        np.testing.assert_allclose(pose.landmark_positions[name].array,
                                   rest.landmark_positions[name].array - wrist, atol=1e-12)
