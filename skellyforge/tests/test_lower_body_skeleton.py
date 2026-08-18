from pathlib import Path

from skellyforge.skellymodels.standard_human.human_skeleton import HumanSkeleton

YAML_PATH = (
    Path(__file__).parent.parent
    / "skellymodels/standard_human/definitions/lower_body.yaml"
)


def test_lower_body_loads():
    skeleton = HumanSkeleton.from_yaml(YAML_PATH)
    assert skeleton.name == "lower_body"
    assert len(skeleton.segments) == 7       # pelvis + 2 thighs + 2 calves + 2 feet
    assert len(skeleton.linkages) == 6       # 2 hips + 2 knees + 2 ankles
    assert len(skeleton.chains) == 2         # left + right leg

    thigh = skeleton.segment("left_upper_leg")
    assert thigh.parent.name == "pelvis"
    assert thigh.origin_landmark.name == "left_hip"
    assert thigh.length == 429.0             # |(0, 429, 0)| — primary target along +Y

    # sidedness: the right side shares the pelvis's right_hip
    rthigh = skeleton.segment("right_upper_leg")
    assert rthigh.origin_landmark.name == "right_hip"

    leg = skeleton.chain("left_leg")
    assert [s.name for s in leg.segments] == [
        "pelvis", "left_upper_leg", "left_lower_leg", "left_foot",
    ]

    knee = next(l for l in skeleton.linkages if l.name == "left_knee")
    assert knee.parent_segment.name == "left_upper_leg"
    assert knee.child_segment.name == "left_lower_leg"
    assert knee.shared_landmark.name == "left_knee"


def test_sided_shares_local_geometry_and_mirrors_rest_direction():
    skeleton = HumanSkeleton.from_yaml(YAML_PATH)

    # full toe complexity: 21 landmarks on each foot
    left_foot = skeleton.segment("left_foot")
    right_foot = skeleton.segment("right_foot")
    assert len(left_foot.landmarks) == 21

    # rest_positions are SIDE-AGNOSTIC local geometry (left == right, no mirroring)
    left_mtp = next(l for l in left_foot.landmarks if l.name == "left_foot_hallux_mtp")
    right_mtp = next(l for l in right_foot.landmarks if l.name == "right_foot_hallux_mtp")
    assert left_mtp.rest_position == (-15.0, 45.0, 30.0)
    assert right_mtp.rest_position == left_mtp.rest_position  # identical local

    # the WORLD rest_direction mirrors Y for the right side
    left_twist = skeleton.segment("left_upper_leg").axes[1].rest_direction
    right_twist = skeleton.segment("right_upper_leg").axes[1].rest_direction
    assert left_twist == (0.0, 1.0, 0.0)
    assert right_twist == (0.0, -1.0, 0.0)
