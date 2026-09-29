"""Real-data review preserves direct observations and refuses invented mappings."""

import numpy as np
import pytest
from scripts.recording_fit_geometry import rigid_target_checks
from scripts.generate_connected_fit_viewer import build_fixture
from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.chain.synthesis import synthesize_fitted_pose
from skellyforge.core.skeleton.pose.fit_connected_pose import LandmarkTarget
from scripts.generate_recording_fit_viewer import (
    TARGET_NAMES,
    target_sources,
    observed_targets,
)


def model():
    return dict(
        mappings=[dict(prefix="source_", entries={n: n for n in TARGET_NAMES})],
        tracker_keypoint_names=["source_" + n for n in TARGET_NAMES],
    )


def test_rigid_distance_check_crosses_joint_origin_without_freezing_chain():
    args, _ = build_fixture()
    _, _, points = synthesize_fitted_pose(**args)
    targets = {
        n: LandmarkTarget(points[n], 1.0)
        for n in ("left_acromion", "left_elbow", "left_wrist")
    }
    checks = rigid_target_checks(args["skeleton"], args["fit"], targets)
    pairs = {tuple(c["targets"]): c for c in checks}
    assert ("left_acromion", "left_wrist") not in pairs
    assert (
        pairs[("left_acromion", "left_elbow")]["minimum_possible_max_endpoint_error_mm"]
        < 1e-10
    )
    direction = points["left_elbow"].array - points["left_acromion"].array
    targets["left_acromion"] = LandmarkTarget(
        Point.from_array(
            values=points["left_acromion"].array
            + direction / np.linalg.norm(direction) * 40.0
        ),
        1.0,
    )
    checks = rigid_target_checks(args["skeleton"], args["fit"], targets)
    upper = next(c for c in checks if c["targets"] == ["left_acromion", "left_elbow"])
    assert upper["minimum_possible_max_endpoint_error_mm"] == pytest.approx(20.0)


def test_only_unique_direct_sources_are_accepted():
    saved = model()
    assert target_sources(saved) == {n: "source_" + n for n in TARGET_NAMES}
    saved["mappings"][0]["entries"]["nose"] = "left_ear"
    with pytest.raises(ValueError, match="unique"):
        target_sources(saved)
    saved = model()
    saved["mappings"][0]["entries"]["nose"] = ["left_ear", "right_ear"]
    with pytest.raises(ValueError, match="direct"):
        target_sources(saved)


def test_saved_landmarks_must_match_direct_keypoints_and_remain_unchanged():
    point = np.array([10.0, 20.0, 30.0])
    record = dict(points={"nose": point}, keypoints={"source_nose": point.copy()})
    targets = observed_targets(
        record, {"nose": "source_nose", "left_ear": "source_left_ear"}, 10.0
    )
    assert set(targets) == {"nose"}
    np.testing.assert_array_equal(targets["nose"].position.array, point)
    record["keypoints"]["source_nose"][0] += 1
    with pytest.raises(ValueError, match="disagrees"):
        observed_targets(record, {"nose": "source_nose"}, 10.0)
    np.testing.assert_array_equal(point, [10.0, 20.0, 30.0])
