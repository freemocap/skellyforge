"""Segment-based center of mass (de Leva 1996) on the standard-human segment layer."""

import numpy as np
import pytest

from skellyforge.kinematics.inertial.anthropometric_parameters import segment_inertial_parameters
from skellyforge.kinematics.inertial.center_of_mass import (
    CoMConfidence,
    calculate_center_of_mass,
)
from skellyforge.kinematics.tpose import build_standard_human_tpose
from skellyforge.skellymodels.standard_human.human_skeleton import HumanSkeleton


@pytest.fixture(scope="module")
def skeleton_and_tpose():
    skeleton = HumanSkeleton.standard_human()
    tpose = build_standard_human_tpose(skeleton)
    return skeleton, tpose


def test_full_observation_sums_mass_to_one(skeleton_and_tpose):
    skeleton, tpose = skeleton_and_tpose
    result = calculate_center_of_mass(skeleton, dict(tpose.landmarks))
    assert sum(result.segment_masses.values()) == pytest.approx(1.0, abs=1e-3)
    assert result.directly_observed_mass == pytest.approx(1.0, abs=1e-3)
    assert result.confidence >= CoMConfidence.high
    assert result.total_body_com.shape == (3,)
    assert np.all(np.isfinite(result.total_body_com))


def test_thigh_com_is_com_fraction_along_the_segment(skeleton_and_tpose):
    skeleton, tpose = skeleton_and_tpose
    result = calculate_center_of_mass(skeleton, dict(tpose.landmarks))
    thigh = skeleton.segment("upper_leg.L")  # de Leva thigh -> upper_leg segment
    com_fraction = segment_inertial_parameters("mean")["thigh"].com_fraction
    hip = tpose.landmarks[thigh.origin_landmark.name]
    knee = tpose.landmarks[thigh.primary_axis.target_landmark]
    expected = hip + (knee - hip) * com_fraction
    assert np.allclose(result.segment_coms["left_thigh"], expected)


def test_occluded_foot_rolls_mass_up_the_chain(skeleton_and_tpose):
    skeleton, tpose = skeleton_and_tpose
    landmarks = dict(tpose.landmarks)
    landmarks.pop("left_foot_ball")  # occlude the left foot (ankle -> foot_ball span broken)
    result = calculate_center_of_mass(skeleton, landmarks)
    assert "left_foot" not in result.segment_coms
    foot_mass = segment_inertial_parameters("mean")["foot"].mass_fraction
    assert result.directly_observed_mass == pytest.approx(1.0 - foot_mass, abs=1e-3)
    assert result.directly_observed_mass < 1.0
