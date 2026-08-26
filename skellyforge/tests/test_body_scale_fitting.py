"""Fitting the dimensionless template to a real body, including when half of it is hidden.

Every test here works on the shipped standard human, observed as a real subject: the rest
pose scaled to `SUBJECT_HEIGHT_MM` millimetres. That is the whole contract - a template
authored as fractions of body height, driven by observations in millimetres - so a test
that stayed in template units would not exercise it.
"""

from __future__ import annotations

import numpy as np
import pytest

from skellyforge.core.math.geometry.spatial_vectors import Point
from skellyforge.core.skeleton.pose.body_scale_fitting import (
    DEFAULT_SCALE_WINDOW_FRAMES,
    InsufficientScaleEvidence,
    StreamingBodyScaleFitter,
    body_scale_voting_segment_names,
    fit_body_scale,
)
from skellyforge.core.skeleton.pose.hydration import hydrate_skeleton
from skellyforge.core.skeleton.pose.rest_pose import RestPose
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.skeleton_pose import SkeletonPose

SUBJECT_HEIGHT_MM: float = 1700.0

# The landmarks a desk hides: everything from the knees down. The hips stay visible, which
# is exactly the situation of somebody seated at a keyboard.
_BELOW_THE_DESK = (
    "knee",
    "ankle",
    "ball",
    "calcaneus",
    "toe_tip",
)


def _skeleton() -> SkeletonDefinition:
    return SkeletonDefinition.from_default_yaml()


def _observed_subject(
    *, skeleton: SkeletonDefinition, height_mm: float = SUBJECT_HEIGHT_MM
) -> dict[str, Point]:
    """A real subject standing in the T-pose: the template, scaled to millimetres."""
    rest_pose = RestPose.from_default_yaml(skeleton=skeleton)
    return {
        name: Point.from_array(values=height_mm * position.array)
        for name, position in rest_pose.landmark_positions.items()
    }


def _hydrate(
    *, skeleton: SkeletonDefinition, observed: dict[str, Point]
) -> SkeletonPose:
    return hydrate_skeleton(skeleton=skeleton, observed=observed, require_all=False)


def _all_landmarks_measured(skeleton: SkeletonDefinition) -> frozenset[str]:
    return frozenset(skeleton.landmarks)


def _fit_from_one_pose(
    *, skeleton: SkeletonDefinition, pose: SkeletonPose, measured: frozenset[str]
):
    fitter = StreamingBodyScaleFitter(
        skeleton=skeleton,
        voting_segment_names=body_scale_voting_segment_names(
            skeleton=skeleton, measured_landmark_names=measured
        ),
    )
    fitter.observe_pose(pose=pose)
    return fitter.current_fit()


# ── every segment reads the same body ──────────────────────────────────────


def test_every_hydrated_segment_reads_the_subjects_height() -> None:
    """The premise the whole fit rests on: `d / p` is the same number for every segment.

    This is also the regression test for the similarity fit. A rigid-fit segment solved
    without a scale puts its origin on its observed centroid, and the thorax then reads a
    height of about 650mm instead of 1700mm.
    """
    skeleton = _skeleton()
    pose = _hydrate(skeleton=skeleton, observed=_observed_subject(skeleton=skeleton))

    assert pose.segment_poses, "the full T-pose must hydrate segments"
    for name, segment_pose in pose.segment_poses.items():
        assert segment_pose.body_scale_estimate == pytest.approx(
            SUBJECT_HEIGHT_MM, rel=1e-6
        ), f"segment {name!r} reads a different body than the rest of the skeleton"


def test_rigid_fit_segments_put_their_origin_on_their_own_origin_landmark() -> None:
    """The thorax, pelvis, skull and carpals - the four that solve by rigid fit."""
    skeleton = _skeleton()
    observed = _observed_subject(skeleton=skeleton)
    pose = _hydrate(skeleton=skeleton, observed=observed)

    rigid_fit_names = [
        name for name, segment in skeleton.segments.items() if segment.supports_rigid_fit
    ]
    assert rigid_fit_names, "the standard human must have segments that rigid-fit"
    for name in rigid_fit_names:
        segment_pose = pose.segment_poses[name]
        expected = observed[
            skeleton.segments[name].frame_definition.origin_point_name
        ].array
        error_mm = float(np.linalg.norm(segment_pose.origin.array - expected))
        assert error_mm < 1e-6, f"segment {name!r} origin is {error_mm:.1f}mm out"


# ── the desk case ──────────────────────────────────────────────────────────


def test_a_seated_subject_still_measures_their_own_height_and_feet() -> None:
    """Sitting at a desk: no knees, no ankles, no feet. The fit must still size them.

    This is the case the proportional template exists for. Nothing here looks for a floor,
    and nothing assumes the subject is standing - the height comes from the arms, trunk and
    head, and the invisible foot is as long as they say it is.
    """
    skeleton = _skeleton()
    full_observation = _observed_subject(skeleton=skeleton)
    seated = {
        name: position
        for name, position in full_observation.items()
        if not any(hidden in name for hidden in _BELOW_THE_DESK)
    }
    assert len(seated) < len(full_observation), "the desk must actually hide something"

    pose = _hydrate(skeleton=skeleton, observed=seated)
    fit = _fit_from_one_pose(
        skeleton=skeleton, pose=pose, measured=frozenset(seated)
    )

    assert fit.body_height == pytest.approx(SUBJECT_HEIGHT_MM, rel=0.02)

    # The foot was never seen, so it wears the pooled height - and lands on its true size.
    assert "left_foot" not in fit.measured_segment_names
    true_foot_length_mm = skeleton.segments["left_foot"].length * SUBJECT_HEIGHT_MM
    assert fit.segment_lengths["left_foot"] == pytest.approx(
        true_foot_length_mm, rel=0.02
    )

    # And every segment has a length, seen or not - that is what the template is for.
    assert set(fit.segment_lengths) == set(skeleton.segments)
    assert all(length > 0.0 for length in fit.segment_lengths.values())


def test_one_side_occluded_still_fits_from_the_other() -> None:
    """Turned side-on to every camera. Left and right pool, so the height survives."""
    skeleton = _skeleton()
    full_observation = _observed_subject(skeleton=skeleton)
    one_sided = {
        name: position
        for name, position in full_observation.items()
        if not name.startswith("right_")
    }

    pose = _hydrate(skeleton=skeleton, observed=one_sided)
    fit = _fit_from_one_pose(
        skeleton=skeleton, pose=pose, measured=frozenset(one_sided)
    )

    assert fit.body_height == pytest.approx(SUBJECT_HEIGHT_MM, rel=0.02)
    # The unseen right thigh is sized from the seen left one, via the pooled height.
    assert fit.segment_lengths["right_upper_leg"] == pytest.approx(
        skeleton.segments["right_upper_leg"].length * SUBJECT_HEIGHT_MM, rel=0.02
    )


# ── robustness ─────────────────────────────────────────────────────────────


def test_a_mistracked_limb_does_not_resize_the_body() -> None:
    """One limb triangulated at three times its length must not move the height at all."""
    skeleton = _skeleton()
    measured = _all_landmarks_measured(skeleton)
    voting = body_scale_voting_segment_names(
        skeleton=skeleton, measured_landmark_names=measured
    )
    honest = {name: [SUBJECT_HEIGHT_MM] * 30 for name in sorted(voting)}

    clean_fit = fit_body_scale(
        skeleton=skeleton, scale_samples=honest, voting_segment_names=voting
    )

    corrupted = dict(honest)
    corrupted["left_lower_arm"] = [3.0 * SUBJECT_HEIGHT_MM] * 30
    corrupted["left_upper_arm"] = [3.0 * SUBJECT_HEIGHT_MM] * 30
    corrupted_fit = fit_body_scale(
        skeleton=skeleton, scale_samples=corrupted, voting_segment_names=voting
    )

    assert corrupted_fit.body_height == pytest.approx(clean_fit.body_height, rel=1e-9)


def test_a_jittering_segment_is_pulled_further_toward_the_pooled_height() -> None:
    """Shrinkage: at the same evidence count, readings that disagree buy less.

    Asserted as MONOTONICITY rather than as a number. How far a given dispersion pulls is a
    tuning constant; that more dispersion pulls further is the design, and no accidental
    retune can satisfy it.
    """
    skeleton = _skeleton()
    voting = body_scale_voting_segment_names(
        skeleton=skeleton, measured_landmark_names=_all_landmarks_measured(skeleton)
    )
    rng = np.random.default_rng(4)
    sample_count = 30

    def fitted_scale_with_jitter(relative_jitter: float) -> tuple[float, float, float]:
        samples = {name: [SUBJECT_HEIGHT_MM] * sample_count for name in sorted(voting)}
        # One segment reads 20% long, with the given amount of frame-to-frame disagreement.
        readings = (
            1.2
            * SUBJECT_HEIGHT_MM
            * (1.0 + relative_jitter * rng.normal(size=sample_count))
        )
        samples["left_lower_leg"] = list(readings)
        fit = fit_body_scale(
            skeleton=skeleton, scale_samples=samples, voting_segment_names=voting
        )
        return (
            fit.segment_scales["left_lower_leg"],
            fit.body_height,
            float(np.median(readings)),
        )

    steady, steady_height, steady_own = fitted_scale_with_jitter(0.01)
    jittery, jittery_height, jittery_own = fitted_scale_with_jitter(0.20)

    # Both land strictly between the pooled height and their own reading...
    assert steady_height < steady < steady_own
    assert jittery_height < jittery < jittery_own
    # ...and the jittery one is the further of the two toward the pooled height.
    assert (jittery - jittery_height) / (jittery_own - jittery_height) < (
        steady - steady_height
    ) / (steady_own - steady_height)


def test_a_steady_segment_keeps_its_own_measurement() -> None:
    """The other end of the same rule: a genuinely long femur stays long."""
    skeleton = _skeleton()
    measured = _all_landmarks_measured(skeleton)
    voting = body_scale_voting_segment_names(
        skeleton=skeleton, measured_landmark_names=measured
    )
    samples = {name: [SUBJECT_HEIGHT_MM] * 30 for name in sorted(voting)}
    long_femur = 1.1 * SUBJECT_HEIGHT_MM
    samples["left_upper_leg"] = [long_femur] * 30

    fit = fit_body_scale(
        skeleton=skeleton, scale_samples=samples, voting_segment_names=voting
    )

    assert fit.segment_scales["left_upper_leg"] == pytest.approx(long_femur, rel=0.05)
    # The right femur was not measured differently, so it keeps the pooled height - a real
    # limb-length difference survives the fit rather than being averaged away.
    assert fit.segment_scales["right_upper_leg"] == pytest.approx(
        fit.body_height, rel=1e-9
    )


# ── who is allowed to vote ─────────────────────────────────────────────────


def test_segments_built_from_synthesized_landmarks_do_not_vote() -> None:
    """The template must not be allowed to quote itself back as evidence."""
    skeleton = _skeleton()
    # A caller whose mapping synthesizes the whole sternum/neck complex.
    synthesized = {
        "left_sternoclavicular",
        "right_sternoclavicular",
        "sternoclavicular_notch",
        "xiphoid_process",
        "neck_center",
        "chest_center",
    }
    measured = frozenset(skeleton.landmarks) - synthesized

    voting = body_scale_voting_segment_names(
        skeleton=skeleton, measured_landmark_names=measured
    )

    assert "sacrolumbar" not in voting, "its primary is a synthesized chest_center"
    assert "thoracic" not in voting, "it rigid-fits over synthesized landmarks"
    assert "left_clavicle" not in voting, "its origin is a synthesized sternoclavicular"
    # The limbs, which are what actually measures the subject, still do.
    for name in ("left_upper_leg", "left_lower_leg", "left_upper_arm", "left_lower_arm"):
        assert name in voting


def test_a_rigid_fit_segment_needs_every_one_of_its_landmarks_measured() -> None:
    """Strict by design: the pelvis is excluded by its synthesized iliac crests."""
    skeleton = _skeleton()
    measured = frozenset(skeleton.landmarks) - {"left_iliac_crest"}

    voting = body_scale_voting_segment_names(
        skeleton=skeleton, measured_landmark_names=measured
    )

    assert "pelvis" not in voting


def test_no_voting_segment_seen_is_a_refusal_not_a_default() -> None:
    skeleton = _skeleton()
    fitter = StreamingBodyScaleFitter(
        skeleton=skeleton,
        voting_segment_names=frozenset({"left_upper_leg"}),
    )
    assert not fitter.has_body_scale
    with pytest.raises(InsufficientScaleEvidence, match="no segment entitled to vote"):
        fitter.current_fit()


def test_readings_from_non_voting_segments_alone_are_still_a_refusal() -> None:
    """Seeing only synthesized geometry is not seeing the subject."""
    skeleton = _skeleton()
    with pytest.raises(InsufficientScaleEvidence):
        fit_body_scale(
            skeleton=skeleton,
            scale_samples={"thoracic": [SUBJECT_HEIGHT_MM]},
            voting_segment_names=frozenset({"left_upper_leg"}),
        )


# ── the streaming wrapper ──────────────────────────────────────────────────


def test_the_window_is_bounded_and_forgets_the_oldest_readings() -> None:
    """A subject who walks closer gets re-fitted within a window, not averaged forever."""
    skeleton = _skeleton()
    fitter = StreamingBodyScaleFitter(
        skeleton=skeleton,
        voting_segment_names=body_scale_voting_segment_names(
            skeleton=skeleton,
            measured_landmark_names=_all_landmarks_measured(skeleton),
        ),
        window_frames=DEFAULT_SCALE_WINDOW_FRAMES,
    )
    short_subject = _observed_subject(skeleton=skeleton, height_mm=1500.0)
    tall_subject = _observed_subject(skeleton=skeleton, height_mm=1900.0)

    for _ in range(DEFAULT_SCALE_WINDOW_FRAMES):
        fitter.observe_pose(pose=_hydrate(skeleton=skeleton, observed=short_subject))
    assert fitter.current_fit().body_height == pytest.approx(1500.0, rel=1e-6)

    for _ in range(DEFAULT_SCALE_WINDOW_FRAMES):
        fitter.observe_pose(pose=_hydrate(skeleton=skeleton, observed=tall_subject))
    assert fitter.current_fit().body_height == pytest.approx(1900.0, rel=1e-6)


def test_a_segment_that_stops_being_seen_keeps_its_measurement() -> None:
    """Standing, then sitting down. The femur does not change length when the desk hides it."""
    skeleton = _skeleton()
    standing = _observed_subject(skeleton=skeleton)
    seated = {
        name: position
        for name, position in standing.items()
        if not any(hidden in name for hidden in _BELOW_THE_DESK)
    }
    fitter = StreamingBodyScaleFitter(
        skeleton=skeleton,
        voting_segment_names=body_scale_voting_segment_names(
            skeleton=skeleton,
            measured_landmark_names=_all_landmarks_measured(skeleton),
        ),
    )

    for _ in range(DEFAULT_SCALE_WINDOW_FRAMES):
        fitter.observe_pose(pose=_hydrate(skeleton=skeleton, observed=standing))
    standing_fit = fitter.current_fit()

    for _ in range(DEFAULT_SCALE_WINDOW_FRAMES):
        fitter.observe_pose(pose=_hydrate(skeleton=skeleton, observed=seated))
    seated_fit = fitter.current_fit()

    assert "left_upper_leg" in seated_fit.measured_segment_names
    assert seated_fit.segment_lengths["left_upper_leg"] == pytest.approx(
        standing_fit.segment_lengths["left_upper_leg"], rel=1e-6
    )
    assert seated_fit.body_height == pytest.approx(standing_fit.body_height, rel=1e-6)


def test_reset_forgets_the_body() -> None:
    skeleton = _skeleton()
    fitter = StreamingBodyScaleFitter(
        skeleton=skeleton,
        voting_segment_names=body_scale_voting_segment_names(
            skeleton=skeleton,
            measured_landmark_names=_all_landmarks_measured(skeleton),
        ),
    )
    fitter.observe_pose(
        pose=_hydrate(skeleton=skeleton, observed=_observed_subject(skeleton=skeleton))
    )
    assert fitter.has_body_scale

    fitter.reset()

    assert not fitter.has_body_scale


def test_the_fit_covers_every_segment_even_from_a_single_reading() -> None:
    skeleton = _skeleton()
    fit = fit_body_scale(
        skeleton=skeleton,
        scale_samples={"left_upper_leg": [SUBJECT_HEIGHT_MM]},
        voting_segment_names=frozenset({"left_upper_leg"}),
    )
    assert set(fit.segment_scales) == set(skeleton.segments)
    assert set(fit.segment_lengths) == set(skeleton.segments)
    assert fit.body_height == pytest.approx(SUBJECT_HEIGHT_MM)
    assert fit.measured_segment_names == frozenset({"left_upper_leg"})


def test_samples_for_an_unknown_segment_are_an_error() -> None:
    skeleton = _skeleton()
    with pytest.raises(KeyError, match="not in skeleton"):
        fit_body_scale(
            skeleton=skeleton,
            scale_samples={"left_upper_leg": [1.0], "tentacle": [1.0]},
            voting_segment_names=frozenset({"left_upper_leg"}),
        )
