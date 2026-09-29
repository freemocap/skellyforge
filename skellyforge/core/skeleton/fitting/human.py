"""Accepted human fit: flexible axial lengths, position preferences, robust keypoints.

Input trajectories and person scale are prepared by the caller. No recording I/O,
viewer imports, tracker dependency, or scale estimation occurs here.
"""
from dataclasses import dataclass
from skellyforge import _native
from .body_model import body_model
from .preparation import prepare_body_fit
from .window_sequence import fit_windows
from .settings import SPINE_PROPORTION_RATIOS


LANDMARKS = ('pelvis_origin', 'chest_center', 'neck_center', 'craniocervical_junction')
POSITION_SCALE_MM = 5.0
KEYPOINT_HUBER_SCALE_MM = 30.0
ACTIVE_FRAMES = 3
MAX_ITERATIONS = 200
FUNCTION_TOLERANCE = 1e-6
REST_LENGTH_FRACTION = 0.5
TWIST_SCALE = 0.5


def human_fit_options():
    """Fresh argument options for the accepted configuration, not anatomical limits."""
    return dict(free_axial_lengths=True, free_length_rest_prior=True,
        length_prior_fraction=REST_LENGTH_FRACTION, chest_line_prior=True,
        shoulder_profile='lower_closer_sc', relaxed_shoulders=True,
        proportional_spine_lengths=True, spine_proportion_ratios=SPINE_PROPORTION_RATIOS,
        shoulder_axis_preference=True, spine_twist_scale=TWIST_SCALE)


def position_priors(model, records, scale=POSITION_SCALE_MM):
    priors=[]
    for name in LANDMARKS:
        b=next(i for i,keys in enumerate(model['display_names']) if name in keys)
        slot=model['display_names'][b].index(name)
        prior=_native.LandmarkPositionPrior()
        prior.segment=b;prior.local_point=model['display'][b][slot].tolist()
        prior.scale=scale
        # Do not silently substitute another landmark or a fabricated measurement.
        prior.targets=[r['points'][name].tolist() for r in records]
        priors.append(prior)
    return priors


def fit_prepared_human(arguments, model, records, *, inspect_windows=(), progress=None):
    """Execute the accepted fit using native arguments prepared from this model."""
    arguments = dict(arguments)
    arguments['landmark_position_priors'] = position_priors(model, records)
    arguments['landmark_huber_scale_mm'] = KEYPOINT_HUBER_SCALE_MM
    return fit_windows(arguments, active_frames=ACTIVE_FRAMES,
        max_iterations=MAX_ITERATIONS, function_tolerance=FUNCTION_TOLERANCE,
        inspect_windows=inspect_windows, progress=progress)


@dataclass
class HumanFit:
    model: dict
    preparation: dict
    sequence: object


def fit_human(skeleton, saved_model, segment_scales, records, *, inspect_windows=(), progress=None):
    """Fit in-memory prepared trajectories to the supplied connected human skeleton.

    Records contain number/time, mapped landmark points, source keypoints, saved
    segment rotations and origins. See README for conventions and prerequisites.
    """
    if len(records) < ACTIVE_FRAMES:
        raise ValueError('Human fitting requires at least three frames')
    options = human_fit_options()
    model = body_model(skeleton, saved_model, segment_scales,
        options['shoulder_profile'], flexible_cervical=True)
    arguments, preparation = prepare_body_fit(model, records, **options)
    sequence = fit_prepared_human(arguments, model, records,
        inspect_windows=inspect_windows, progress=progress)
    return HumanFit(model, preparation, sequence)
