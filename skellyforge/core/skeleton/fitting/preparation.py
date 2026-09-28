"""Build native arguments from prepared geometry and in-memory trajectories.

No recording reads, viewer objects or optimization occur here.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge import _native
from . import settings as fit_settings
from .body_model import frame_targets
from .centerline import mapped_centerline, CHEST_LINE_DISTANCE_SCALE_MM, CHEST_LINE_ANTERIOR_SCALE_MM

def prepare_body_fit(m, selected, *, length_prior_fraction=fit_settings.LENGTH_PRIOR_FRACTION,
                           lengthening_prior_fraction=None, free_axial_lengths=False, chest_line_prior=False, chest_line_distance_scale=CHEST_LINE_DISTANCE_SCALE_MM, shoulder_profile=None, relaxed_shoulders=False, equal_spine_lengths=False, proportional_spine_lengths=False, spine_proportion_ratios=None, free_length_rest_prior=False, shared_spine_length=False, shoulder_axis_preference=False, shared_total_bound_fractions=None, spine_twist_scale=None, sc_anterior_scale=None):
    observed, indices, initial, roots = [], [], [], []
    seed_fallbacks = 0
    root_name=m['names'][0]
    root_seeds=[r for r in selected if r['rotations'].get(root_name) is not None and root_name in r['origins']]
    if not root_seeds:
        raise ValueError('No saved root pose available in the selected interval for initialization')
    root_seed_fallbacks=0
    for record in selected:
        obs, slots = frame_targets(record, m)
        observed.append(obs); indices.append(slots)
        root_seed=record
        if record['rotations'].get(root_name) is None or root_name not in record['origins']:
            root_seed=min(root_seeds,key=lambda r:abs(r['time']-record['time']))
            root_seed_fallbacks+=1
        quaternions = []
        for b, name in enumerate(m['names']):
            q = root_seed['rotations'][name] if b==0 else record['rotations'].get(name)
            if q is None:
                q = (Rotation.from_quat(quaternions[m['parents'][b-1]], scalar_first=True)
                     * Rotation.from_quat(m['relative'][b-1], scalar_first=True)).as_quat(scalar_first=True)
                seed_fallbacks += 1
            quaternions.append(np.asarray(q).tolist())
        initial.append(quaternions)
        roots.append(root_seed['origins'][root_name].tolist())
    chest_body = next(b for b, keys in enumerate(m['display_names']) if 'chest_center' in keys)
    chest_slot = m['display_names'][chest_body].index('chest_center')
    line_frames = [mapped_centerline(r['keypoints'], m['sources']) for r in selected]
    prior = None
    if chest_line_prior:
        prior = _native.LandmarkLinePrior()
        prior.segment = chest_body
        prior.local_point = m['display'][chest_body][chest_slot].tolist()
        prior.frames = [None if f is None else [f['origin'],f['lateral'],f['anterior']] for f in line_frames]
        prior.distance_scale = chest_line_distance_scale
        prior.anterior_scale = CHEST_LINE_ANTERIOR_SCALE_MM
    relaxed = [m['names'].index(side+'_upper_arm') for side in ('left','right')] if relaxed_shoulders else []
    for child in relaxed:
        if m['names'][m['parents'][child-1]] not in ('left_clavicle','right_clavicle'):
            raise ValueError('Shoulder experiment requires authored clavicle-to-upper-arm linkages')
    equality = None
    if equal_spine_lengths:
        equality = _native.LengthEqualityPrior()
        equality.segment_a = m['names'].index('sacrolumbar')
        equality.segment_b = m['names'].index('thoracic')
        equality.scale = fit_settings.SPINE_LENGTH_EQUALITY_SCALE_MM
    if shared_spine_length and not proportional_spine_lengths:
        raise ValueError("Shared spine length requires explicit spine proportions")
    proportion = None
    if proportional_spine_lengths:
        if equal_spine_lengths:raise ValueError('Choose equal lengths or proportional lengths, not both')
        proportion=_native.LengthProportionPrior()
        proportion.segments=[m['names'].index(n) for n in fit_settings.SPINE_PROPORTION_SEGMENTS]
        proportion.ratios=list(fit_settings.SPINE_PROPORTION_RATIOS if spine_proportion_ratios is None else spine_proportion_ratios)
        proportion.scale=fit_settings.SPINE_LENGTH_PROPORTION_SCALE_MM
    if shared_total_bound_fractions is not None and not shared_spine_length:raise ValueError("Total bounds require shared spine length")
    shared = None
    if shared_spine_length:
        shared = _native.SharedAxialLength()
        shared.segments = proportion.segments
        shared.ratios = proportion.ratios
        if shared_total_bound_fractions is not None:
            reference_total=sum(m['references'][b] for b in shared.segments)
            shared.minimum_total=reference_total*shared_total_bound_fractions[0]
            shared.maximum_total=reference_total*shared_total_bound_fractions[1]
    times = [r['time']-selected[0]['time'] for r in selected]
    from .axis_priors import shoulder_axis_prior
    axis_prior = shoulder_axis_prior(selected, m) if shoulder_axis_preference else None
    from .spine_priors import twist_priors
    twist = twist_priors(m, spine_twist_scale) if spine_twist_scale is not None else []
    from .sc_prior import sc_prior, audit_cervical_attachment
    audit_cervical_attachment(m)
    sc = sc_prior(m, line_frames, sc_anterior_scale) if sc_anterior_scale is not None else None
    arguments = dict(
        landmark_half_space_prior=sc, segment_axis_prior=axis_prior, relative_twist_priors=twist,
        relaxed_linkage_children=relaxed, linkage_scale=fit_settings.SHOULDER_LINKAGE_SCALE_MM,
        linkage_acceleration_scale=fit_settings.SHOULDER_LINKAGE_ACCELERATION_SCALE_MM_S2,
        length_equality_prior=equality, length_proportion_prior=None if shared else proportion, shared_axial_length=shared, landmark_line_prior=prior, length_prior_fraction=length_prior_fraction,
        lengthening_prior_fraction=lengthening_prior_fraction, free_axial_lengths=free_axial_lengths, free_length_rest_prior=free_length_rest_prior,
        length_acceleration_scale=fit_settings.LENGTH_ACCELERATION_SCALE_MM_S2,
        local=m['local'], observed=observed, observation_indices=indices,
        parent_attachments=m['attachments'], child_attachments=[[0.,0.,0.]]*len(m['parents']), parent_indices=m['parents'],
        times=times, position_scale=fit_settings.POSITION_RESIDUAL_SCALE_MM, linear_acceleration_scale=fit_settings.ROOT_ACCELERATION_SCALE_MM_S2, angular_acceleration_scale=fit_settings.ANGULAR_ACCELERATION_SCALE_RAD_S2,
        initial_quaternions=initial, initial_roots=roots, rest_relative_quaternions=m['relative'], rest_pose_scale=fit_settings.REST_POSE_RESIDUAL_SCALE_RAD,
        axial_reference_lengths=m['references'])
    return arguments, dict(axis_prior=axis_prior, chest_body=chest_body, chest_slot=chest_slot, indices=indices, line_frames=line_frames, observed=observed, proportion=proportion, relaxed=relaxed, root_seed_fallbacks=root_seed_fallbacks, sc=sc, seed_fallbacks=seed_fallbacks, shared=shared, times=times, twist=twist)
