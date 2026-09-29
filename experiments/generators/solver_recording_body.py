"""Full saved skeleton in the Ceres lab; source recording is read-only.

Keypoints are measurements; saved direct mappings hydrate landmark targets.
Every authored landmark remains in the model and in the fitted output. Targets
without their source keypoint in a frame contribute no measurement residual.
"""
from skellyforge.tools.viewer.workspace import REPO_ROOT, OUTPUT_FOLDER, prepare_output
import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from skellyforge import _native
from skellyforge.core.skeleton.fitting import settings as fit_settings
from experiments.generators.solver_shoulder_offsets import reference_positions, shoulder_diagnostics
from experiments.generators.solver_chest_line import (mapped_centerline, line_diagnostics, CHEST_LINE_DISTANCE_SCALE_MM, CHEST_LINE_ANTERIOR_SCALE_MM)
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from skellyforge.tools.recording_data import read_recording, recording_path, digest
from experiments.generators.solver_axial_geometry import axial_points
from experiments.generators.solver_recording_context import add_context
from experiments.generators.generate_solver_viewer import render_experiments


from skellyforge.core.skeleton.fitting.body_model import body_model, frame_targets
from skellyforge.core.skeleton.fitting.preparation import prepare_body_fit


def display_bodies(skeleton, model):
    names, parents = model["names"], model["parents"]
    display, display_names = model["display"], model["display_names"]
    attachments = model["attachments"]
    joints = {j.child.name: j for j in skeleton.joints.values()}

    region_roots = {'pelvis': 'Trunk', 'cervical_spine': 'Head and neck',
                    'left_clavicle': 'Left arm', 'right_clavicle': 'Right arm',
                    'left_carpals': 'Left hand', 'right_carpals': 'Right hand',
                    'left_upper_leg': 'Left leg and foot', 'right_upper_leg': 'Right leg and foot'}
    regions = []
    bodies = []
    for b, name in enumerate(names):
        regions.append(region_roots.get(name, regions[parents[b-1]] if b else 'Body'))
        segment = skeleton.segments[name]
        links = [dict(position=display[b][display_names[b].index(segment.frame_definition.primary_point_name)].tolist(),
                      label=segment.frame_definition.primary_point_name)] if segment.frame_definition.primary_point_name in display_names[b] else []
        links.extend(dict(position=attachments[c-1], label=joints[names[c]].connect_at.name)
                     for c, parent in enumerate(parents, 1) if parent == b)
        bodies.append(dict(id=name, label=name, region=regions[b], landmark_names=display_names[b], edges=[], attachments=links))
    return bodies


def recording_body_catalog(path, start=180, end=213, *, length_prior_fraction=fit_settings.LENGTH_PRIOR_FRACTION,
                           lengthening_prior_fraction=None, free_axial_lengths=False, chest_line_prior=False, chest_line_distance_scale=CHEST_LINE_DISTANCE_SCALE_MM, shoulder_profile=None, relaxed_shoulders=False, equal_spine_lengths=False, proportional_spine_lengths=False, spine_proportion_ratios=None, free_length_rest_prior=False, shared_spine_length=False, shoulder_axis_preference=False, shared_total_bound_fractions=None, spine_twist_scale=None, sc_anterior_scale=None, solve_sequence=None):
    records, scale, provenance = read_recording(path, include_model=True)
    saved = provenance.pop('model')
    skeleton = SkeletonSnapshot.from_dict(provenance.pop('skeleton')).restore()
    m = body_model(skeleton, saved, scale.segment_scales, shoulder_profile, flexible_cervical=proportional_spine_lengths)
    m['bodies'] = display_bodies(skeleton, m)
    selected = [r for r in records if start <= r['number'] <= end]
    if len(selected) != end-start+1 or len(selected) < 3:
        raise ValueError('Requested consecutive interval is not available')
    arguments, preparation = prepare_body_fit(m, selected,
        length_prior_fraction=length_prior_fraction, lengthening_prior_fraction=lengthening_prior_fraction, free_axial_lengths=free_axial_lengths, chest_line_prior=chest_line_prior, chest_line_distance_scale=chest_line_distance_scale, shoulder_profile=shoulder_profile, relaxed_shoulders=relaxed_shoulders, equal_spine_lengths=equal_spine_lengths, proportional_spine_lengths=proportional_spine_lengths, spine_proportion_ratios=spine_proportion_ratios, free_length_rest_prior=free_length_rest_prior, shared_spine_length=shared_spine_length, shoulder_axis_preference=shoulder_axis_preference, shared_total_bound_fractions=shared_total_bound_fractions, spine_twist_scale=spine_twist_scale, sc_anterior_scale=sc_anterior_scale)
    axis_prior = preparation["axis_prior"]
    chest_body = preparation["chest_body"]
    chest_slot = preparation["chest_slot"]
    indices = preparation["indices"]
    line_frames = preparation["line_frames"]
    observed = preparation["observed"]
    proportion = preparation["proportion"]
    relaxed = preparation["relaxed"]
    root_seed_fallbacks = preparation["root_seed_fallbacks"]
    sc = preparation["sc"]
    seed_fallbacks = preparation["seed_fallbacks"]
    shared = preparation["shared"]
    times = preparation["times"]
    twist = preparation["twist"]
    result = (solve_sequence or _native.fit_chain_sequence)(**arguments)
    # pybind vector properties copy the complete sequence on access. Read once,
    # before iterating over frames and segments; preserve native values exactly.
    poses = SimpleNamespace(**{name: getattr(result, name) for name in (
        'quaternions', 'translations', 'initial_quaternions', 'initial_translations',
        'lengths', 'linkage_displacements', 'roots', 'costs')})
    axial = [b for b, ref in enumerate(m['references']) if ref]
    fixed_lengths = set(getattr(result, 'processing', {}).get('fixed_length_segments', []))
    frames = []
    attachment_errors = []
    linkage_separations = []
    for i, record in enumerate(selected):
        bodies = []
        linkages = []
        diagnostics = {'Recording frame': record['number'], 'Recording timestamp (s)': record['time']}
        for b, name in enumerate(m['names']):
            q, t = poses.quaternions[i][b], poses.translations[i][b]
            iq, it = poses.initial_quaternions[i][b], poses.initial_translations[i][b]
            ref, length = m['references'][b], poses.lengths[i][b]
            local = axial_points(m['display'][b], ref, length)
            fitted = Rotation.from_quat(q, scalar_first=True).apply(local)+t
            starting = Rotation.from_quat(iq, scalar_first=True).apply(m['display'][b])+it
            by_name = {m['targets'][b][slot]: point for slot, point in zip(indices[i][b], observed[i][b])}
            obs = [by_name.get(key) for key in m['display_names'][b]]
            errors = [float(np.linalg.norm(fitted[j]-p)) if p is not None else None for j,p in enumerate(obs)]
            support = np.array([m['local'][b][j] for j in indices[i][b]])
            rank = int(np.linalg.matrix_rank(support-support.mean(axis=0), tol=fit_settings.TARGET_RANK_TOLERANCE_MM)) if len(support) else 0
            if b:
                parent = m['parents'][b-1]
                expected = np.array(poses.translations[i][parent])+Rotation.from_quat(poses.quaternions[i][parent], scalar_first=True).apply(
                    axial_points(m['attachments'][b-1], m['references'][parent], poses.lengths[i][parent]))
                delta=np.asarray(poses.linkage_displacements[i][b])
                if b in relaxed:
                    separation=float(np.linalg.norm(delta));linkage_separations.append(separation)
                    diagnostics[name+' linkage separation (mm)']=separation
                    linkages.append(dict(child=b,parent=parent,local_displacement=delta.tolist(),parent_point=expected.tolist(),child_point=list(t)))
                displaced=expected+Rotation.from_quat(poses.quaternions[i][parent],scalar_first=True).apply(delta)
                attachment_errors.append(float(np.linalg.norm(displaced-t)))
            bodies.append(dict(axial_scale=length/ref if ref else 1., mechanical_model='axial' if ref and b not in fixed_lengths else 'rigid',
                target_count=len(support), target_rank=rank, local=local.tolist(), truth=None, observed=obs,
                fitted=fitted.tolist(), initial=starting.tolist(), quaternion=q, translation=t,
                initial_quaternion=iq, initial_translation=it, reference_quaternion=None, reference_translation=None,
                residuals=errors, truth_rms=None))
        spine_axes = [Rotation.from_quat(poses.quaternions[i][m['names'].index(name)], scalar_first=True).apply([0,0,1]) for name in ('sacrolumbar','thoracic')]
        diagnostics['Spine bend (degrees)'] = float(np.degrees(np.arccos(np.clip(np.dot(*spine_axes),-1,1))))
        diagnostics['Unavailable source keypoints for selected mappings'] = len(m['sources'])-sum(map(len,observed[i]))
        diagnostics['Mapped keypoint targets'] = sum(map(len,observed[i]))
        if not diagnostics['Mapped keypoint targets']:
            diagnostics['Pose support warning']='No mapped keypoint targets: fitted pose follows temporal and model residuals.'
        for b in axial:
            diagnostics[m['names'][b]+' length (mm)'] = poses.lengths[i][b]
        frames.append(dict(linkages=linkages, time=times[i], bodies=bodies, root=poses.roots[i], costs=poses.costs,
            lengths=[poses.lengths[i][b] for b in axial], reference_lengths=[m['references'][b] for b in axial],
            converged=result.converged, seconds=result.seconds, report=result.report, diagnostics=diagnostics,
            observability='All model landmarks remain defined. Direct mapped keypoints supply measurement residuals. Exact joints, rest-pose and temporal residuals couple the full skeleton. Target counts do not establish global observability.'))
    for frame,geometry in zip(frames,line_frames):
        line_diagnostics(frame,geometry,chest_body,chest_slot)
    shoulder_diagnostics(frames, m['bodies'])
    errors = [e for f in frames for b in f['bodies'] for e in b['residuals'] if e is not None]
    summary = {'Ceres parameter blocks': result.parameter_blocks, 'Ceres residual blocks': result.residual_blocks,
               'Observed landmark RMS (mm)': float(np.sqrt(np.mean(np.square(errors)))),
               'Maximum attachment equation error (mm)': max(attachment_errors), 'Rest-pose initialization fallbacks': seed_fallbacks}
    if relaxed:
        summary['Maximum shoulder separation (mm)']=max(linkage_separations)
        summary['RMS shoulder separation (mm)']=float(np.sqrt(np.mean(np.square(linkage_separations))))
    summary['Frames at axial length bounds'] = sum(any(abs(poses.lengths[i][b]-bound*m['references'][b])<fit_settings.BOUND_CONTACT_TOLERANCE_MM
        for b in axial for bound in ((0.,) if free_axial_lengths else fit_settings.AXIAL_LENGTH_BOUND_FRACTIONS)) for i in range(len(frames)))
    if root_seed_fallbacks:
        summary['Root initialization fallbacks']=root_seed_fallbacks
    summary['Frames without mapped keypoint targets']=sum(not sum(map(len,f)) for f in observed)
    for region in sorted({b['region'] for b in m['bodies']}):
        values = [e for f in frames for b,definition in zip(f['bodies'],m['bodies']) if definition['region']==region for e in b['residuals'] if e is not None]
        summary[region+' target RMS (mm)'] = float(np.sqrt(np.mean(np.square(values)))) if values else 'No direct targets'
    method = dict(frames=frames, summary=summary, problem=dict(chain=True, parents=m['parents'], connected=True,
        temporal=True, acceleration=True, rest_prior=True, axial_segments=axial, **(dict(relaxed_linkage_children=relaxed) if relaxed else {})), settings=dict(
        relaxed_linkage_children=relaxed, linkage_scale_mm=fit_settings.SHOULDER_LINKAGE_SCALE_MM,
        linkage_acceleration_scale_mm_s2=fit_settings.SHOULDER_LINKAGE_ACCELERATION_SCALE_MM_S2,
        length_equality_prior=dict(enabled=equal_spine_lengths,
            segments=[m['names'].index('sacrolumbar'),m['names'].index('thoracic')],
            segment_names=['sacrolumbar','thoracic'], scale_mm=fit_settings.SPINE_LENGTH_EQUALITY_SCALE_MM),
        shoulder_geometry=m['shoulder_geometry'],
        chest_line_prior=dict(enabled=chest_line_prior, body_index=chest_body, landmark_index=chest_slot,
            landmark='chest_center', distance_scale_mm=chest_line_distance_scale,
            anterior_scale_mm=CHEST_LINE_ANTERIOR_SCALE_MM,
            frame_definition='Hip center to shoulder center: up; cross(up, left-to-right hip vector): anterior; cross(anterior, up): lateral.'),
        free_length_rest_prior=free_length_rest_prior, free_axial_lengths=free_axial_lengths,
        axial_length_bound_fractions=(0., None) if free_axial_lengths else fit_settings.AXIAL_LENGTH_BOUND_FRACTIONS, axial_reference_lengths=m['references'], length_prior_fraction=length_prior_fraction,
        lengthening_prior_fraction=length_prior_fraction if lengthening_prior_fraction is None else lengthening_prior_fraction,
        length_acceleration_scale=fit_settings.LENGTH_ACCELERATION_SCALE_MM_S2,
        position_scale_mm=fit_settings.POSITION_RESIDUAL_SCALE_MM, linear_motion_scale=fit_settings.ROOT_ACCELERATION_SCALE_MM_S2, angular_motion_scale=fit_settings.ANGULAR_ACCELERATION_SCALE_RAD_S2, scale_units=['mm/s^2','rad/s^2'],
        rest_pose_scale_radians=fit_settings.REST_POSE_RESIDUAL_SCALE_RAD, rest_relative_quaternions=m['relative'], direct_mapping_sources=m['sources'],
        initialization='Saved segment quaternions and root origin. Where a saved quaternion is unavailable: parent seed composed with authored relative T-pose. Initialization is not a measurement residual.',
        costs_by_family=dict(landmark_position_prior=result.position_prior_cost, sc_anterior=result.half_space_cost, relative_twist=result.twist_prior_cost, axis_prior=result.axis_prior_cost, length_proportion=result.length_proportion_cost, length_equality=result.length_equality_cost, linkage_prior=result.linkage_prior_cost, linkage_acceleration=result.linkage_acceleration_cost, landmarks=result.landmark_cost, relative_pose=result.relative_pose_cost,
            root_acceleration=result.root_acceleration_cost, segment_angular_acceleration=result.angular_acceleration_costs,
            chest_line_prior=result.line_prior_cost, length_prior=result.length_prior_cost, length_acceleration=result.length_acceleration_cost)),
        objective='One connected full-body Ceres problem. Direct mapped keypoint targets counted once; no derived-landmark measurement residuals. Exact attachments; sacrolumbar and thoracic axial lengths; all other geometry rigid. Quaternion rest-pose and temporal residuals are preferences, not joint limits.')
    if proportional_spine_lengths:
        method['settings']['length_proportion_prior']=dict(enabled=not shared_spine_length,segments=proportion.segments,
            segment_names=list(fit_settings.SPINE_PROPORTION_SEGMENTS),ratios=list(proportion.ratios),
            fractions=(np.array(proportion.ratios)/sum(proportion.ratios)).tolist(),scale_mm=proportion.scale,
            source='User-supplied experimental ratios; Winter/de Leva attribution not verified.')
        method['objective']=method['objective'].replace('sacrolumbar and thoracic axial lengths','sacrolumbar, thoracic and cervical axial lengths')
        method['objective']+=' Three-length proportion preference; total length remains free. User-supplied ratios, not verified published anthropometry.'
    method['settings']['sc_anterior_prior']=None if sc is None else dict(scale_mm=sc.scale, supported_frames=sum(f is not None for f in line_frames), source='Hip/shoulder keypoint frame; shoulder midpoint plane; anterior side preferred, no positive margin.')
    if twist:method['settings']['relative_twist_priors']=[dict(parent=m['names'][p.parent],child=m['names'][p.child],
        reference=p.reference,axis=p.axis,scale=p.scale) for p in twist]
    if axis_prior:
        method['settings']['segment_axis_prior']=dict(segment='thoracic',scale=axis_prior.scale,
            local_lateral=axis_prior.local_lateral,local_anterior=axis_prior.local_anterior,
            source='Left-to-right shoulder keypoint axis; transverse projection chord residual, not an independent measurement.',
            supported_frames=sum(f is not None for f in axis_prior.frames))
        method['objective']+=' Soft thoracic SC-axis/shoulder-axis transverse alignment preference.'
    if shared_spine_length:
        method['settings']['shared_axial_length'] = dict(segments=shared.segments, ratios=shared.ratios,
            segment_names=list(fit_settings.SPINE_PROPORTION_SEGMENTS), parameter='total length in mm per frame')
        if shared_total_bound_fractions is not None:method['settings']['shared_axial_length'].update(minimum_total_mm=shared.minimum_total, maximum_total_mm=shared.maximum_total, bound_fractions=shared_total_bound_fractions)
        method['objective'] = method['objective'].replace('Three-length proportion preference; total length remains free.',
            'One total axial length parameter per frame; segment lengths have exact prescribed proportions.')
    if equal_spine_lengths:
        method['objective'] += ' Soft equal-length preference between sacrolumbar and thoracic: one difference residual per frame; not a fixed sum or temporal smoothing.'
    if root_seed_fallbacks:
        method['settings']['initialization']+=' Missing saved root poses use the nearest available root pose within this interval as initialization only; no keypoint targets are added.'
    if relaxed:
        method['objective']=method['objective'].replace('Exact attachments;', 'Clavicle-to-upper-arm attachments have parent-local XYZ displacement parameters; all other attachments exact;')
        method['objective']+=' Shoulder displacements have zero-reference and local acceleration residuals; no hard displacement bounds. The shoulder keypoint remains assigned once to the acromion; arm keypoints influence the upper arm through the connected arm. No separately observed humeral joint center is invented.'
        for frame in frames:
            frame['observability']=frame['observability'].replace('Exact joints,', 'Exact joints except two explicitly relaxed shoulder linkages,')
    if free_axial_lengths:
        method['objective'] += ' Nonnegative spine lengths, no upper bound or length-acceleration residuals. Widths remain fixed.'
        method['objective'] += (' Existing rest-length residual enabled.' if free_length_rest_prior else ' Rest-length residual disabled.')
    if chest_line_prior:
        method['objective'] += ' Chest-center line preference: lateral and front/back distance, plus an extra anterior-only penalty. This is not independent measurement evidence.'
    if shoulder_profile:
        method['objective'] += ' Experimental fixed SC attachment geometry: '+m['shoulder_geometry']['label']+'. Saved person scale and rigid clavicle lengths retained.'
    method['ceres_full_report']=result.full_report
    if hasattr(result,'processing'):
        processing=result.processing;method['processing']=processing
        method['settings']['initialization']=processing['initialization']+(' Full refinement starts from every assembled window parameter.' if processing['refined'] else '')
        method['objective']+=' Sequential active windows with two fixed history frames.'
        if processing['refined']:method['objective']+=' Followed by one full-sequence optimization initialized from the windowed result.'
        else:method['objective']+=' Final assembled state is evaluated globally without another optimization.'
        method['settings']['processing']=dict(active_frames=processing['active_frames'],refined=processing['refined'],boundary_frames=processing['boundary_frames'],max_iterations=processing['max_iterations'],function_tolerance=processing['function_tolerance'])
        if processing.get('initial_function_tolerance') is not None:
            method['settings']['processing']['initial_function_tolerance']=processing['initial_function_tolerance']
        summary.update({'Window solve seconds':processing['window_solve_seconds'],
                        'Final pass solve seconds':processing['final_pass_seconds'],
                        'Processing wall seconds':processing['wall_seconds'],
                        'Windows':len(processing['windows']),
                        'Windows not converged':sum(not w['converged'] for w in processing['windows'])})
        for window in processing['windows']:
            for index in range(window['active_start'],window['committed_end']+1):
                frames[index]['diagnostics'].update({'Finalizing window':window['index'],
                    'Window active start':selected[window['active_start']]['number'],
                    'Window active end':selected[window['active_end']]['number'],
                    'Window fixed start':selected[window['fixed_start']]['number'],
                    'Window iterations':window['iterations'],'Window solve seconds':window['seconds']})
    provenance['method'] = method['objective']
    if digest(path) != provenance['sha256']:
        raise RuntimeError('Source recording changed during solve')
    print(result.report, summary, flush=True)
    return dict(id='recording_body', label='14 - Real recording / connected full body',
        description=f'Frames {start}-{end}: all {len(m["names"])} saved segments, including head, fingers and feet. {len(axial)} variable axial segment lengths. Source keypoints, mappings and person scale unchanged. Any experimental attachment geometry is explicitly listed in solver settings. Experimental fit, not production output.',
        controls=[], bodies=m['bodies'], methods=[dict(id='full_body', label='Connected full body / axial spine')],
        runs=[dict(parameters={}, times=times, length_series=True, methods=dict(full_body=method))],
        metadata=dict(recording=provenance, units='mm', quaternion_order='wxyz', ceres_version=_native.ceres_version,
            native_sha256=hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest()))


def publish_method(experiment, method_id, label, *, page=None):
    """Retain the existing comparison only when inputs and initialization match."""
    page = (OUTPUT_FOLDER / 'solver_viewer.html') if page is None else page
    bank = json.loads(re.search(r'const EXPERIMENTS\s*=\s*(\[.*?\]);', page.read_text(encoding='utf-8'), re.S).group(1))
    previous = next((e for e in bank if e['id']==experiment['id']), None)
    method = experiment['runs'][0]['methods']['full_body']
    method['metadata'] = experiment['metadata']
    experiment['runs'][0]['methods'] = {method_id: method}
    experiment['methods'] = [dict(id=method_id, label=label)]
    if previous:
        old = previous['runs'][0]
        if (previous['bodies'] != experiment['bodies'] or old['times'] != experiment['runs'][0]['times']
                or previous['metadata']['recording']['sha256'] != experiment['metadata']['recording']['sha256']):
            raise ValueError('Existing comparison has different model, timestamps or recording; refusing to mix runs')
        for saved_method in old['methods'].values():
            fixed_settings=('position_scale_mm','linear_motion_scale','angular_motion_scale',
                'rest_pose_scale_radians','rest_relative_quaternions','axial_reference_lengths',
                'length_acceleration_scale','direct_mapping_sources')
            for key in fixed_settings:
                if saved_method.get('settings',{}).get(key)!=method.get('settings',{}).get(key):
                    raise ValueError(f'Comparison changed another solver setting: {key}')
            for a,b in zip(saved_method['frames'],method['frames'],strict=True):
                for aa,bb in zip(a['bodies'],b['bodies'],strict=True):
                    if any(aa[key]!=bb[key] for key in ['observed','initial_quaternion','initial_translation']):
                        raise ValueError('Comparison targets or initialization changed; refusing to mix runs')
            # Add identical geometric review context to cached fits; never change their poses.
            line_settings = method.get('settings',{}).get('chest_line_prior')
            if line_settings:
                for old_frame,new_frame in zip(saved_method['frames'],method['frames'],strict=True):
                    line_diagnostics(old_frame, None if new_frame['chest_line'] is None else
                        {k:new_frame['chest_line'][k] for k in ('origin','shoulder','up','lateral','anterior')},
                        line_settings['body_index'],line_settings['landmark_index'])
            saved_method.setdefault('metadata', previous['metadata'])
        old['methods'][method_id] = method
        previous['methods'] = [m for m in previous['methods'] if m['id']!=method_id]+experiment['methods']
        for choice in previous['methods']:
            if choice['id']=='full_body':
                settings=old['methods']['full_body'].get('settings',{})
                if 'length_prior_fraction' in settings:
                    choice['label']=f'Baseline / shorten {settings["length_prior_fraction"]}, lengthen {settings.get("lengthening_prior_fraction",settings["length_prior_fraction"])}'
        experiment = previous
    render_experiments(bank=[e for e in bank if e['id'] != experiment['id']]+[experiment])


def main():
    prepare_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=180)
    parser.add_argument('--end', type=int, default=213)
    parser.add_argument('--parquet', type=Path)
    parser.add_argument('--variant', choices=['baseline','wider_symmetric','shortening_biased','free_lengths','chest_line'], default='baseline')
    parser.add_argument('--output-json', type=Path, help='Save a solve for later publication instead of updating the viewer')
    args = parser.parse_args()
    variants={
        'chest_line': ('chest_line', 'Free spine + chest-center line preference', fit_settings.LENGTH_PRIOR_FRACTION, fit_settings.LENGTH_PRIOR_FRACTION),
        'free_lengths': ('free_lengths', 'Free spine lengths / no length penalties', fit_settings.LENGTH_PRIOR_FRACTION, fit_settings.LENGTH_PRIOR_FRACTION),
        'baseline': ('full_body', 'Baseline symmetric', fit_settings.LENGTH_PRIOR_FRACTION, fit_settings.LENGTH_PRIOR_FRACTION),
        'wider_symmetric': ('wider_symmetric', 'Wider symmetric', fit_settings.WIDER_LENGTH_PRIOR_FRACTION, fit_settings.WIDER_LENGTH_PRIOR_FRACTION),
        'shortening_biased': ('shortening_biased', 'Shortening biased', fit_settings.SHORTENING_BIASED_COMPRESSION_FRACTION, fit_settings.SHORTENING_BIASED_EXTENSION_FRACTION),
    }
    method_id,label,shortening,lengthening=variants[args.variant]
    experiment = recording_body_catalog(args.parquet or recording_path(), args.start, args.end,
                                       length_prior_fraction=shortening, lengthening_prior_fraction=lengthening,
                                       free_axial_lengths=args.variant in ("free_lengths", "chest_line"),
                                       chest_line_prior=args.variant == "chest_line")
    add_context(experiment)
    if args.output_json:
        args.output_json.write_text(json.dumps(experiment),encoding='utf-8')
        print(f'Saved {args.variant}: {args.output_json}',flush=True)
        return
    publish_method(experiment, method_id, label if args.variant in ('free_lengths','chest_line') else f'{label} / shorten {shortening}, lengthen {lengthening}')


if __name__ == '__main__':
    main()
