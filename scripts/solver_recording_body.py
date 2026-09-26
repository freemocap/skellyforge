"""Full saved skeleton in the Ceres lab; source recording is read-only.

Keypoints are measurements; saved direct mappings hydrate landmark targets.
Every authored landmark remains in the model and in the fitted output. Targets
without their source keypoint in a frame contribute no measurement residual.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from skellyforge import _native
from scripts import solver_fit_settings as fit_settings
from scripts.solver_shoulder_offsets import reference_positions, shoulder_diagnostics
from scripts.solver_chest_line import (mapped_centerline, line_diagnostics, CHEST_LINE_DISTANCE_SCALE_MM, CHEST_LINE_ANTERIOR_SCALE_MM)
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from scripts.recording_data import read_recording, recording_path, digest
from scripts.solver_axial_geometry import axial_points
from scripts.solver_recording_context import add_context
from scripts.generate_solver_viewer import render_experiments


def body_model(skeleton, saved, scales, shoulder_profile=None):
    positions, shoulder_geometry = reference_positions(skeleton, scales, shoulder_profile)
    joints = {j.child.name: j for j in skeleton.joints.values()}
    roots = set(skeleton.segments) - set(joints)
    if len(roots) != 1:
        raise ValueError('Full-body review requires one connected root')
    names = [roots.pop()]
    while len(names) < len(skeleton.segments):
        children = [n for n in skeleton.segments if n not in names and joints[n].parent.name in names]
        if not children:
            raise ValueError('Disconnected or cyclic skeleton')
        names.extend(children)
    parents = [names.index(joints[n].parent.name) for n in names[1:]]
    attachments = [positions[joints[n].connect_at.name].tolist() for n in names[1:]]
    display_names = [[k for k, v in skeleton.landmarks.items() if v.segment == n] for n in names]
    display = [np.array([positions[k] for k in keys]) for n, keys in zip(names, display_names)]
    # A direct keypoint may also map to a child's origin. That is the same
    # connected point, not an independent measurement to count twice.
    candidates = {}
    for mapping in saved['mappings']:
        for name, entry in mapping['entries'].items():
            if name in skeleton.landmarks and isinstance(entry, str):
                source = (mapping['prefix'] or '') + entry
                candidates.setdefault(source, []).append(name)
    sources = {}
    for source, keys in candidates.items():
        key = max(keys, key=lambda k: (np.linalg.norm(skeleton.landmarks[k].local_position.array), k))
        owner = skeleton.landmarks[key].segment
        for other in keys:
            if other == key:
                continue
            other_owner = skeleton.landmarks[other].segment
            joint = joints.get(other_owner)
            if not (joint and joint.parent.name == owner and joint.connect_at.name == key
                    and np.allclose(skeleton.landmarks[other].local_position.array, 0)):
                raise ValueError(f'Ambiguous repeated keypoint mapping: {source}: {keys}')
        sources[key] = source
    targets = [[k for k in keys if k in sources] for keys in display_names]
    local = [[display[b][display_names[b].index(k)].tolist() for k in keys] for b, keys in enumerate(targets)]
    rest = saved['rest_pose']['orientations']
    relative = [[rest[n][c] for c in ('w', 'x', 'y', 'z')] for n in names[1:]]
    # Explicit review grouping by existing tree roots, never name-pattern inference.
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
    references = [0.] * len(names)
    for segment, endpoint in [('sacrolumbar', 'chest_center'), ('thoracic', 'neck_center')]:
        b = names.index(segment)
        references[b] = float(display[b][display_names[b].index(endpoint), 2])
        if references[b] <= 0:
            raise ValueError('Axial reference extent must be positive')
    return dict(names=names, parents=parents, attachments=attachments, display_names=display_names,
                display=display, targets=targets, sources=sources, local=local, relative=relative,
                references=references, bodies=bodies, shoulder_geometry=shoulder_geometry)


def frame_targets(record, model):
    observed, indices = [], []
    for keys in model['targets']:
        values, slots = [], []
        for j, key in enumerate(keys):
            source = model['sources'][key]
            if source not in record['keypoints']:
                continue
            if key not in record['points'] or not np.allclose(record['points'][key], record['keypoints'][source], atol=fit_settings.DIRECT_MAPPING_TOLERANCE_MM, rtol=0):
                raise ValueError(f'Frame {record["number"]}: direct mapping disagrees with keypoint {source} -> {key}')
            values.append(record['points'][key].tolist())
            slots.append(j)
        observed.append(values)
        indices.append(slots)
    return observed, indices


def recording_body_catalog(path, start=180, end=213, *, length_prior_fraction=fit_settings.LENGTH_PRIOR_FRACTION,
                           lengthening_prior_fraction=None, free_axial_lengths=False, chest_line_prior=False, shoulder_profile=None, relaxed_shoulders=False):
    records, scale, provenance = read_recording(path, include_model=True)
    saved = provenance.pop('model')
    skeleton = SkeletonSnapshot.from_dict(provenance.pop('skeleton')).restore()
    m = body_model(skeleton, saved, scale.segment_scales, shoulder_profile)
    selected = [r for r in records if start <= r['number'] <= end]
    if len(selected) != end-start+1 or len(selected) < 3:
        raise ValueError('Requested consecutive interval is not available')
    observed, indices, initial, roots = [], [], [], []
    seed_fallbacks = 0
    for record in selected:
        obs, slots = frame_targets(record, m)
        observed.append(obs); indices.append(slots)
        quaternions = []
        for b, name in enumerate(m['names']):
            q = record['rotations'].get(name)
            if q is None:
                if b == 0:
                    raise ValueError('No saved root pose for initialization')
                q = (Rotation.from_quat(quaternions[m['parents'][b-1]], scalar_first=True)
                     * Rotation.from_quat(m['relative'][b-1], scalar_first=True)).as_quat(scalar_first=True)
                seed_fallbacks += 1
            quaternions.append(np.asarray(q).tolist())
        initial.append(quaternions)
        roots.append(record['origins'][m['names'][0]].tolist())
    chest_body = next(b for b, keys in enumerate(m['display_names']) if 'chest_center' in keys)
    chest_slot = m['display_names'][chest_body].index('chest_center')
    line_frames = [mapped_centerline(r['keypoints'], m['sources']) for r in selected]
    prior = None
    if chest_line_prior:
        prior = _native.LandmarkLinePrior()
        prior.segment = chest_body
        prior.local_point = m['display'][chest_body][chest_slot].tolist()
        prior.frames = [None if f is None else [f['origin'],f['lateral'],f['anterior']] for f in line_frames]
        prior.distance_scale = CHEST_LINE_DISTANCE_SCALE_MM
        prior.anterior_scale = CHEST_LINE_ANTERIOR_SCALE_MM
    relaxed = [m['names'].index(side+'_upper_arm') for side in ('left','right')] if relaxed_shoulders else []
    for child in relaxed:
        if m['names'][m['parents'][child-1]] not in ('left_clavicle','right_clavicle'):
            raise ValueError('Shoulder experiment requires authored clavicle-to-upper-arm linkages')
    times = [r['time']-selected[0]['time'] for r in selected]
    print(f'Fitting {len(m["names"])} segments / {len(selected)} frames / {sum(len(v) for f in observed for v in f)} mapped keypoint targets', flush=True)
    result = _native.fit_chain_sequence(
        relaxed_linkage_children=relaxed, linkage_scale=fit_settings.SHOULDER_LINKAGE_SCALE_MM,
        linkage_acceleration_scale=fit_settings.SHOULDER_LINKAGE_ACCELERATION_SCALE_MM_S2,
        landmark_line_prior=prior, length_prior_fraction=length_prior_fraction,
        lengthening_prior_fraction=lengthening_prior_fraction, free_axial_lengths=free_axial_lengths,
        length_acceleration_scale=fit_settings.LENGTH_ACCELERATION_SCALE_MM_S2,
        local=m['local'], observed=observed, observation_indices=indices,
        parent_attachments=m['attachments'], child_attachments=[[0.,0.,0.]]*len(m['parents']), parent_indices=m['parents'],
        times=times, position_scale=fit_settings.POSITION_RESIDUAL_SCALE_MM, linear_acceleration_scale=fit_settings.ROOT_ACCELERATION_SCALE_MM_S2, angular_acceleration_scale=fit_settings.ANGULAR_ACCELERATION_SCALE_RAD_S2,
        initial_quaternions=initial, initial_roots=roots, rest_relative_quaternions=m['relative'], rest_pose_scale=fit_settings.REST_POSE_RESIDUAL_SCALE_RAD,
        axial_reference_lengths=m['references'])
    axial = [b for b, ref in enumerate(m['references']) if ref]
    frames = []
    attachment_errors = []
    linkage_separations = []
    for i, record in enumerate(selected):
        bodies = []
        linkages = []
        diagnostics = {'Recording frame': record['number'], 'Recording timestamp (s)': record['time']}
        for b, name in enumerate(m['names']):
            q, t = result.quaternions[i][b], result.translations[i][b]
            iq, it = result.initial_quaternions[i][b], result.initial_translations[i][b]
            ref, length = m['references'][b], result.lengths[i][b]
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
                expected = np.array(result.translations[i][parent])+Rotation.from_quat(result.quaternions[i][parent], scalar_first=True).apply(
                    axial_points(m['attachments'][b-1], m['references'][parent], result.lengths[i][parent]))
                delta=np.asarray(result.linkage_displacements[i][b])
                if b in relaxed:
                    separation=float(np.linalg.norm(delta));linkage_separations.append(separation)
                    diagnostics[name+' linkage separation (mm)']=separation
                    linkages.append(dict(child=b,parent=parent,local_displacement=delta.tolist(),parent_point=expected.tolist(),child_point=list(t)))
                displaced=expected+Rotation.from_quat(result.quaternions[i][parent],scalar_first=True).apply(delta)
                attachment_errors.append(float(np.linalg.norm(displaced-t)))
            bodies.append(dict(axial_scale=length/ref if ref else 1., mechanical_model='axial' if ref else 'rigid',
                target_count=len(support), target_rank=rank, local=local.tolist(), truth=None, observed=obs,
                fitted=fitted.tolist(), initial=starting.tolist(), quaternion=q, translation=t,
                initial_quaternion=iq, initial_translation=it, reference_quaternion=None, reference_translation=None,
                residuals=errors, truth_rms=None))
        spine_axes = [Rotation.from_quat(result.quaternions[i][b], scalar_first=True).apply([0,0,1]) for b in axial]
        diagnostics['Spine bend (degrees)'] = float(np.degrees(np.arccos(np.clip(np.dot(*spine_axes),-1,1))))
        diagnostics['Unavailable source keypoints for selected mappings'] = len(m['sources'])-sum(map(len,observed[i]))
        for b in axial:
            diagnostics[m['names'][b]+' length (mm)'] = result.lengths[i][b]
        frames.append(dict(linkages=linkages, time=times[i], bodies=bodies, root=result.roots[i], costs=result.costs,
            lengths=[result.lengths[i][b] for b in axial], reference_lengths=[m['references'][b] for b in axial],
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
    summary['Frames at axial length bounds'] = sum(any(abs(result.lengths[i][b]-bound*m['references'][b])<fit_settings.BOUND_CONTACT_TOLERANCE_MM
        for b in axial for bound in ((0.,) if free_axial_lengths else fit_settings.AXIAL_LENGTH_BOUND_FRACTIONS)) for i in range(len(frames)))
    for region in sorted({b['region'] for b in m['bodies']}):
        values = [e for f in frames for b,definition in zip(f['bodies'],m['bodies']) if definition['region']==region for e in b['residuals'] if e is not None]
        summary[region+' target RMS (mm)'] = float(np.sqrt(np.mean(np.square(values)))) if values else 'No direct targets'
    method = dict(frames=frames, summary=summary, problem=dict(chain=True, parents=m['parents'], connected=True,
        temporal=True, acceleration=True, rest_prior=True, axial_segments=axial, **(dict(relaxed_linkage_children=relaxed) if relaxed else {})), settings=dict(
        relaxed_linkage_children=relaxed, linkage_scale_mm=fit_settings.SHOULDER_LINKAGE_SCALE_MM,
        linkage_acceleration_scale_mm_s2=fit_settings.SHOULDER_LINKAGE_ACCELERATION_SCALE_MM_S2,
        shoulder_geometry=m['shoulder_geometry'],
        chest_line_prior=dict(enabled=chest_line_prior, body_index=chest_body, landmark_index=chest_slot,
            landmark='chest_center', distance_scale_mm=CHEST_LINE_DISTANCE_SCALE_MM,
            anterior_scale_mm=CHEST_LINE_ANTERIOR_SCALE_MM,
            frame_definition='Hip center to shoulder center: up; cross(up, left-to-right hip vector): anterior; cross(anterior, up): lateral.'),
        free_axial_lengths=free_axial_lengths,
        axial_length_bound_fractions=(0., None) if free_axial_lengths else fit_settings.AXIAL_LENGTH_BOUND_FRACTIONS, axial_reference_lengths=m['references'], length_prior_fraction=length_prior_fraction,
        lengthening_prior_fraction=length_prior_fraction if lengthening_prior_fraction is None else lengthening_prior_fraction,
        length_acceleration_scale=fit_settings.LENGTH_ACCELERATION_SCALE_MM_S2,
        position_scale_mm=fit_settings.POSITION_RESIDUAL_SCALE_MM, linear_motion_scale=fit_settings.ROOT_ACCELERATION_SCALE_MM_S2, angular_motion_scale=fit_settings.ANGULAR_ACCELERATION_SCALE_RAD_S2, scale_units=['mm/s^2','rad/s^2'],
        rest_pose_scale_radians=fit_settings.REST_POSE_RESIDUAL_SCALE_RAD, rest_relative_quaternions=m['relative'], direct_mapping_sources=m['sources'],
        initialization='Saved segment quaternions and root origin. Where a saved quaternion is unavailable: parent seed composed with authored relative T-pose. Initialization is not a measurement residual.',
        costs_by_family=dict(linkage_prior=result.linkage_prior_cost, linkage_acceleration=result.linkage_acceleration_cost, landmarks=result.landmark_cost, relative_pose=result.relative_pose_cost,
            root_acceleration=result.root_acceleration_cost, segment_angular_acceleration=result.angular_acceleration_costs,
            chest_line_prior=result.line_prior_cost, length_prior=result.length_prior_cost, length_acceleration=result.length_acceleration_cost)),
        objective='One connected full-body Ceres problem. Direct mapped keypoint targets counted once; no derived-landmark measurement residuals. Exact attachments; sacrolumbar and thoracic axial lengths; all other geometry rigid. Quaternion rest-pose and temporal residuals are preferences, not joint limits.')
    if relaxed:
        method['objective']=method['objective'].replace('Exact attachments;', 'Clavicle-to-upper-arm attachments have parent-local XYZ displacement parameters; all other attachments exact;')
        method['objective']+=' Shoulder displacements have zero-reference and local acceleration residuals; no hard displacement bounds. The shoulder keypoint remains assigned once to the acromion; arm keypoints influence the upper arm through the connected arm. No separately observed humeral joint center is invented.'
        for frame in frames:
            frame['observability']=frame['observability'].replace('Exact joints,', 'Exact joints except two explicitly relaxed shoulder linkages,')
    if free_axial_lengths:
        method['objective'] += ' Free-length diagnostic: nonnegative lengths, no upper bound, no length-prior or length-acceleration residuals. Widths remain fixed.'
    if chest_line_prior:
        method['objective'] += ' Chest-center line preference: lateral and front/back distance, plus an extra anterior-only penalty. This is not independent measurement evidence.'
    if shoulder_profile:
        method['objective'] += ' Experimental fixed SC attachment geometry: '+m['shoulder_geometry']['label']+'. Saved person scale and rigid clavicle lengths retained.'
    provenance['method'] = method['objective']
    if digest(path) != provenance['sha256']:
        raise RuntimeError('Source recording changed during solve')
    print(result.report, summary, flush=True)
    return dict(id='recording_body', label='14 - Real recording / connected full body',
        description=f'Frames {start}-{end}: all {len(m["names"])} saved segments, including head, fingers and feet. Two axial spine lengths. Source keypoints, mappings and person scale unchanged. Any experimental attachment geometry is explicitly listed in solver settings. Experimental fit, not production output.',
        controls=[], bodies=m['bodies'], methods=[dict(id='full_body', label='Connected full body / axial spine')],
        runs=[dict(parameters={}, times=times, length_series=True, methods=dict(full_body=method))],
        metadata=dict(recording=provenance, units='mm', quaternion_order='wxyz', ceres_version=_native.ceres_version,
            native_sha256=hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest()))


def publish_method(experiment, method_id, label, *, page=None):
    """Retain the existing comparison only when inputs and initialization match."""
    page = Path(__file__).with_name('solver_viewer.html') if page is None else page
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
