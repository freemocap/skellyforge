"""Read-only Ceres torso review using saved model definitions and four targets."""
import argparse
import json
import re
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge import _native
from scripts import solver_fit_settings as fit_settings
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from skellyforge.core.math.geometry.rotation_quaternion import RotationQuaternion
from scripts.recording_data import recording_path, read_recording, digest
from scripts.solver_torso_experiment import model, initialization, NAMES, TARGETS
from scripts.generate_solver_viewer import render_experiments
from scripts.solver_axial_geometry import axial_references,axial_points


def recording_catalog(path, *, flexible=False):
    records,scale,provenance=read_recording(path,include_model=True)
    saved=provenance.pop('model');skeleton=SkeletonSnapshot.from_dict(provenance.pop('skeleton')).restore()
    joints={j.child.name:j for j in skeleton.joints.values()}
    rest=SimpleNamespace(parents={n:joints[n].parent.name for n in NAMES[1:]},
        connect_ats={n:joints[n].connect_at.name for n in NAMES[1:]},
        relative_orientations={n:RotationQuaternion(**q) for n,q in saved['rest_pose']['orientations'].items()})
    m=model(skeleton,rest,scale.segment_scales)
    targets={n for names in TARGETS for n in names};sources={}
    for mapping in saved['mappings']:
        for name,entry in mapping['entries'].items():
            if name not in targets:continue
            if name in sources or not isinstance(entry,str):raise ValueError('Expected unique direct mapping for '+name)
            sources[name]=(mapping['prefix'] or '')+entry
    if set(sources)!=targets or len(set(sources.values()))!=4:raise ValueError('Expected four unique direct target mappings')
    selected=[];omitted=[];observed=[]
    for record in records:
        if any(n not in record['points'] or not np.isfinite(record['points'][n]).all() for n in targets):
            omitted.append(record['number']);continue
        for name,source in sources.items():
            if source not in record['keypoints'] or not np.allclose(record['points'][name],record['keypoints'][source],atol=fit_settings.DIRECT_MAPPING_TOLERANCE_MM,rtol=0):
                raise ValueError('Saved target disagrees with source keypoint: '+name)
        selected.append(record)
        observed.append([[record['points'][n].tolist() for n in names] for names in TARGETS])
    if len(selected)<3 or any(b['number']!=a['number']+1 for a,b in zip(selected,selected[1:])):
        raise ValueError('Review requires a consecutive complete-target interval; no gap interpolation')
    references=axial_references(m)
    initial,roots=initialization(m,observed)
    times=np.array([r['time'] for r in selected]);times-=times[0]
    result=_native.fit_chain_sequence(
        length_prior_fraction=fit_settings.LENGTH_PRIOR_FRACTION,
        length_acceleration_scale=fit_settings.LENGTH_ACCELERATION_SCALE_MM_S2,
        local=m['local'],observed=observed,parent_attachments=m['parent_attachments'],child_attachments=m['child_attachments'],parent_indices=m['parents'],times=times.tolist(),
        position_scale=fit_settings.POSITION_RESIDUAL_SCALE_MM,linear_acceleration_scale=fit_settings.ROOT_ACCELERATION_SCALE_MM_S2,angular_acceleration_scale=fit_settings.ANGULAR_ACCELERATION_SCALE_RAD_S2,initial_quaternions=initial,initial_roots=roots,rest_relative_quaternions=m['relative'],rest_pose_scale=fit_settings.REST_POSE_RESIDUAL_SCALE_RAD,axial_reference_lengths=references if flexible else [])
    frames=[]
    for i,record in enumerate(selected):
        bodies=[];diagnostics={'Recording frame':record['number'],'Recording timestamp (s)':record['time']}
        for b,n in enumerate(NAMES):
            q,t=result.quaternions[i][b],result.translations[i][b]
            iq,it=result.initial_quaternions[i][b],result.initial_translations[i][b]
            length=result.lengths[i][b] if flexible else references[b]
            local=axial_points(m['display'][b],references[b],length)
            fitted=Rotation.from_quat(q,scalar_first=True).apply(local)+t
            initial_points=Rotation.from_quat(iq,scalar_first=True).apply(m['display'][b])+it
            obs=[observed[i][b][TARGETS[b].index(k)] if k in TARGETS[b] else None for k in m['display_names'][b]]
            errors=[float(np.linalg.norm(fitted[j]-point)) if point is not None else None for j,point in enumerate(obs)]
            if b:
                parent=m['parents'][b-1]
                expected=np.array(result.translations[i][parent])+Rotation.from_quat(result.quaternions[i][parent],scalar_first=True).apply(axial_points(m['parent_attachments'][b-1],references[parent],result.lengths[i][parent] if flexible else references[parent]))
                diagnostics[n+' attachment error (mm)']=float(np.linalg.norm(expected-t))
            bodies.append(dict(axial_scale=length/references[b] if references[b] else 1.,local=local.tolist(),truth=None,observed=obs,fitted=fitted.tolist(),initial=initial_points.tolist(),quaternion=q,translation=t,
                initial_quaternion=iq,initial_translation=it,reference_quaternion=None,reference_translation=None,residuals=errors,truth_rms=None))
        spine_a=Rotation.from_quat(result.quaternions[i][1],scalar_first=True).apply([0,0,1])
        spine_b=Rotation.from_quat(result.quaternions[i][2],scalar_first=True).apply([0,0,1])
        diagnostics['Spine bend (degrees)']=float(np.degrees(np.arccos(np.clip(np.dot(spine_a,spine_b),-1,1))))
        for b in [1,2]:diagnostics[NAMES[b]+' length (mm)']=result.lengths[i][b] if flexible else references[b]
        frames.append(dict(lengths=[result.lengths[i][b] if flexible else references[b] for b in [1,2]],reference_lengths=[references[b] for b in [1,2]],time=float(times[i]),bodies=bodies,root=result.roots[i],costs=result.costs,converged=result.converged,seconds=result.seconds,report=result.report,diagnostics=diagnostics,
            observability='Real recording: no known segment poses. Four observed targets only; internal spine poses and clavicle roll depend on the model and residual preferences.'))
    errors=[e for f in frames for b in f['bodies'] for e in b['residuals'] if e is not None]
    summary={'Ceres parameter blocks':result.parameter_blocks,'Ceres residual blocks':result.residual_blocks,'Observed landmark RMS (mm)':float(np.sqrt(np.mean(np.square(errors)))),
        'Observed error 95th percentile (mm)':float(np.percentile(errors,95)),'Maximum observed error (mm)':max(errors),'Omitted incomplete frames':str(omitted)}
    for b,names in enumerate(TARGETS):
        for name in names:
            j=m['display_names'][b].index(name)
            summary[name+' RMS (mm)']=float(np.sqrt(np.mean([f['bodies'][b]['residuals'][j]**2 for f in frames])))
    if flexible:
        summary['Frames at length bounds']=sum(any(abs(result.lengths[i][b]-bound*references[b])<fit_settings.BOUND_CONTACT_TOLERANCE_MM for b in [1,2] for bound in fit_settings.AXIAL_LENGTH_BOUND_FRACTIONS) for i in range(len(frames)))
    method=dict(frames=frames,summary=summary,problem=dict(chain=True,parents=m['parents'],connected=True,temporal=True,acceleration=True,rest_prior=True,axial_segments=[1,2] if flexible else []),
        settings=dict(axial_length_bound_fractions=fit_settings.AXIAL_LENGTH_BOUND_FRACTIONS,axial_reference_lengths=references,length_prior_fraction=fit_settings.LENGTH_PRIOR_FRACTION,length_acceleration_scale=fit_settings.LENGTH_ACCELERATION_SCALE_MM_S2,position_scale_mm=fit_settings.POSITION_RESIDUAL_SCALE_MM,linear_motion_scale=fit_settings.ROOT_ACCELERATION_SCALE_MM_S2,angular_motion_scale=fit_settings.ANGULAR_ACCELERATION_SCALE_RAD_S2,scale_units=['mm/s^2','rad/s^2'],rest_pose_scale_radians=fit_settings.REST_POSE_RESIDUAL_SCALE_RAD,rest_relative_quaternions=m['relative'],segment_scales_mm={n:scale.segment_scales[n] for n in NAMES},
            costs_by_family=dict(length_prior=result.length_prior_cost,length_acceleration=result.length_acceleration_cost,landmarks=result.landmark_cost,relative_pose=result.relative_pose_cost,root_acceleration=result.root_acceleration_cost,segment_angular_acceleration=result.angular_acceleration_costs),
            initialization='Four observed targets plus saved T-pose; saved segment poses are not used.'),objective=('Axial spine model: local Z scales by length/reference, preserving local X/Y and exact deformed attachments. Length priors and acceleration residuals are experimental preferences.' if flexible else 'Rigid four-target Ceres model with rest-pose residuals and saved model scale. No ground-truth errors are available.'))
    if digest(Path(path))!=provenance['sha256']:raise RuntimeError('Source recording changed during solve')
    provenance['method']='Experimental Ceres rigid torso fit from four saved direct landmarks. Source data unchanged; no saved segment poses supplied to solver.'
    return dict(id='recording_torso',label='12 - Real recording / rigid torso',description=f"Prepared recording frames {selected[0]['number']}?{selected[-1]['number']}. Saved skeleton, rest pose and scale. Actual timestamps; no reference motion. Omitted incomplete frames: {omitted}.",
        controls=[],bodies=m['bodies'],methods=[dict(id='rest_prior',label='With T-pose residuals')],runs=[dict(length_series=True,parameters={},times=times.tolist(),methods=dict(rest_prior=method))],
        metadata=dict(schema_version=1,units='mm',quaternion_order='wxyz',ceres_version=_native.ceres_version,native_sha256=hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),recording=provenance))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parquet',type=Path);args=parser.parse_args()
    page=Path(__file__).with_name('solver_viewer.html')
    if not page.exists():raise FileNotFoundError('Generate the synthetic solver viewer first')
    bank=json.loads(re.search(r'const EXPERIMENTS\s*=\s*(\[.*?\]);',page.read_text(encoding='utf-8'),re.S).group(1))
    path=args.parquet or recording_path()
    experiment=recording_catalog(path)
    flex=recording_catalog(path,flexible=True)
    experiment['runs'][0]['methods']['flexible']=flex['runs'][0]['methods']['rest_prior']
    experiment['methods']=[dict(id='rest_prior',label='Rigid spine'),dict(id='flexible',label='Flexible axial spine')]
    experiment['label']='12 - Real recording / rigid vs flexible spine'
    experiment['description']+=' Compare fixed geometry against explicit local-Z spine lengths; neither model supplies ground truth.'
    experiment['metadata']['recording']['method']='Experimental Ceres comparison: fixed geometry versus explicit axial sacrolumbar/thoracic lengths; same four observations, T-pose, initialization and timestamps.'
    from scripts.solver_recording_context import add_context
    add_context(experiment)
    for name,method in experiment['runs'][0]['methods'].items():print(name,method['frames'][0]['report'],method['summary'])
    render_experiments(bank=[e for e in bank if e['id']!='recording_torso']+[experiment])

if __name__=='__main__':main()
