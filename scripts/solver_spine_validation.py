"""Audit the accepted experiment on prepared test and sample recordings.

This reports geometric agreement and temporal variation, not anatomical truth.
The recordings are independently processed; common-time length differences also
include differences in reconstruction, filtering and person-scale estimation.
"""
import json
from pathlib import Path

import numpy as np
from scripts.recording_data import digest, recording_path
from scripts.solver_spine_positions import OUTPUT, LANDMARKS, audit

SEGMENTS = ('sacrolumbar', 'thoracic', 'cervical_spine')
ENDPOINTS = (('pelvis_origin','chest_center'), ('chest_center','neck_center'),
             ('neck_center','craniocervical_junction'))


def change_metrics(values, times):
    values=np.asarray(values,dtype=float);times=np.asarray(times,dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(times).all() or not (np.diff(times)>0).all():
        raise ValueError('Finite values and strictly increasing times required')
    changes=np.diff(values,axis=0)
    steps=np.abs(changes) if changes.ndim==1 else np.linalg.norm(changes,axis=1)
    rates=steps/np.diff(times)
    return dict(step_p95_mm=float(np.percentile(steps,95)),step_max_mm=float(max(steps)),
        rate_p95_mm_s=float(np.percentile(rates,95)),rate_max_mm_s=float(max(rates)),
        worst_interval=int(np.argmax(rates)))


def angular_rates(quaternions,times):
    q=np.asarray(quaternions,dtype=float)
    np.testing.assert_allclose(np.linalg.norm(q,axis=1),1.,atol=1e-6,rtol=0)
    dot=np.clip(np.abs(np.sum(q[1:]*q[:-1],axis=1)),0,1)
    return np.degrees(2*np.arccos(dot))/np.diff(times)


def evaluate(candidate):
    method=candidate['runs'][0]['methods']['full_body'];frames=method['frames']
    times=np.asarray([f['diagnostics']['Recording timestamp (s)'] for f in frames])
    records=[dict(number=f['diagnostics']['Recording frame'],
        points={k:np.asarray(v) for k,v in f['recording_context']['landmarks'].items()}) for f in frames]
    acceptance=audit(candidate,records)
    if not acceptance['passed']:raise ValueError('Axial displacement acceptance failed')
    settings=method['settings'];costs=settings['costs_by_family']
    total=sum(sum(v) if isinstance(v,list) else v for v in costs.values())
    np.testing.assert_allclose(total,frames[0]['costs'][-1],rtol=1e-9,atol=1e-6)
    result=dict(frames=len(frames),duration_s=float(times[-1]-times[0]),
        fps=float(1/np.median(np.diff(times))),source=candidate['metadata']['recording'],
        ceres_seconds=method['processing']['window_solve_seconds'],
        processing_wall_seconds=method['processing']['wall_seconds'],
        unconverged_windows=sum(not w['converged'] for w in method['processing']['windows']),
        axial_positions=acceptance,segments={},landmark_motion={},lengths=[])
    iterations=[w['iterations'] for w in method['processing']['windows']]
    result['iterations']=dict(median=float(np.median(iterations)),p95=float(np.percentile(iterations,95)),maximum=max(iterations))
    for name,(start,end) in zip(SEGMENTS,ENDPOINTS):
        b=next(i for i,v in enumerate(candidate['bodies']) if v['id']==name)
        lengths=np.asarray([f['diagnostics'][name+' length (mm)'] for f in frames]);reference=settings['axial_reference_lengths'][b]
        if not np.isfinite(lengths).all() or not (lengths>0).all():raise ValueError('Nonpositive or nonfinite spine length')
        source=np.asarray([np.linalg.norm(r['points'][end]-r['points'][start]) for r in records])
        rates=angular_rates([f['bodies'][b]['quaternion'] for f in frames],times)
        result['lengths'].append(lengths.tolist())
        result['segments'][name]=dict(reference_mm=reference,mean_mm=float(lengths.mean()),
            std_mm=float(lengths.std()),minimum_mm=float(lengths.min()),maximum_mm=float(lengths.max()),
            rest_deviation_rms_mm=float(np.sqrt(np.mean((lengths-reference)**2))),
            length_change=change_metrics(lengths,times),
            saved_endpoint_distance_std_mm=float(source.std()),saved_endpoint_distance_change=change_metrics(source,times),
            angular_rate_p95_deg_s=float(np.percentile(rates,95)),angular_rate_max_deg_s=float(max(rates)))
    for name in LANDMARKS:
        b=next(i for i,v in enumerate(candidate['bodies']) if name in v['landmark_names'])
        slot=candidate['bodies'][b]['landmark_names'].index(name)
        fitted=np.asarray([f['bodies'][b]['fitted'][slot] for f in frames])
        target=np.asarray([r['points'][name] for r in records])
        result['landmark_motion'][name]=dict(fitted=change_metrics(fitted,times),
            saved=change_metrics(target,times),fit_minus_saved=change_metrics(fitted-target,times))
    return result,times,settings,candidate['metadata']['native_sha256']


def main():
    reports={};series={};configs={};binaries={}
    for name,path in [('test',OUTPUT/'test/candidate.json'),('sample',OUTPUT/'candidate.json')]:
        candidate=json.loads(path.read_text())
        reports[name],times,configs[name],binaries[name]=evaluate(candidate)
        if reports[name]['source']['sha256']!=digest(recording_path(name)):
            raise ValueError(f'{name} prepared recording changed since fitting')
        series[name]=(times,np.asarray(reports[name].pop('lengths')))
        del candidate
    if binaries['test']!=binaries['sample']:raise ValueError('Fits use different native binaries')
    for key in ('landmark_position_priors','landmark_huber_scale_mm','length_prior_fraction',
                'lengthening_prior_fraction','free_axial_lengths','free_length_rest_prior',
                'position_scale_mm','linear_motion_scale','angular_motion_scale','rest_pose_scale_radians','processing'):
        if configs['test'][key]!=configs['sample'][key]:raise ValueError('Fit settings differ: '+key)
    for key in ('profile','lowering_fraction','forward_factor'):
        if configs['test']['shoulder_geometry'][key]!=configs['sample']['shoulder_geometry'][key]:
            raise ValueError('Shoulder configuration differs: '+key)
    for key in ('ratios','scale_mm'):
        if configs['test']['length_proportion_prior'][key]!=configs['sample']['length_proportion_prior'][key]:
            raise ValueError('Spine ratio configuration differs: '+key)
    # Evaluate at actual shared timestamps, without comparing different world bases.
    test_times,test_lengths=series['test'];sample_times,sample_lengths=series['sample']
    inside=(test_times>=sample_times[0])&(test_times<=sample_times[-1])
    comparison={}
    for i,name in enumerate(SEGMENTS):
        sampled=np.interp(test_times[inside],sample_times,sample_lengths[i])
        delta=test_lengths[i,inside]-sampled
        comparison[name]=dict(length_difference_rms_mm=float(np.sqrt(np.mean(delta**2))),
            maximum_difference_mm=float(np.max(np.abs(delta))))
    report=dict(recordings=reports,common_timestamp_length_comparison=comparison,
        interpretation='Variation includes real motion and processed-target variation; no ground-truth jitter threshold is assumed. Matching solver settings does not imply frame-rate invariance: three active frames cover different durations.')
    (OUTPUT/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
