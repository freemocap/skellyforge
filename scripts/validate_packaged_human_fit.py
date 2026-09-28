"""Compare the packaged human fit with a saved accepted viewer result.

Reads prepared recordings and the baseline; writes only a diagnostic JSON report.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge.core.skeleton.fitting import fit_human
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
from scripts.recording_data import recording_path, read_recording


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',choices=('test','sample'),required=True)
    args=parser.parse_args()
    folder=Path('build/spine_positions') / ('test' if args.recording=='test' else '')
    baseline=json.loads((folder/'candidate.json').read_text())
    records,scale,meta=read_recording(recording_path(args.recording),include_model=True)
    if meta['sha256']!=baseline['metadata']['recording']['sha256']:
        raise ValueError('Baseline and prepared recording differ')
    skeleton=SkeletonSnapshot.from_dict(meta['skeleton']).restore()
    started=perf_counter()
    result=fit_human(skeleton,meta['model'],scale.segment_scales,records)
    elapsed=perf_counter()-started
    fit=result.sequence;model=result.model
    frames=baseline['runs'][0]['methods']['full_body']['frames']
    if len(frames)!=len(records):raise ValueError('Baseline frame count differs')
    if model['names']!=[b['id'] for b in baseline['bodies']]:raise ValueError('Segment order differs')
    q=np.asarray(fit.quaternions);t=np.asarray(fit.translations);lengths=np.asarray(fit.lengths)
    comparisons={
      'quaternion_components':(q,np.asarray([[b['quaternion'] for b in f['bodies']] for f in frames])),
      'translations_mm':(t,np.asarray([[b['translation'] for b in f['bodies']] for f in frames])),
      'axial_lengths_mm':(lengths[:,np.asarray(model['references'])>0],np.asarray([f['lengths'] for f in frames])),
    }
    fitted=[];old=[]
    for i,frame in enumerate(frames):
        for b,body in enumerate(frame['bodies']):
            local=np.array(model['display'][b],copy=True)
            if model['references'][b]>0:local[:,2]*=lengths[i,b]/model['references'][b]
            fitted.extend(Rotation.from_quat(q[i,b],scalar_first=True).apply(local)+t[i,b])
            old.extend(body['fitted'])
    comparisons['fitted_landmarks_mm']=(np.asarray(fitted),np.asarray(old))
    differences={key:float(np.max(np.abs(a-b))) for key,(a,b) in comparisons.items()}
    attrs=dict(landmark_position_prior='position_prior_cost',sc_anterior='half_space_cost',relative_twist='twist_prior_cost',axis_prior='axis_prior_cost',length_proportion='length_proportion_cost',length_equality='length_equality_cost',linkage_prior='linkage_prior_cost',linkage_acceleration='linkage_acceleration_cost',landmarks='landmark_cost',relative_pose='relative_pose_cost',root_acceleration='root_acceleration_cost',segment_angular_acceleration='angular_acceleration_costs',chest_line_prior='line_prior_cost',length_prior='length_prior_cost',length_acceleration='length_acceleration_cost')
    previous_costs=baseline["runs"][0]["methods"]["full_body"]["settings"]["costs_by_family"]
    cost_differences={key:float(np.max(np.abs(np.asarray(getattr(fit,attr))-np.asarray(previous_costs[key])))) for key,attr in attrs.items()}
    report=dict(recording=args.recording,frames=len(records),source_sha256=meta['sha256'],
        maximum_absolute_difference=differences,exact_match=all(v==0 for v in differences.values()),
        native_seconds=fit.seconds,wall_seconds=elapsed,
        previous_native_seconds=baseline['runs'][0]['methods']['full_body']['processing']['window_solve_seconds'],
        converged=fit.converged, residual_family_cost_max_differences=cost_differences)
    target=Path('build/package_extraction');target.mkdir(exist_ok=True)
    (target/(args.recording+'.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if max(cost_differences.values())>1e-9:raise AssertionError('Residual family costs changed')
    if not report['exact_match']:raise AssertionError('Packaged fit changed saved numerical results')


if __name__=='__main__':main()
