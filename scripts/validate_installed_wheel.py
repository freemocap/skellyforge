"""Validate a wheel from an isolated interpreter, using prepared reference inputs.

prepare reads the selected local recording. run imports only installed packages;
invoke it with Python -I and a working directory outside the repository.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter


def prepare(target, recording='test'):
    import numpy as np
    from scripts.recording_data import read_recording, recording_path
    from skellyforge.tests.test_native_window_sequence import seeded
    from skellyforge.core.skeleton.fitting import fit_windows
    records,scale,meta=read_recording(recording_path(recording),include_model=True)
    synthetic,_=seeded()
    reference=fit_windows(synthetic)
    folder = Path('build/spine_positions')
    if recording == 'test':
        folder /= 'test'
    candidate=json.loads((folder/'candidate.json').read_text(encoding='utf-8'))
    assert candidate['metadata']['recording']['sha256']==meta['sha256']
    method=candidate['runs'][0]['methods']['full_body']
    payload=dict(skeleton=meta['skeleton'],saved_model=meta['model'],scales=scale.segment_scales,
        records=records,source_sha256=meta['sha256'],synthetic=synthetic,
        synthetic_reference=dict(quaternions=reference.quaternions,translations=reference.translations,lengths=reference.lengths),
        frames=method['frames'],segment_names=[b['id'] for b in candidate['bodies']])
    def serialize(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    target.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(target,'wt',encoding='utf-8') as stream:json.dump(payload,stream,default=serialize,allow_nan=False)


def run(inputs,report_path):
    import numpy as np
    import skellyforge
    from skellyforge import _native
    from skellyforge.core.skeleton.fitting import fit_windows,fit_human
    from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot
    prefix=Path(sys.prefix).resolve()
    for module in (skellyforge,_native):
        if not Path(module.__file__).resolve().is_relative_to(prefix):
            raise RuntimeError(f'Package came from outside isolated environment: {module.__file__}')
    with gzip.open(inputs,'rt',encoding='utf-8') as stream:data=json.load(stream)
    synthetic=fit_windows(data['synthetic'],inspect_windows=(0,2))
    for field,expected in data['synthetic_reference'].items():
        np.testing.assert_allclose(getattr(synthetic,field),expected,atol=1e-9,rtol=1e-12)
    for snapshot in synthetic.inspected_windows.values():
        problem=snapshot['final'];ids={p.id for p in problem.parameters}
        assert all(p.quantity and p.frame>=0 for p in problem.parameters)
        assert all(r.purpose and set(r.parameter_ids)<=ids for r in problem.residuals)
    records=data['records']
    for record in records:
        for key in ('points','keypoints','origins','rotations'):
            record[key]={name:None if value is None else np.asarray(value) for name,value in record[key].items()}
    started=perf_counter()
    fitted=fit_human(SkeletonSnapshot.from_dict(data['skeleton']).restore(),data['saved_model'],data['scales'],records,inspect_windows=(0,))
    elapsed=perf_counter()-started;result=fitted.sequence;frames=data['frames']
    assert fitted.model['names']==data['segment_names']
    comparisons={
        'quaternions':(result.quaternions,[[b['quaternion'] for b in f['bodies']] for f in frames]),
        'translations_mm':(result.translations,[[b['translation'] for b in f['bodies']] for f in frames]),
        'axial_lengths_mm':(np.asarray(result.lengths)[:,np.asarray(fitted.model['references'])>0],[f['lengths'] for f in frames]),
    }
    differences={key:float(np.max(np.abs(np.asarray(actual)-np.asarray(expected)))) for key,(actual,expected) in comparisons.items()}
    report=dict(python=sys.version,package=skellyforge.__file__,extension=_native.__file__,ceres=_native.ceres_version,
        source_sha256=data['source_sha256'],frames=len(records),synthetic_inspection_passed=True,
        maximum_absolute_differences=differences,native_seconds=result.seconds,wall_seconds=elapsed)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    for actual,expected in comparisons.values():np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-12)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--recording', choices=('test', 'sample'), default='test')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.inputs, args.recording)
    else:
        if args.report is None:parser.error('run requires --report')
        run(args.inputs,args.report)


if __name__=='__main__':main()
