"""Benchmark saved 3/5/7-frame Ceres windows and optional warm global refinement."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from scripts.recording_data import recording_path,read_recording
from scripts.solver_recording_body import recording_body_catalog
from scripts.solver_recording_context import add_context
from scripts.solver_window_sequence import fit_windows,refine_window_result,DEFAULT_MAX_ITERATIONS
from scripts.solver_shoulder_offsets import shoulder_diagnostics
from scripts.generate_recording_comparison import comparison_data,write_comparison,FOLDER

RECORDING_FUNCTION_TOLERANCE = 1e-6


def publish(candidates, *, comparison='processing'):
    pages={'processing':'recording_processing_comparison.html', 'ratios':'recording_spine_ratio_comparison.html',
           'shoulders':'recording_sc_offset_comparison.html', 'rest_lengths':'recording_rest_length_comparison.html',
           'performance':'recording_sample_performance.html', 'shared_length':'recording_shared_spine_length.html', 'axial_axes':'recording_axial_axes.html', 'spine_stability':'recording_spine_stability.html', 'sc_anterior':'recording_sc_anterior.html'}
    pages['centerline']='recording_centerline.html'
    pages['length_coupling']='recording_length_coupling.html'
    pages['fixed_neck']='recording_fixed_neck.html'
    if comparison not in pages:raise ValueError('Unknown comparison type')
    ratio_comparison=comparison=='ratios'
    shoulder_comparison=comparison in ('shoulders','rest_lengths')
    first=next(iter(candidates.values()))
    experiment={**first,'id':'recording_processing','methods':[],
                'runs':[{**first['runs'][0],'methods':{}}]}
    for name,candidate in candidates.items():
        shoulder_diagnostics(candidate['runs'][0]['methods']['full_body']['frames'],candidate['bodies'])
        body_schema=lambda e:[(b['id'],b['landmark_names'],b['region'],b['edges']) for b in e['bodies']]
        same_geometry=body_schema(candidate)==body_schema(first) if shoulder_comparison else candidate['bodies']==first['bodies']
        if not same_geometry or candidate['runs'][0]['times']!=first['runs'][0]['times']:
            raise ValueError('Processing comparison geometry or frames differ')
        if candidate['metadata']['recording']['sha256']!=first['metadata']['recording']['sha256']:
            raise ValueError('Comparison source recordings differ')
        method=candidate['runs'][0]['methods']['full_body'];p=method['processing']
        # Cached JSON uses lists where fresh Python settings may use tuples.
        method['settings']=json.loads(json.dumps(method['settings']))
        method['settings'].setdefault('free_length_rest_prior',False)
        reference=first['runs'][0]['methods']['full_body']
        if (ratio_comparison or shoulder_comparison) and method['settings']['processing']!=reference['settings']['processing']:
            raise ValueError('Ratio comparison changed processing settings')
        if comparison in ('shared_length','axial_axes','spine_stability','sc_anterior') and method['settings']['processing']!=reference['settings']['processing']:
            raise ValueError('Shared length comparison changed processing settings')
        for key,value in reference['settings'].items():
            if comparison=='length_coupling' and key in ('length_prior_fraction','lengthening_prior_fraction','shared_axial_length'):
                continue
            if comparison=='length_coupling' and key=='length_proportion_prior':
                if any(method['settings'][key].get(k)!=v for k,v in value.items() if k!='enabled'):
                    raise ValueError('Length coupling comparison changed proportion definition')
                continue
            if comparison=='centerline' and key=='chest_line_prior':
                if any(method['settings'][key].get(k)!=v for k,v in value.items() if k!='distance_scale_mm'):
                    raise ValueError('Centerline comparison changed another line setting')
                continue
            if comparison=='sc_anterior' and key=='sc_anterior_prior':continue
            if comparison=='spine_stability' and key in ('relative_twist_priors','length_prior_fraction','lengthening_prior_fraction'):continue
            if comparison=='spine_stability' and key=='shared_axial_length':
                if any(method['settings'][key].get(k)!=v for k,v in value.items() if k not in ('minimum_total_mm','maximum_total_mm','bound_fractions')):raise ValueError('Spine comparison changed shared ratios')
                continue
            if comparison=='axial_axes' and key=='segment_axis_prior':continue
            if comparison=='shared_length' and key=='shared_axial_length':continue
            if comparison=='shared_length' and key=='length_proportion_prior':
                if any(method['settings'][key].get(k)!=v for k,v in value.items() if k!='enabled'):
                    raise ValueError('Shared length comparison changed proportions')
                continue
            if comparison=='rest_lengths' and key in ('free_length_rest_prior','length_prior_fraction','lengthening_prior_fraction'):continue
            if shoulder_comparison and key=='shoulder_geometry':
                actual=method['settings'][key]
                for fixed in ('lowering_fraction','thoracic_reference_length_mm','saved_segment_scales','original_local_positions_mm'):
                    if actual[fixed]!=value[fixed]:raise ValueError('Shoulder comparison changed '+fixed)
                for landmark,position in value['experimental_local_positions_mm'].items():
                    target=actual['experimental_local_positions_mm'][landmark]
                    if target[0]!=position[0] or target[2]!=position[2]:raise ValueError('Shoulder comparison changed lateral or axial offsets')
                continue
            if ratio_comparison and key=='length_proportion_prior':
                actual=method['settings'][key]
                if any(actual.get(k)!=v for k,v in value.items() if k not in ('ratios','fractions')):
                    raise ValueError('Ratio comparison changed another proportion-prior setting')
                continue
            if key not in ('processing','initialization','costs_by_family') and method['settings'].get(key)!=value:
                raise ValueError('Processing comparison changed model setting: '+key)
        label=f'{p["active_frames"]} active frames'+(' + full refinement' if p['refined'] else ' / sequential')
        if comparison in ('spine_stability','sc_anterior','centerline','length_coupling','fixed_neck'):label=method['comparison_label']
        if comparison=='shared_length':
            label='One shared spine length' if method['settings'].get('shared_axial_length') else 'Three independent spine lengths'
        if comparison=='axial_axes':
            label='Shared length + SC/shoulder axis preference' if method['settings'].get('segment_axis_prior') else 'Shared length baseline'
        if comparison=='performance':
            label+=f' / tolerance {p["function_tolerance"]:g}'
            if p.get('initial_function_tolerance') is not None:
                label+=f' / first window {p["initial_function_tolerance"]:g}'
        if shoulder_comparison:
            ratios=':'.join(f'{v:g}' for v in method['settings']['length_proportion_prior']['ratios'])
            label=f"{ratios} / SC forward {100*method['settings']['shoulder_geometry']['forward_factor']:g}%"
        if ratio_comparison:label=' : '.join(f'{v:g}' for v in method['settings']['length_proportion_prior']['ratios'])+' / lumbar : thoracic : cervical'
        if comparison=='rest_lengths':
            scale=method['settings']['length_prior_fraction']
            label+=f' / rest scale {scale:g}' if method['settings']['free_length_rest_prior'] else ' / no rest prior'
        method['body_definitions']=candidate['bodies'];method['metadata']=candidate['metadata']
        experiment['methods'].append(dict(id=name,label=label))
        experiment['runs'][0]['methods'][name]=method
    write_comparison(comparison_data([experiment]),pages[comparison],
        '<a href="recording_spine_ratio_comparison.html">Spine ratio comparison</a>' if shoulder_comparison else
        '<a href="recording_processing_comparison.html">Window comparison</a>' if ratio_comparison else
        '<a href="recording_comparison.html">Previous full-recording fit</a>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--end',type=int)
    parser.add_argument('--windows',nargs='+',type=int,default=[3])
    parser.add_argument('--refine',action='store_true',help='One full-sequence refinement of the first window result')
    parser.add_argument('--publish-only',action='store_true',help='Rebuild the viewer from saved comparison JSON; do not fit again')
    parser.add_argument('--max-iterations',type=int,default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument('--function-tolerance',type=float,default=RECORDING_FUNCTION_TOLERANCE)
    args=parser.parse_args()
    path=recording_path();records,_,_=read_recording(path)
    end=records[-1]['number'] if args.end is None else args.end
    folder=FOLDER.parent/'build'/'processing_comparison'/('full' if args.start==0 and end==records[-1]['number'] else 'movement_window');folder.mkdir(parents=True,exist_ok=True)
    if args.publish_only:
        names=[f'windows_{size}' for size in args.windows]
        if args.refine:names.append(f'refined_{args.windows[0]}')
        publish({name:json.loads((folder/(name+'.json')).read_text(encoding='utf-8')) for name in names})
        return
    candidates={};saved={}
    def progress(window,total):
        if window['index']%20==0 or window['index']==total-1:
            print(f'Window {window["index"]+1}/{total}: {window["seconds"]:.3f}s, {window["iterations"]} iterations, converged={window["converged"]}',flush=True)
    def make(name,solver):
        started=perf_counter()
        candidate=recording_body_catalog(path,args.start,end,free_axial_lengths=True,chest_line_prior=True,
            shoulder_profile='lower_sc',relaxed_shoulders=True,proportional_spine_lengths=True,solve_sequence=solver)
        method=candidate['runs'][0]['methods']['full_body'];method['summary']['This invocation read and fit wall seconds']=perf_counter()-started
        costs=method['settings']['costs_by_family']
        total=sum(sum(v) if isinstance(v,list) else v for v in costs.values())
        if abs(total-method['frames'][0]['costs'][-1])>1e-6:raise ValueError('Global cost accounting mismatch')
        add_context(candidate)
        (folder/(name+'.json')).write_text(json.dumps(candidate,allow_nan=False),encoding='utf-8')
        candidates[name]=candidate;publish(candidates)
        print('SAVED',name,json.dumps(method['summary']),flush=True)
    for size in args.windows:
        def solve(size=size,**kwargs):
            result=fit_windows(kwargs,active_frames=size,max_iterations=args.max_iterations,function_tolerance=args.function_tolerance,progress=progress)
            if size==args.windows[0]:saved.update(arguments=kwargs,result=result)
            return result
        make(f'windows_{size}',solve)
    if args.refine:
        def refine(**kwargs):
            for key in ('times','observed','local','axial_reference_lengths','parent_attachments','parent_indices'):
                if kwargs[key]!=saved['arguments'][key]:raise ValueError('Refinement inputs changed')
            print('Starting full-sequence refinement from saved window result',flush=True)
            return refine_window_result(saved['arguments'],saved['result'],max_iterations=args.max_iterations,function_tolerance=args.function_tolerance)
        make(f'refined_{args.windows[0]}',refine)


if __name__=='__main__':main()
