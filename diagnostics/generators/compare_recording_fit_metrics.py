"""Compare saved test/sample fits at identical times and shared keypoint targets."""
from skellyforge.tools.viewer.workspace import REPO_ROOT, OUTPUT_FOLDER, prepare_output
import html
import json
from pathlib import Path
import numpy as np

ROOT = REPO_ROOT
FILES = {'Test': ROOT/'build/rest_length_comparison/sc_75_rest_50.json',
         'Sample': ROOT/'build/sample_performance/tolerance_1e-06.json'}
SPINE = ('sacrolumbar', 'thoracic', 'cervical_spine')


def weights(times):
    dt=np.diff(times)
    if len(dt)==0 or not (dt>0).all():
        raise ValueError('Need increasing sample times')
    return (np.r_[dt,0]+np.r_[0,dt])/2


def weighted_rms(values, mass):
    return float(np.sqrt(np.average(np.square(values),weights=mass))) if values else None


def main():
    prepare_output()
    bank={name:json.loads(path.read_text()) for name,path in FILES.items()}
    methods={name:c['runs'][0]['methods']['full_body'] for name,c in bank.items()}
    frames={name:m['frames'] for name,m in methods.items()}
    test_times=np.array([f['time'] for f in frames['Test']])
    sample_times=np.array([f['time'] for f in frames['Sample']])
    pairs=[]
    for i,t in enumerate(test_times):
        j=int(np.argmin(abs(sample_times-t)))
        if abs(sample_times[j]-t)<1e-5:pairs.append((i,j))
    if len(pairs)<3:raise ValueError('No common timestamp grid; do not interpolate implicitly')
    times=test_times[[i for i,j in pairs]]
    mass=weights(times)
    mappings={name:m['settings']['direct_mapping_sources'] for name,m in methods.items()}
    if mappings['Test']!=mappings['Sample']:raise ValueError('Different keypoint mappings')
    def errors(name,frame):
        return {landmark:value for definition,body in zip(bank[name]['bodies'],frame['bodies'],strict=True)
                for landmark,value in zip(definition['landmark_names'],body['residuals'],strict=True)
                if value is not None}
    values={name:[] for name in bank}; error_mass=[]; shoulder={name:{side:[] for side in ('left','right')} for name in bank}
    shoulder_mass={side:[] for side in ('left','right')}
    series={name:{'target_rms':[], **{s:[] for s in SPINE}} for name in bank}
    for k,(i,j) in enumerate(pairs):
        fs={'Test':frames['Test'][i],'Sample':frames['Sample'][j]}
        es={name:errors(name,f) for name,f in fs.items()}
        shared=sorted(es['Test'].keys() & es['Sample'].keys())
        error_mass.extend([mass[k]]*len(shared))
        for name,f in fs.items():
            v=[es[name][key] for key in shared];values[name].extend(v)
            series[name]['target_rms'].append(float(np.sqrt(np.mean(np.square(v)))) if v else None)
            for s in SPINE:series[name][s].append(f['diagnostics'][s+' length (mm)'])
        for side in shoulder_mass:
            key=side+'_acromion'
            if key in shared:
                shoulder_mass[side].append(mass[k])
                for name in bank:shoulder[name][side].append(es[name][key])
    result={'common_frames':len(pairs),'time_range_seconds':[float(times[0]),float(times[-1])],
            'shared_keypoint_targets':len(error_mass),'fits':{}}
    rows=[]
    for name,m in methods.items():
        out={'all_saved_frames_target_rms_mm':m['summary']['Observed landmark RMS (mm)'],
             'matched_time_shared_target_rms_mm':weighted_rms(values[name],error_mass),
             'shoulder_target_rms_mm':{s:weighted_rms(shoulder[name][s],shoulder_mass[s]) for s in shoulder_mass},
             'spine':{},'geometry':{}}
        rows.append((name,'All shared keypoint targets','RMS mm',out['matched_time_shared_target_rms_mm']))
        for s,v in out['shoulder_target_rms_mm'].items():rows.append((name,s+' shoulder target','RMS mm',v))
        for index,s in enumerate(SPINE):
            v=np.array(series[name][s]);ref=frames[name][0]['reference_lengths'][index]
            mean=float(np.average(v,weights=mass));deviation=v-ref
            entry={'reference_mm':ref,'mean_mm':mean,'std_mm':float(np.sqrt(np.average((v-mean)**2,weights=mass))),
                   'rms_from_reference_mm':float(np.sqrt(np.average(deviation**2,weights=mass))),
                   'relative_rms_from_reference':float(np.sqrt(np.average((deviation/ref)**2,weights=mass))),
                   'rest_residual_rms':float(np.sqrt(np.average((deviation/(.5*ref))**2,weights=mass))),
                   'minimum_mm':float(v.min()),'maximum_mm':float(v.max())}
            out['spine'][s]=entry
            for metric,value in entry.items():rows.append((name,s,metric,value))
        indices=[i if name=='Test' else j for i,j in pairs]
        for key in ('SC midpoint above shoulder midpoint (mm)','SC offset from neck along torso up (mm)',
                    'left_upper_arm linkage separation (mm)','right_upper_arm linkage separation (mm)'):
            selected=[(frames[name][i]['diagnostics'].get(key),mass[k]) for k,i in enumerate(indices)]
            selected=[(v,w) for v,w in selected if v is not None]
            v,w=map(np.array,zip(*selected));mean=float(np.average(v,weights=w))
            entry={'mean_mm':mean,'std_mm':float(np.sqrt(np.average((v-mean)**2,weights=w))),
                   'rms_mm':float(np.sqrt(np.average(v*v,weights=w)))}
            out['geometry'][key]=entry
            for metric,value in entry.items():rows.append((name,key,metric,value))
        result['fits'][name]=out
    output=ROOT/'build/sample_performance/test_sample_metrics.json'
    output.write_text(json.dumps(result,indent=2))
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in (n,item,metric,f'{value:.3f}' if value is not None else 'Unavailable'))+'</tr>' for n,item,metric,value in rows)
    # Plots retain every native frame so matching the statistics does not hide
    # excursions between the test recording's decimated timestamps.
    plots={name:{'times':[f['time'] for f in fs],
                 'series':{'target_rms':[float(np.sqrt(np.mean(np.square(list(errors(name,f).values()))))) if errors(name,f) else None for f in fs],
                           **{s:[f['diagnostics'][s+' length (mm)'] for f in fs] for s in SPINE}}}
           for name,fs in frames.items()}
    page='''<!doctype html><meta charset="utf-8"><title>Test / sample fit metrics</title>
<script src="vendor/plotly-basic-2.35.2.min.js"></script>
<style>body{background:#101925;color:#e0eaf4;font:15px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:5px 14px;border-bottom:1px solid #34475b;text-align:left}.plot{height:400px;min-height:230px;resize:vertical;overflow:auto;border:1px solid #34475b;margin:15px 0}</style>
<h1>Test versus sample: accepted settings</h1>
<p>Matched timestamps, shared available keypoint targets, duration-weighted statistics. No interpolation. RMS is Euclidean point error, not per-coordinate error. SC values are model geometry, not measurement errors. Different calibration, filtering, and fitted person scales remain confounds.</p>
<p>Both fits: 3 active frames, 20:20:12, SC forward 75%, rest scale 50%, tolerance 1e-6. The same frame count spans different time intervals at 6 and 30 Hz.</p>
<p>Plots retain every native frame and each dataset's available targets; the numeric table uses matched timestamps and shared targets. Length statistics include unobserved frames. Zoom and pan with Plotly; drag each panel's lower-right corner to resize.</p>
<a href="recording_sample_performance.html">Sample 3D viewer</a> · <a href="recording_rest_length_comparison.html">Test 3D viewer</a>
<div id="plots"></div><details><summary>All numeric statistics</summary><table><tr><th>Dataset</th><th>Quantity</th><th>Metric</th><th>Value</th></tr>TABLE</table></details>
<script>const d=DATA;for(const key of ['target_rms','sacrolumbar','thoracic','cervical_spine']){const el=document.createElement('div');el.className='plot';document.getElementById('plots').appendChild(el);Plotly.newPlot(el,Object.entries(d).map(([name,s],i)=>({x:s.times,y:s.series[key],name,mode:'lines',line:{color:i?'#ffba69':'#72daca'},connectgaps:false})),{title:key==='target_rms'?'Available keypoint-target RMS per frame (mm)':key+' length (mm)',paper_bgcolor:'#101925',plot_bgcolor:'#101925',font:{color:'#e0eaf4'},xaxis:{title:'Recording time (s)'},yaxis:{title:'mm'},margin:{t:45,b:45,l:65,r:15},legend:{orientation:'h',y:1.15}},{responsive:true,scrollZoom:true});new ResizeObserver(()=>Plotly.Plots.resize(el)).observe(el);}</script>'''
    (OUTPUT_FOLDER/'recording_test_sample_metrics.html').write_text(page.replace('TABLE',table).replace('DATA',json.dumps(plots,allow_nan=False)),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
