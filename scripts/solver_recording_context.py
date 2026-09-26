"""Saved body context and exact annotated-video frame previews; never refits data."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
from scripts.recording_data import read_recording,digest
from scripts.generate_real_skeleton_viewer import saved_segments
from scripts.generate_solver_viewer import render_experiments


def check_video_times(video_times,records):
    if len(video_times)!=len(records):raise ValueError('Annotated video and recording frame counts differ')
    for index,(timestamp,record) in enumerate(zip(video_times,records)):
        if record['number']!=index or abs(timestamp-(record['time']-records[0]['time']))>1e-4:
            raise ValueError(f'Annotated video timestamp/frame mismatch at {index}')


def annotated_previews(path,records):
    paths=sorted((path.parent/'annotated_videos').glob('*.mp4'))
    if not paths:raise FileNotFoundError('No annotated videos beside '+str(path))
    if not shutil.which('ffprobe') or not shutil.which('ffmpeg'):raise RuntimeError('Annotated previews require ffprobe and ffmpeg on PATH')
    folder=Path(__file__).parent/'.solver_media'/'recording_torso'
    folder.mkdir(parents=True,exist_ok=True)
    views=[]
    for index,video in enumerate(paths):
        data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)],text=True))
        times=[float(f['best_effort_timestamp_time']) for f in data['frames']]
        check_video_times(times,records)
        camera=folder/f'camera{index}';camera.mkdir(exist_ok=True)
        signature=dict(source=str(video),sha256=digest(video),count=len(times),width=960,quality=3)
        manifest=camera/'manifest.json'
        cached=manifest.exists() and json.loads(manifest.read_text())==signature and all((camera/f'{i:06d}.jpg').exists() for i in range(len(times)))
        if not cached:
            subprocess.run(['ffmpeg','-v','error','-y','-i',str(video),'-map','0:v:0','-vf','scale=960:-2','-fps_mode','passthrough','-q:v','3','-start_number','0',str(camera/'%06d.jpg')],check=True)
            if digest(video)!=signature['sha256']:raise RuntimeError('Annotated video changed during decoding')
            if not all((camera/f'{i:06d}.jpg').exists() for i in range(len(times))):raise RuntimeError('Annotated preview decode incomplete')
            manifest.write_text(json.dumps(signature,indent=2))
        views.append(dict(label=video.name,source=str(video),sha256=signature['sha256'],base=f'.solver_media/recording_torso/camera{index}/',count=len(times)))
    return views


def add_context(experiment,*,videos=True):
    provenance=experiment['metadata']['recording'];path=Path(provenance['path'])
    records,fit,meta=read_recording(path)
    if meta['sha256']!=provenance['sha256']:raise ValueError('Recording differs from the fitted run; regenerate fit first')
    by_number={r['number']:r for r in records}
    for run in experiment['runs']:
        for method in run['methods'].values():
            for frame in method['frames']:
                record=by_number[frame['diagnostics']['Recording frame']]
                frame['recording_context']=dict(segments=saved_segments(record,meta['skeleton'],fit),landmarks={n:p.tolist() for n,p in record['points'].items()},keypoints={n:p.tolist() for n,p in record['keypoints'].items()})
    if videos:experiment['annotated_views']=annotated_previews(path,records)
    if digest(path)!=provenance['sha256']:raise RuntimeError('Recording changed while preparing context')
    return experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-videos',action='store_true');args=parser.parse_args()
    page=Path(__file__).with_name('solver_viewer.html')
    bank=json.loads(re.search(r'const EXPERIMENTS\s*=\s*(\[.*?\]);',page.read_text(encoding='utf-8'),re.S).group(1))
    experiment=next(e for e in bank if e['id']=='recording_torso')
    add_context(experiment,videos=not args.no_videos)
    render_experiments(bank=bank)
    print('Saved body context added; annotated cameras:',len(experiment.get('annotated_views',[])))

if __name__=='__main__':main()
