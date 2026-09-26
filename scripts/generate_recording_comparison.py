"""Build a lightweight overlay viewer from saved solver-lab results; never fit."""
import json
import re
from pathlib import Path

FOLDER = Path(__file__).parent
PALETTE = ['#9cafff', '#da95e8', '#bdb88a', '#69d7d0', '#d5dce5', '#77b5ff', '#e8ae71', '#82e39c', '#ff7899']


def comparison_data(bank):
    choices = []
    has_equality = any(e['id']=='recording_spine_equality' for e in bank)
    for experiment_id, group, allowed in (
        ('recording_spine_equality', 'Spine length coupling', {'equal_spine_lengths','lower_sc_relaxed'}),
        ('recording_body', 'Spine comparisons', None),
        ('recording_shoulders', 'Shoulder offsets', {'lower_sc', 'lower_closer_sc'}),
        ('recording_shoulder_linkages', 'Relaxed shoulder connections', {'lower_closer_sc_relaxed'} if has_equality else {'lower_sc_relaxed', 'lower_closer_sc_relaxed'}),
    ):
        experiment = next((e for e in bank if e['id'] == experiment_id), None)
        if experiment is None:
            continue
        if len(experiment['runs']) != 1:
            raise ValueError('Recording comparison requires one explicitly selected run per experiment')
        for choice in experiment['methods']:
            if allowed is None or choice['id'] in allowed:
                choices.append((experiment, group, choice, experiment['runs'][0]['methods'][choice['id']]))
    if not choices:
        raise ValueError('No saved real full-body fits; generate the recording solver experiments first')
    priority = {'Spine length coupling': -1, 'Relaxed shoulder connections': 0, 'Shoulder offsets': 1, 'Spine comparisons': 2}
    choices.sort(key=lambda choice: priority[choice[1]])
    reference = choices[0][0]
    ref_frames = choices[0][3]['frames']
    frame_ids = [f['diagnostics']['Recording frame'] for f in ref_frames]
    times = [f['diagnostics']['Recording timestamp (s)'] for f in ref_frames]
    contexts = [f['recording_context'] for f in ref_frames]
    solutions = []
    for index, (experiment, group, choice, method) in enumerate(choices):
        frames = method['frames']
        if (experiment['metadata']['recording']['sha256'] != reference['metadata']['recording']['sha256']
                or [f['diagnostics']['Recording frame'] for f in frames] != frame_ids
                or [f['diagnostics']['Recording timestamp (s)'] for f in frames] != times
                or [f['recording_context'] for f in frames] != contexts):
            raise ValueError('Cannot overlay fits with different recordings, frame times or recording context')
        definitions = method.get('body_definitions', experiment['bodies'])
        if [(b['id'], b['landmark_names']) for b in definitions] != [(b['id'], b['landmark_names']) for b in reference['bodies']]:
            raise ValueError('Cannot overlay different skeleton identities')
        solutions.append(dict(
            id=choice['id'], label=choice['label'], group=group,
            color={'equal_spine_lengths':'#ffd273','lower_sc':'#77b5ff', 'lower_sc_relaxed':'#82e39c', 'lower_closer_sc':'#e8ae71', 'lower_closer_sc_relaxed':'#ff7899'}.get(choice['id'], PALETTE[index % len(PALETTE)]),
            definitions=definitions, settings=method['settings'], objective=method['objective'],
            provenance=method.get('metadata', experiment['metadata']), summary=method['summary'],
            converged=frames[0]['converged'], seconds=frames[0]['seconds'], report=frames[0]['report'],
            frames=[dict(bodies=[dict(quaternion=b['quaternion'], translation=b['translation'],
                                      fitted=b['fitted'], axial_scale=b.get('axial_scale', 1)) for b in f['bodies']],
                         linkages=f.get('linkages', []), diagnostics=f['diagnostics']) for f in frames]))
    views = next((e.get('annotated_views') for e, _, _, _ in reversed(choices) if e.get('annotated_views')), [])
    mappings = choices[0][3]['settings']['direct_mapping_sources']
    hand_landmarks = {name for body in reference['bodies'] if body['region'] in ('Left hand', 'Right hand') for name in body['landmark_names']}
    return dict(recording=reference['metadata']['recording'], frame_ids=frame_ids, times=times,
                contexts=contexts, solutions=solutions, videos=views,
                hand_landmarks=sorted(hand_landmarks), hand_keypoints=sorted({mappings[n] for n in hand_landmarks if n in mappings}))


def main():
    source = FOLDER / 'solver_viewer.html'
    match = re.search(r'const EXPERIMENTS\s*=\s*(\[.*?\]);', source.read_text(encoding='utf-8'), re.S)
    if not match:
        raise ValueError('Saved solver viewer has no experiment data')
    data = comparison_data(json.loads(match.group(1)))
    page = (FOLDER / 'recording_comparison.html.template').read_text(encoding='utf-8')
    page = page.replace('__DATA__', json.dumps(data, allow_nan=False).replace('<', '\\u003c'))
    target = FOLDER / 'recording_comparison.html'
    target.write_text(page, encoding='utf-8')
    print(f'{target.resolve()} — {len(data["solutions"])} saved fits / {len(data["times"])} frames / {target.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
