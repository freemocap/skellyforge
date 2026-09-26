"""Read-only adapter checks against the prepared recording, when available."""
import pytest
from scripts.solver_recording_body import body_model, frame_targets
from scripts.recording_data import read_recording, recording_path
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot


@pytest.fixture(scope='module')
def inputs():
    path=recording_path()
    if not path.exists():
        pytest.skip('Prepared reference recording is not available')
    records,scale,provenance=read_recording(path,include_model=True)
    skeleton=SkeletonSnapshot.from_dict(provenance['skeleton']).restore()
    return records,skeleton,body_model(skeleton,provenance['model'],scale.segment_scales)


def test_full_model_and_single_counted_keypoint_mappings(inputs):
    _,skeleton,model=inputs
    assert set(model['names'])==set(skeleton.segments)
    assert {n for names in model['display_names'] for n in names}==set(skeleton.landmarks)
    assert len(set(model['sources'].values()))==len(model['sources'])
    for child,parent in enumerate(model['parents'],1):
        assert parent<child
    for body in model['bodies']:
        assert body['region']


def test_unavailable_keypoint_does_not_remove_model_landmark(inputs):
    records,_,model=inputs
    record=next(r for r in records if r['number']==192)
    before,slots=frame_targets(record,model)
    key=next(iter(model['sources']))
    changed={**record,'keypoints':dict(record['keypoints'])}
    del changed['keypoints'][model['sources'][key]]
    after,_=frame_targets(changed,model)
    assert sum(map(len,after))==sum(map(len,before))-1
    assert key in {n for names in model['display_names'] for n in names}


def test_direct_mapping_disagreement_is_not_silently_accepted(inputs):
    records,_,model=inputs
    record=next(r for r in records if r['number']==192)
    key=next(iter(model['sources']))
    changed={**record,'points':{**record['points'],key:record['points'][key]+1}}
    with pytest.raises(ValueError,match='direct mapping disagrees'):
        frame_targets(changed,model)


def test_missing_tail_keypoints_only_changes_initialization(monkeypatch,inputs):
    import numpy as np
    from scripts import solver_recording_body as adapter
    captured={}
    class Captured(Exception):
        pass
    def capture(**kwargs):
        captured.update(kwargs)
        raise Captured
    monkeypatch.setattr(adapter._native,'fit_chain_sequence',capture)
    with pytest.raises(Captured):
        adapter.recording_body_catalog(recording_path(),214,221,free_axial_lengths=True,chest_line_prior=True,
                                       shoulder_profile='lower_sc',relaxed_shoulders=True,equal_spine_lengths=True)
    assert len(captured['times'])==8
    assert sum(map(len,captured['observed'][1]))>0
    for i in range(2,8):
        assert sum(map(len,captured['observed'][i]))==0
        assert sum(map(len,captured['observation_indices'][i]))==0
        np.testing.assert_array_equal(captured['initial_roots'][i],captured['initial_roots'][1])
        np.testing.assert_array_equal(captured['initial_quaternions'][i][0],captured['initial_quaternions'][1][0])
        assert len(captured['initial_quaternions'][i])==61


def test_interval_without_any_saved_root_seed_is_explicit_error(inputs):
    from scripts.solver_recording_body import recording_body_catalog
    with pytest.raises(ValueError,match='No saved root pose available'):
        recording_body_catalog(recording_path(),216,221)
