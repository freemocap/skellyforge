"""Comparisons must retain provenance and reject mismatched inputs."""
import copy
import json
import pytest
from scripts import solver_recording_body as review


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    body=dict(observed=[[1,2,3]],initial_quaternion=[1,0,0,0],initial_translation=[0,0,0])
    experiment=dict(id='recording_body',bodies=[dict(id='pelvis')],
        metadata=dict(native_sha256='old-build',recording=dict(sha256='same-data')),
        methods=[dict(id='full_body',label='Baseline')],
        runs=[dict(times=[0],methods=dict(full_body=dict(frames=[dict(bodies=[body])])) )])
    page=tmp_path/'viewer.html';page.write_text('const EXPERIMENTS='+json.dumps([experiment])+';')
    captured=[]
    monkeypatch.setattr(review,'render_experiments',lambda *,bank:captured.extend(bank))
    candidate=copy.deepcopy(experiment);candidate['metadata']['native_sha256']='new-build'
    return page,candidate,captured


def test_comparison_retains_baseline_and_each_native_provenance(comparison):
    page,candidate,captured=comparison
    review.publish_method(candidate,'wider_symmetric','Wider',page=page)
    methods=captured[0]['runs'][0]['methods']
    assert set(methods)=={'full_body','wider_symmetric'}
    assert methods['full_body']['metadata']['native_sha256']=='old-build'
    assert methods['wider_symmetric']['metadata']['native_sha256']=='new-build'


@pytest.mark.parametrize('field',['observed','initial_quaternion','initial_translation'])
def test_changed_targets_or_initialization_cannot_be_presented_as_same_input(comparison,field):
    page,candidate,captured=comparison
    candidate['runs'][0]['methods']['full_body']['frames'][0]['bodies'][0][field]=[]
    with pytest.raises(ValueError,match='initialization changed'):
        review.publish_method(candidate,'wider_symmetric','Wider',page=page)
    assert not captured


def test_different_recording_cannot_be_mixed(comparison):
    page,candidate,captured=comparison
    candidate['metadata']['recording']['sha256']='different-data'
    with pytest.raises(ValueError,match='refusing to mix runs'):
        review.publish_method(candidate,'wider_symmetric','Wider',page=page)
    assert not captured


def test_other_residual_settings_cannot_change_in_length_comparison(comparison):
    page,candidate,captured=comparison
    candidate['runs'][0]['methods']['full_body']['settings']={'angular_motion_scale':42.}
    with pytest.raises(ValueError,match='another solver setting'):
        review.publish_method(candidate,'wider_symmetric','Wider',page=page)
    assert not captured
