"""Saved-fit overlay export must preserve geometry and reject mismatched inputs."""
import copy
import pytest
from scripts.generate_recording_comparison import comparison_data


def bank():
    definitions=[dict(id='bone',region='Trunk',landmark_names=['tip'],attachments=[dict(label='tip',position=[0,0,10])])]
    frame=dict(diagnostics={'Recording frame':180,'Recording timestamp (s)':30.},
               recording_context=dict(keypoints={'point':[1,2,3]},landmarks={'tip':[1,2,3]},segments={}),
               converged=True,seconds=1.,report='converged',
               bodies=[dict(quaternion=[1,0,0,0],translation=[1,2,3],fitted=[[1,2,13]],axial_scale=1)])
    method=dict(frames=[frame],settings=dict(direct_mapping_sources={'tip':'point'}),objective='test',summary={})
    return [dict(id='recording_body',metadata=dict(recording={'path':'record.parquet','sha256':'source'}),
                 bodies=definitions,methods=[dict(id='a',label='A'),dict(id='b',label='B')],
                 runs=[dict(methods={'a':copy.deepcopy(method),'b':copy.deepcopy(method)})])]


def test_export_preserves_saved_geometry_and_input():
    source=bank();before=copy.deepcopy(source);result=comparison_data(source)
    assert source==before
    assert result['frame_ids']==[180]
    assert result['times']==[30.]
    assert len(result['solutions'])==2
    for solution in result['solutions']:
        assert solution['frames'][0]['bodies'][0]['fitted']==[[1,2,13]]
        assert solution['definitions']==source[0]['bodies']


@pytest.mark.parametrize('field,value',[('Recording frame',181),('Recording timestamp (s)',30.01)])
def test_rejects_unsynchronized_frames(field,value):
    source=bank();source[0]['runs'][0]['methods']['b']['frames'][0]['diagnostics'][field]=value
    with pytest.raises(ValueError,match='different recordings, frame times'):
        comparison_data(source)


def test_rejects_changed_keypoint_reference_layer():
    source=bank();source[0]['runs'][0]['methods']['b']['frames'][0]['recording_context']['keypoints']['point'][0]=10
    with pytest.raises(ValueError,match='recording context'):
        comparison_data(source)


def test_full_recording_is_not_mixed_with_short_window_fits():
    source=bank();full=copy.deepcopy(source[0]);full['id']='recording_full_body'
    for method in full['runs'][0]['methods'].values():
        method['frames'][0]['diagnostics']['Recording frame']=0
        method['frames'][0]['diagnostics']['Recording timestamp (s)']=0.
    result=comparison_data(source+[full])
    assert result['frame_ids']==[0]
    assert len(result['solutions'])==2
    assert all(s['group']=='Full recording' for s in result['solutions'])
