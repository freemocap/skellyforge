"""Fixed shoulder geometry must stay separate from person scale and pose targets."""
import numpy as np
import pytest
from scripts.recording_data import read_recording, recording_path
from scripts.solver_recording_body import body_model, frame_targets
from scripts.solver_shoulder_offsets import SC_LANDMARKS, SHOULDER_PROFILES
from skellyforge.core.skeleton.skeleton_snapshot import SkeletonSnapshot


@pytest.fixture(scope='module')
def inputs():
    path=recording_path()
    if not path.exists():pytest.skip('Prepared reference recording unavailable')
    records,scale,p=read_recording(path,include_model=True)
    return records,scale,SkeletonSnapshot.from_dict(p['skeleton']).restore(),p['model']


@pytest.mark.parametrize('profile',list(SHOULDER_PROFILES))
def test_profiles_change_only_declared_landmarks_and_matching_attachments(inputs,profile):
    records,scale,skeleton,saved=inputs
    scales=dict(scale.segment_scales)
    original={k:v.local_position.array.copy() for k,v in skeleton.landmarks.items()}
    baseline=body_model(skeleton,saved,scales)
    candidate=body_model(skeleton,saved,scales,profile)
    assert scales==scale.segment_scales
    assert candidate['parents']==baseline['parents']
    assert candidate['relative']==baseline['relative']
    assert candidate['references']==baseline['references']
    assert candidate['local']==baseline['local'] # All direct-keypoint local targets unchanged.
    assert candidate['sources']==baseline['sources']
    for name,landmark in skeleton.landmarks.items():
        np.testing.assert_array_equal(landmark.local_position.array,original[name])
    changed=[]
    for b,keys in enumerate(baseline['display_names']):
        for j,key in enumerate(keys):
            if not np.array_equal(baseline['display'][b][j],candidate['display'][b][j]):changed.append(key)
    assert set(changed)==set(SC_LANDMARKS)
    for child,parent in enumerate(candidate['parents'],1):
        name=candidate['names'][child]
        joint=next(j for j in skeleton.joints.values() if j.child.name==name)
        slot=candidate['display_names'][parent].index(joint.connect_at.name)
        np.testing.assert_array_equal(candidate['attachments'][child-1],candidate['display'][parent][slot])
        if name not in ('left_clavicle','right_clavicle'):
            assert candidate['attachments'][child-1]==baseline['attachments'][child-1]
    for record in records[180:214]:
        assert frame_targets(record,baseline)==frame_targets(record,candidate)
    g=candidate['shoulder_geometry'];spec=SHOULDER_PROFILES[profile]
    for name in SC_LANDMARKS:
        old=np.array(g['original_local_positions_mm'][name]);new=np.array(g['experimental_local_positions_mm'][name])
        assert old[0]==new[0]
        assert new[1]==pytest.approx(old[1]*spec['forward_factor'])
        assert old[2]-new[2]==pytest.approx(g['thoracic_reference_length_mm']*spec['lowering_fraction'])
