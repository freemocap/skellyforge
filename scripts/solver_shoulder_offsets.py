"""Explicit fixed reference-geometry experiments; never estimate person scale here."""
import numpy as np
from scripts.solver_fit_settings import BOUND_CONTACT_TOLERANCE_MM

SC_LOWERING_REFERENCE_FRACTION = 0.10
SC_REDUCED_FORWARD_FACTOR = 0.75
SHOULDER_PROFILES = {
    'lower_sc': dict(label='Lower SC attachments', lowering_fraction=SC_LOWERING_REFERENCE_FRACTION, forward_factor=1.),
    'lower_closer_sc': dict(label='Lower SC + reduced forward offset', lowering_fraction=SC_LOWERING_REFERENCE_FRACTION, forward_factor=SC_REDUCED_FORWARD_FACTOR),
}
SC_LANDMARKS = ('left_sternoclavicular', 'right_sternoclavicular', 'sternoclavicular_notch')


def reference_positions(skeleton, scales, profile=None):
    positions = {name: np.array(landmark.local_position.array * scales[landmark.segment], copy=True)
                 for name, landmark in skeleton.landmarks.items()}
    if profile is None:
        return positions, None
    spec = SHOULDER_PROFILES[profile]
    neck = positions['neck_center']
    extent = float(neck[2])
    if extent <= 0:
        raise ValueError('Shoulder offset experiment requires positive thoracic reference extent')
    before = {name: positions[name].tolist() for name in SC_LANDMARKS}
    for name in SC_LANDMARKS:
        if skeleton.landmarks[name].segment != skeleton.landmarks['neck_center'].segment:
            raise ValueError('SC and neck-center landmarks must share their authored reference frame')
        positions[name][1] = neck[1] + (positions[name][1]-neck[1])*spec['forward_factor']
        positions[name][2] -= extent*spec['lowering_fraction']
    return positions, dict(profile=profile, **spec, thoracic_reference_length_mm=extent,
        saved_segment_scales=dict(scales), original_local_positions_mm=before,
        experimental_local_positions_mm={name: positions[name].tolist() for name in SC_LANDMARKS},
        policy='Fixed reference-geometry change before pose fitting. Saved person scale, widths, clavicle lengths and rest quaternions unchanged. Local Z follows the existing thoracic axial deformation.')


def shoulder_diagnostics(frames, bodies):
    indices = {body['id']: i for i, body in enumerate(bodies)}
    for frame in frames:
        if any(length <= BOUND_CONTACT_TOLERANCE_MM for length in frame['lengths']):
            frame['diagnostics']['Spine fit warning'] = 'A free spine length reached zero; this is not a usable anatomical pose.'
        line = frame.get('chest_line')
        if line is not None:
            up = np.asarray(line['up'])
            shoulder_center = np.asarray(line['shoulder'])
            thoracic = indices['thoracic']
            slot = bodies[thoracic]['landmark_names'].index('neck_center')
            neck = np.asarray(frame['bodies'][thoracic]['fitted'][slot])
            sc = np.mean([frame['bodies'][indices[side+'_clavicle']]['translation']
                          for side in ('left', 'right')], axis=0)
            frame['diagnostics']['Neck center above shoulder midpoint (mm)'] = float((neck-shoulder_center) @ up)
            frame['diagnostics']['SC midpoint above shoulder midpoint (mm)'] = float((sc-shoulder_center) @ up)
            frame['diagnostics']['SC offset from neck along torso up (mm)'] = float((sc-neck) @ up)
    for side in ('left', 'right'):
        index = next(i for i,b in enumerate(bodies) if b['id']==side+'_clavicle')
        slot = bodies[index]['landmark_names'].index(side+'_acromion')
        for frame in frames:
            body=frame['bodies'][index]
            fitted=np.asarray(body['fitted'][slot]);origin=np.asarray(body['translation'])
            frame['diagnostics'][side+' clavicle length (mm)']=float(np.linalg.norm(fitted-origin))
            observed=body['observed'][slot]
            if observed is not None:
                frame['diagnostics'][side+' shoulder target error (mm)']=float(np.linalg.norm(fitted-observed))
                frame['diagnostics'][side+' SC to shoulder keypoint distance (mm)']=float(np.linalg.norm(origin-observed))
