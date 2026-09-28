"""Explicit fixed reference-geometry experiments; never estimate person scale here."""
import numpy as np
from .settings import BOUND_CONTACT_TOLERANCE_MM

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


