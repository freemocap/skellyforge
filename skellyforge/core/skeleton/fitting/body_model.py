"""Prepare connected geometry and mapped keypoint targets without recording I/O."""
import numpy as np
from . import settings as fit_settings
from .reference_geometry import reference_positions

def body_model(skeleton, saved, scales, shoulder_profile=None, flexible_cervical=False):
    positions, shoulder_geometry = reference_positions(skeleton, scales, shoulder_profile)
    joints = {j.child.name: j for j in skeleton.joints.values()}
    roots = set(skeleton.segments) - set(joints)
    if len(roots) != 1:
        raise ValueError('Full-body review requires one connected root')
    names = [roots.pop()]
    while len(names) < len(skeleton.segments):
        children = [n for n in skeleton.segments if n not in names and joints[n].parent.name in names]
        if not children:
            raise ValueError('Disconnected or cyclic skeleton')
        names.extend(children)
    parents = [names.index(joints[n].parent.name) for n in names[1:]]
    attachments = [positions[joints[n].connect_at.name].tolist() for n in names[1:]]
    display_names = [[k for k, v in skeleton.landmarks.items() if v.segment == n] for n in names]
    display = [np.array([positions[k] for k in keys]) for n, keys in zip(names, display_names)]
    # A direct keypoint may also map to a child's origin. That is the same
    # connected point, not an independent measurement to count twice.
    candidates = {}
    for mapping in saved['mappings']:
        for name, entry in mapping['entries'].items():
            if name in skeleton.landmarks and isinstance(entry, str):
                source = (mapping['prefix'] or '') + entry
                candidates.setdefault(source, []).append(name)
    sources = {}
    for source, keys in candidates.items():
        key = max(keys, key=lambda k: (np.linalg.norm(skeleton.landmarks[k].local_position.array), k))
        owner = skeleton.landmarks[key].segment
        for other in keys:
            if other == key:
                continue
            other_owner = skeleton.landmarks[other].segment
            joint = joints.get(other_owner)
            if not (joint and joint.parent.name == owner and joint.connect_at.name == key
                    and np.allclose(skeleton.landmarks[other].local_position.array, 0)):
                raise ValueError(f'Ambiguous repeated keypoint mapping: {source}: {keys}')
        sources[key] = source
    targets = [[k for k in keys if k in sources] for keys in display_names]
    local = [[display[b][display_names[b].index(k)].tolist() for k in keys] for b, keys in enumerate(targets)]
    rest = saved['rest_pose']['orientations']
    relative = [[rest[n][c] for c in ('w', 'x', 'y', 'z')] for n in names[1:]]
    references = [0.] * len(names)
    axial_endpoints=[('sacrolumbar', 'chest_center'), ('thoracic', 'neck_center')]
    if flexible_cervical:axial_endpoints.append(('cervical_spine','craniocervical_junction'))
    for segment, endpoint in axial_endpoints:
        b = names.index(segment)
        references[b] = float(display[b][display_names[b].index(endpoint), 2])
        if references[b] <= 0:
            raise ValueError('Axial reference extent must be positive')
    return dict(names=names, parents=parents, attachments=attachments, display_names=display_names,
                display=display, targets=targets, sources=sources, local=local, relative=relative,
                references=references, shoulder_geometry=shoulder_geometry)


def frame_targets(record, model):
    observed, indices = [], []
    for keys in model['targets']:
        values, slots = [], []
        for j, key in enumerate(keys):
            source = model['sources'][key]
            if source not in record['keypoints']:
                continue
            # A saved reconstruction can be absent even with a few valid
            # keypoints. Direct mappings still define targets in that case.
            if key in record['points'] and not np.allclose(record['points'][key], record['keypoints'][source], atol=fit_settings.DIRECT_MAPPING_TOLERANCE_MM, rtol=0):
                raise ValueError(f'Frame {record["number"]}: direct mapping disagrees with keypoint {source} -> {key}')
            values.append(record['keypoints'][source].tolist())
            slots.append(j)
        observed.append(values)
        indices.append(slots)
    return observed, indices


