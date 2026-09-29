"""Synthetic torso inputs; no experiment execution or output generation."""
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition
from skellyforge.core.skeleton.pose.rest_pose import RestPose

NAMES=['pelvis','sacrolumbar','thoracic','left_clavicle','right_clavicle']
TARGETS=[['left_hip_socket','right_hip_socket'],[],[],['left_acromion'],['right_acromion']]

def model(skeleton=None, rest=None, scales=None):
    skeleton=SkeletonDefinition.from_default_yaml() if skeleton is None else skeleton
    rest=RestPose.from_default_yaml(skeleton=skeleton) if rest is None else rest
    scales=dict.fromkeys(NAMES,1700.) if scales is None else scales
    parents=[NAMES.index(rest.parents[n]) for n in NAMES[1:]]
    attachments=[(skeleton.landmarks[rest.connect_ats[n]].local_position.array*scales[rest.parents[n]]).tolist() for n in NAMES[1:]]
    display_names=[[k for k,v in skeleton.landmarks.items() if v.segment==n] for n in NAMES]
    display=[np.array([skeleton.landmarks[k].local_position.array*scales[skeleton.landmarks[k].segment] for k in names]) for names in display_names]
    local=[[(skeleton.landmarks[k].local_position.array*scales[skeleton.landmarks[k].segment]).tolist() for k in names] for names in TARGETS]
    relative=[rest.relative_orientations[n].as_array().tolist() for n in NAMES[1:]]
    bodies=[]
    for i,n in enumerate(NAMES):
        links=[]
        if i==0:
            links.extend(dict(position=p,label=k) for p,k in zip(local[0],TARGETS[0]))
        if i: links.append(dict(position=[0,0,0],label='parent attachment'))
        for b,parent in enumerate(parents,1):
            if parent==i: links.append(dict(position=attachments[b-1],label='to '+NAMES[b]))
        if i in [3,4]: links.append(dict(position=local[i][0],label='shoulder landmark / distal endpoint'))
        bodies.append(dict(id=n,label=n,landmark_names=display_names[i],edges=[],attachments=links))
    return dict(parents=parents,parent_attachments=attachments,child_attachments=[[0.,0.,0.]]*4,
        local=local,relative=relative,display=display,display_names=display_names,bodies=bodies)

def axis_quaternion(axis,angle):
    return Rotation.from_quat([np.cos(angle/2),*(np.array(axis)*np.sin(angle/2))],scalar_first=True)

def forward(m,rotations,root):
    translations=[np.array(root)]
    for b,parent in enumerate(m['parents'],1):
        translations.append(translations[parent]+rotations[parent].apply(m['parent_attachments'][b-1]))
    return translations

def inputs(m,noise,motion):
    times=np.linspace(0,2,21)
    rng=np.random.default_rng(23)
    records=[]
    observed=[]
    for time in times:
        pulse=np.sin(np.pi*time/2)**2
        rotations=[axis_quaternion([0,0,1],.12*np.sin(np.pi*time))]
        for b,parent in enumerate(m['parents'],1):
            relative=Rotation.from_quat(m['relative'][b-1],scalar_first=True)
            if b in [3,4] and motion in [0,1]:
                angle=.65*pulse*(1 if b==3 else -1) if motion==0 or b==3 else 0.
                relative=axis_quaternion([0,1,0],angle)*relative
            if b==2 and motion==2: relative=axis_quaternion([0,0,1],.5*np.sin(np.pi*time))*relative
            rotations.append(rotations[parent]*relative)
        translations=forward(m,rotations,[15*np.sin(np.pi*time),0,100])
        truth=[r.apply(points)+t for r,points,t in zip(rotations,m['display'],translations)]
        obs=[]
        for b,names in enumerate(TARGETS):
            obs.append([(truth[b][m['display_names'][b].index(k)]+rng.normal(0,noise,3)).tolist() for k in names])
        observed.append(obs)
        records.append(dict(rotations=rotations,translations=translations,truth=truth))
    return times,observed,records

def initialization(m,observed):
    """Only four observations and authored rest rotations; never reference poses."""
    quaternions=[]; roots=[]
    for frame in observed:
        left,right=np.array(frame[0]);root=(left+right)/2
        x=right-left;x=x/np.linalg.norm(x)
        up=(np.array(frame[3][0])+frame[4][0])/2-root
        z=up-x*np.dot(up,x)
        if np.linalg.norm(z)<1e-8: raise ValueError('Hip/shoulder initialization is degenerate')
        z/=np.linalg.norm(z);y=np.cross(z,x)
        rotations=[Rotation.from_matrix(np.column_stack([x,y,z]))]
        for b,parent in enumerate(m['parents'],1):
            rotations.append(rotations[parent]*Rotation.from_quat(m['relative'][b-1],scalar_first=True))
        quaternions.append([r.as_quat(scalar_first=True).tolist() for r in rotations]);roots.append(root.tolist())
    return quaternions,roots
