"""Rigid authored torso, four observed landmarks, explicit relative-pose priors."""
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge import _native
from scripts import solver_fit_settings as fit_settings
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


def torso_experiment(noise=1,motion=0):
    m=model();times,observed,records=inputs(m,noise,motion)
    initial,roots=initialization(m,observed)
    methods={}
    for name,enabled in [('no_prior',False),('rest_prior',True)]:
        result=_native.fit_chain_sequence(
        length_prior_fraction=fit_settings.LENGTH_PRIOR_FRACTION,
        length_acceleration_scale=fit_settings.LENGTH_ACCELERATION_SCALE_MM_S2,
        local=m['local'],observed=observed,parent_attachments=m['parent_attachments'],child_attachments=m['child_attachments'],parent_indices=m['parents'],times=times.tolist(),
            position_scale=fit_settings.POSITION_RESIDUAL_SCALE_MM,linear_acceleration_scale=fit_settings.ROOT_ACCELERATION_SCALE_MM_S2,angular_acceleration_scale=fit_settings.ANGULAR_ACCELERATION_SCALE_RAD_S2,
            initial_quaternions=initial,initial_roots=roots,rest_relative_quaternions=m['relative'] if enabled else [],rest_pose_scale=fit_settings.REST_POSE_RESIDUAL_SCALE_RAD)
        frames=[]
        for i,record in enumerate(records):
            bodies=[];diagnostics={}
            for b,n in enumerate(NAMES):
                q,t=result.quaternions[i][b],result.translations[i][b]
                iq,it=result.initial_quaternions[i][b],result.initial_translations[i][b]
                r=Rotation.from_quat(q,scalar_first=True)
                fitted=r.apply(m['display'][b])+t
                initial_points=Rotation.from_quat(iq,scalar_first=True).apply(m['display'][b])+it
                obs=[observed[i][b][TARGETS[b].index(k)] if k in TARGETS[b] else None for k in m['display_names'][b]]
                errors=[float(np.linalg.norm(fitted[j]-p)) if p is not None else None for j,p in enumerate(obs)]
                angle=float(np.degrees((r.inv()*record['rotations'][b]).magnitude()))
                diagnostics[n+' angular error (degrees)']=angle
                if b:
                    parent=m['parents'][b-1]
                    expected=np.array(result.translations[i][parent])+Rotation.from_quat(result.quaternions[i][parent],scalar_first=True).apply(m['parent_attachments'][b-1])
                    diagnostics[n+' attachment error (mm)']=float(np.linalg.norm(expected-t))
                bodies.append(dict(local=m['display'][b].tolist(),truth=record['truth'][b].tolist(),observed=obs,fitted=fitted.tolist(),initial=initial_points.tolist(),quaternion=q,translation=t,
                    initial_quaternion=iq,initial_translation=it,reference_quaternion=record['rotations'][b].as_quat(scalar_first=True).tolist(),reference_translation=record['translations'][b].tolist(),
                    residuals=errors,truth_rms=float(np.sqrt(np.mean(np.sum((fitted-record['truth'][b])**2,axis=1))))))
            frames.append(dict(time=float(times[i]),bodies=bodies,root=result.roots[i],costs=result.costs,converged=result.converged,seconds=result.seconds,report=result.report,diagnostics=diagnostics,
                observability='Only two hip and two acromion landmarks are observed. Spine poses and clavicle roll are not uniquely measured. Rest-pose residuals are preferences, not observations.'))
        summary={'Ceres parameter blocks':result.parameter_blocks,'Ceres residual blocks':result.residual_blocks,
            'Observed landmark RMS (mm)':float(np.sqrt(np.mean([e*e for f in frames for b in f['bodies'] for e in b['residuals'] if e is not None])))}
        for b,n in enumerate(NAMES):
            summary[n+' angular RMS vs known (degrees)']=float(np.sqrt(np.mean([f['diagnostics'][n+' angular error (degrees)']**2 for f in frames])))
        methods[name]=dict(frames=frames,summary=summary,problem=dict(chain=True,parents=m['parents'],connected=True,temporal=True,acceleration=True,rest_prior=enabled),
            settings=dict(position_scale_mm=fit_settings.POSITION_RESIDUAL_SCALE_MM,linear_motion_scale=fit_settings.ROOT_ACCELERATION_SCALE_MM_S2,angular_motion_scale=fit_settings.ANGULAR_ACCELERATION_SCALE_RAD_S2,scale_units=['mm/s^2','rad/s^2'],rest_pose_scale_radians=fit_settings.REST_POSE_RESIDUAL_SCALE_RAD,rest_relative_quaternions=m['relative'],model_scale_mm=1700.,
                initialization='Hip midpoint and hip/shoulder basis; authored relative rest quaternions. Same initialization in both methods.',
                costs_by_family=dict(landmarks=result.landmark_cost,root_acceleration=result.root_acceleration_cost,segment_angular_acceleration=result.angular_acceleration_costs,relative_pose=result.relative_pose_cost)),
            objective='Four measured landmark residual blocks per frame, exact rigid attachments, temporal acceleration residuals'+('; plus four relative-quaternion rest-pose residuals per frame.' if enabled else '; no rest-pose residuals.'))
    return dict(parameters=dict(noise=noise,motion=motion),times=times.tolist(),methods=methods)


def torso_catalog():
    return dict(id='torso',label='11 - Rigid torso / four landmarks',description='Existing Forge human definitions at a declared 1700 mm scale. Rigid segments and exact attachments. Compare identical four-landmark inputs with rest-pose residuals off/on; reference motion is evaluation only.',
        controls=[dict(id='noise',label='Noise / mm per axis',values=[1,0,5]),dict(id='motion',label='Motion: 0 shoulder elevation, 1 left shrug, 2 torso twist',values=[0,1,2])],bodies=model()['bodies'],
        methods=[dict(id='no_prior',label='No rest-pose residuals'),dict(id='rest_prior',label='With rest-pose residuals')],runs=[torso_experiment(n,m) for n in [1,0,5] for m in [0,1,2]])
