"""Rigid authored torso, four observed landmarks, explicit relative-pose priors."""
from test_support.torso import model, axis_quaternion, forward, inputs, initialization, NAMES, TARGETS
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge import _native
from skellyforge.core.skeleton.fitting import settings as fit_settings













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
