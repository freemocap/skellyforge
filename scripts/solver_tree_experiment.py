"""Dense-observation infrastructure check, not the four-landmark torso solve."""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from skellyforge import _native
from scripts import solver_fit_settings as fit_settings
from scripts.solver_axial_geometry import axial_points


def tree_inputs(noise=1., deforming=False):
    local=np.array(list(itertools.product([-15.,15.],[-15.,15.],[-40.,40.])))
    parents=[0,1,2,2]
    attachments=np.array([[0.,0.,40.],[0.,0.,40.],[0.,-35.,40.],[0.,35.,40.]])
    child=np.tile([0.,0.,-40.],(4,1))
    times=np.linspace(0,2,21)
    rng=np.random.default_rng(17)
    records=[]
    for time in times:
        phase=np.pi*time
        angles=[.1*np.sin(phase),.15*np.sin(phase),.2*np.sin(phase),.7+.25*np.sin(phase),-.7-.25*np.cos(phase)]
        rotations=[Rotation.from_quat([np.cos(a/2),np.sin(a/2),0.,0.],scalar_first=True) for a in angles]
        references=[0.,80.,80.,0.,0.]
        lengths=[0.,80.-24*np.sin(phase/2)**2,80.-16*np.sin(phase/2)**2,0.,0.] if deforming else references
        translations=[np.array([20*np.sin(phase),0.,70.])]
        for b,p in enumerate(parents,1):
            translations.append(translations[p]+rotations[p].apply(axial_points(attachments[b-1],references[p],lengths[p]))-rotations[b].apply(axial_points(child[b-1],references[b],lengths[b])))
        truth=np.array([r.apply(axial_points(local,references[b],lengths[b]))+t for b,(r,t) in enumerate(zip(rotations,translations))])
        records.append(dict(lengths=lengths,rotations=rotations,translations=translations,truth=truth,observed=truth+rng.normal(0,noise,truth.shape)))
    return local,parents,attachments,child,times,records


def tree_experiment(noise=1., *, deforming=False, axial=False):
    local,parents,parent,child,times,records=tree_inputs(noise,deforming=deforming)
    references=[0.,80.,80.,0.,0.]
    result=_native.fit_chain_sequence(
        length_prior_fraction=fit_settings.LENGTH_PRIOR_FRACTION,
        length_acceleration_scale=fit_settings.LENGTH_ACCELERATION_SCALE_MM_S2,
        local=[local.tolist()]*5,observed=[r['observed'].tolist() for r in records],
        parent_attachments=parent.tolist(),child_attachments=child.tolist(),parent_indices=parents,times=times.tolist(),
        position_scale=fit_settings.POSITION_RESIDUAL_SCALE_MM,linear_acceleration_scale=fit_settings.ROOT_ACCELERATION_SCALE_MM_S2,angular_acceleration_scale=fit_settings.ANGULAR_ACCELERATION_SCALE_RAD_S2,axial_reference_lengths=references if axial else [])
    frames=[]
    for i,record in enumerate(records):
        bodies=[]
        diagnostics={}
        for b in range(5):
            q,t=result.quaternions[i][b],result.translations[i][b]
            iq,it=result.initial_quaternions[i][b],result.initial_translations[i][b]
            r=Rotation.from_quat(q,scalar_first=True)
            fitted=r.apply(axial_points(local,references[b],result.lengths[i][b] if axial else references[b]))+t
            initial=Rotation.from_quat(iq,scalar_first=True).apply(local)+it
            error=float(np.sqrt(np.mean(np.sum((fitted-record['truth'][b])**2,axis=1))))
            angle=float(np.degrees((r.inv()*record['rotations'][b]).magnitude()))
            diagnostics[f'Segment {b} angular error (degrees)']=angle
            diagnostics[f'Segment {b} landmark error vs known (mm)']=error
            bodies.append(dict(axial_scale=result.lengths[i][b]/references[b] if axial and references[b] else 1.,reference_axial_scale=record["lengths"][b]/references[b] if references[b] else 1.,local=local.tolist(),truth=record['truth'][b].tolist(),observed=record['observed'][b].tolist(),fitted=fitted.tolist(),initial=initial.tolist(),
                quaternion=q,translation=t,initial_quaternion=iq,initial_translation=it,
                reference_quaternion=record['rotations'][b].as_quat(scalar_first=True).tolist(),reference_translation=record['translations'][b].tolist(),
                truth_rms=error,residuals=np.linalg.norm(fitted-record['observed'][b],axis=1).tolist()))
        for b,p in enumerate(parents,1):
            a=Rotation.from_quat(result.quaternions[i][p],scalar_first=True).apply(axial_points(parent[b-1],references[p],result.lengths[i][p] if axial else references[p]))+result.translations[i][p]
            c=Rotation.from_quat(result.quaternions[i][b],scalar_first=True).apply(axial_points(child[b-1],references[b],result.lengths[i][b] if axial else references[b]))+result.translations[i][b]
            diagnostics[f'Linkage {p} to {b} equation error (mm)']=float(np.linalg.norm(a-c))
        frames.append(dict(lengths=[result.lengths[i][b] if axial else references[b] for b in [1,2]],reference_lengths=[record["lengths"][b] for b in [1,2]],time=float(times[i]),bodies=bodies,root=result.roots[i],costs=result.costs,converged=result.converged,seconds=result.seconds,report=result.report,diagnostics=diagnostics,
            observability='Infrastructure check: eight synthetic observations per segment. This is not the four-landmark torso problem.'))
    return dict(length_series=deforming,length_reference_label="known",parameters=dict(noise=noise),times=times.tolist(),methods=dict(temporal=dict(frames=frames,
        summary={'Ceres parameter blocks':result.parameter_blocks,'Ceres residual blocks':result.residual_blocks},
        problem=dict(chain=True,parents=parents,connected=True,temporal=True,acceleration=True,axial_segments=[1,2] if axial else []),
        settings=dict(axial_length_bound_fractions=fit_settings.AXIAL_LENGTH_BOUND_FRACTIONS,axial_reference_lengths=references,length_prior_fraction=fit_settings.LENGTH_PRIOR_FRACTION,length_acceleration_scale=fit_settings.LENGTH_ACCELERATION_SCALE_MM_S2,position_scale_mm=fit_settings.POSITION_RESIDUAL_SCALE_MM,linear_motion_scale=fit_settings.ROOT_ACCELERATION_SCALE_MM_S2,angular_motion_scale=fit_settings.ANGULAR_ACCELERATION_SCALE_RAD_S2,scale_units=['mm/s^2','rad/s^2'],
            costs_by_family=dict(length_prior=result.length_prior_cost,length_acceleration=result.length_acceleration_cost,landmarks=result.landmark_cost,root_acceleration=result.root_acceleration_cost,segment_angular_acceleration=result.angular_acceleration_costs)),
        objective='General tree paths: each landmark residual uses root position and only its ancestor quaternions. Dense observations test assembly before sparse torso fitting.')))
