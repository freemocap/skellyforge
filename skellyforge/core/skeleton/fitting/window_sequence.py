"""Sequential fixed-lag fitting using the native Ceres objective.

No pose averaging or post-fit smoothing. Two committed frames remain constant
in each problem so acceleration residuals cross the moving boundary.
"""
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from skellyforge import _native
from .terminal_progress import TerminalProgress


BOUNDARY_FRAMES = 2
DEFAULT_MAX_ITERATIONS = _native.ChainSolveOptions().max_iterations
DEFAULT_FUNCTION_TOLERANCE = _native.ChainSolveOptions().function_tolerance


def frame_weights(times):
    dt=np.diff(times)
    if len(times)<3 or not np.isfinite(times).all() or not (dt>0).all():
        raise ValueError('At least three strictly increasing finite timestamps required')
    return (np.r_[dt,0]+np.r_[0,dt])/2


@dataclass
class WindowSequenceFit:
    final: object
    processing: dict
    seconds: float
    converged: bool
    report: str
    initial_quaternions: list
    initial_translations: list
    inspected_windows: dict | None = None

    def __getattr__(self,name):
        return getattr(self.final,name)


def fit_windows(arguments, *, active_frames=3, max_iterations=DEFAULT_MAX_ITERATIONS, function_tolerance=DEFAULT_FUNCTION_TOLERANCE, initial_function_tolerance=None, progress=None, fixed_length_segments=(), inspect_windows=()):
    if active_frames<3 or not isinstance(active_frames,int):
        raise ValueError('Active window must contain at least three frames')
    if arguments.get('allow_displacement'):
        raise ValueError('Window experiment uses general shoulder linkages, not legacy axial displacement')
    if 'solve_options' in arguments:
        raise ValueError('Window controller owns solve_options')
    args=dict(arguments)
    position_priors=args.pop('landmark_position_priors', [])
    huber_scale=args.pop('landmark_huber_scale_mm',0.)
    prior_targets=[p.targets for p in position_priors]
    def sliced_position_priors(lo,hi):
        result=[]
        for source,targets in zip(position_priors,prior_targets):
            p=_native.LandmarkPositionPrior()
            p.segment=source.segment;p.local_point=source.local_point;p.scale=source.scale
            p.targets=targets[lo:hi];result.append(p)
        return result
    times=np.asarray(args['times'],dtype=float);n=len(times)
    if any(len(targets)!=n for targets in prior_targets):
        raise ValueError('Position prior targets must match the full recording timestamps')
    if active_frames>n:raise ValueError('Window exceeds recording length')
    inspect_windows=set(inspect_windows)
    if any(not isinstance(i,int) or i<0 or i>n-active_frames for i in inspect_windows):
        raise ValueError('Inspected window index is outside the sequence')
    inspections={}
    if not args.get('initial_quaternions') or not args.get('initial_roots'):
        raise ValueError('Window fitting requires explicit segment/root initialization')
    weights=frame_weights(times)
    quaternions=deepcopy(args['initial_quaternions']);roots=deepcopy(args['initial_roots'])
    bodies=len(args['local']);references=args.get('axial_reference_lengths') or [0.]*bodies
    lengths=[list(references) for _ in times]
    displacements=[[[0.,0.,0.] for _ in range(bodies)] for _ in times]
    axis_prior=args.get('segment_axis_prior')
    axis_frames=axis_prior.frames if axis_prior is not None else None
    half_space=args.get('landmark_half_space_prior')
    half_frames=half_space.frames if half_space is not None else None
    prior=args.get('landmark_line_prior')
    # This native vector also copies on access; slice a single Python snapshot.
    line_frames=prior.frames if prior is not None else None
    initial_q=[None]*n;initial_t=[None]*n;introduced=0
    trace=[];window_seconds=0.;wall_start=perf_counter()
    total_windows = n-active_frames+1
    terminal = TerminalProgress(frames=n, segments=bodies, total=total_windows, active=active_frames,
        boundary=BOUNDARY_FRAMES, iterations=max_iterations, tolerance=function_tolerance)
    nonconverged = 0
    for first in range(total_windows):
        lo=max(0,first-BOUNDARY_FRAMES);hi=first+active_frames
        if first:
            # Newly arriving frame starts from the preceding connected solution.
            for values in (quaternions,roots,lengths,displacements):values[hi-1]=deepcopy(values[hi-2])
        window=dict(args)
        for name in ('times','observed','observation_indices'):
            if name in args:window[name]=args[name][lo:hi]
        window['initial_quaternions']=quaternions[lo:hi]
        window['initial_roots']=roots[lo:hi]
        if prior is not None:
            sliced=_native.LandmarkLinePrior()
            for name in ('segment','local_point','distance_scale','anterior_scale'):setattr(sliced,name,getattr(prior,name))
            sliced.frames=line_frames[lo:hi];window['landmark_line_prior']=sliced
        if half_space is not None:
            sliced_half=_native.LandmarkHalfSpacePrior()
            for name in ('segment','local_point','scale'):setattr(sliced_half,name,getattr(half_space,name))
            sliced_half.frames=half_frames[lo:hi];window['landmark_half_space_prior']=sliced_half
        if axis_prior is not None:
            sliced_axis=_native.SegmentAxisPrior()
            for name in ('segment','local_lateral','local_anterior','scale'):setattr(sliced_axis,name,getattr(axis_prior,name))
            sliced_axis.frames=axis_frames[lo:hi];window['segment_axis_prior']=sliced_axis
        options=_native.ChainSolveOptions()
        options.inspect_problem=first in inspect_windows
        options.landmark_position_priors=sliced_position_priors(lo,hi)
        options.landmark_huber_scale_mm=huber_scale
        options.fixed_length_segments=list(fixed_length_segments)
        options.initial_lengths=lengths[lo:hi];options.initial_linkage_displacements=displacements[lo:hi]
        options.frame_weights=weights[lo:hi].tolist();options.fixed_prefix_frames=first-lo
        options.function_tolerance=initial_function_tolerance if first==0 and initial_function_tolerance is not None else function_tolerance
        options.max_iterations=max_iterations;window['solve_options']=options
        started=perf_counter()
        try:
            result=_native.fit_chain_sequence(**window)
        except BaseException:
            terminal.clear()
            raise
        elapsed=perf_counter()-started
        terminal.clear()
        if not result.usable:
            terminal.line(f'FAILED: window {first+1}/{total_windows}, frames {lo}-{hi-1}', 'ERROR')
            raise RuntimeError(f"Window {first}: {result.report}")
        if options.inspect_problem:
            inspections[first]=dict(frame_start=lo,active_start=first,frame_end=hi-1,
                initial=result.problem_initial,final=result.problem_final)
        for target,source in ((quaternions,result.quaternions),(roots,result.roots),
                              (lengths,result.lengths),(displacements,result.linkage_displacements)):
            # Fixed history is immutable, including lengths and shoulder displacements.
            if not np.array_equal(np.asarray(target[lo:first]),np.asarray(source[:first-lo])):
                raise RuntimeError('Native solve changed a committed boundary state')
            target[first:hi]=deepcopy(source[first-lo:])
        for index in range(introduced,hi):
            initial_q[index]=result.initial_quaternions[index-lo]
            initial_t[index]=result.initial_translations[index-lo]
        introduced=hi;window_seconds+=result.seconds
        trace.append(dict(index=len(trace),fixed_start=lo,active_start=first,active_end=hi-1,
            committed_end=hi-1 if hi==n else first,seconds=result.seconds,wall_seconds=elapsed,
            iterations=result.iterations,converged=result.converged,initial_cost=result.costs[0],final_cost=result.costs[-1],
            parameter_blocks=result.parameter_blocks,residual_blocks=result.residual_blocks,
            report=result.report,full_report=result.full_report))
        nonconverged += not result.converged
        if progress:progress(trace[-1],total_windows)
        terminal.update(trace[-1], unconverged=nonconverged)
    options=_native.ChainSolveOptions()
    options.landmark_position_priors=sliced_position_priors(0,n)
    options.landmark_huber_scale_mm=huber_scale
    options.fixed_length_segments=list(fixed_length_segments)
    options.initial_lengths=lengths;options.initial_linkage_displacements=displacements
    options.frame_weights=weights.tolist();options.max_iterations=max_iterations
    options.function_tolerance=function_tolerance
    options.evaluate_only=True
    terminal.line('Evaluating full sequence (no optimization; excluded from window ETA)', 'INFO')
    started=perf_counter()
    final=_native.fit_chain_sequence(**{**args,'initial_quaternions':quaternions,'initial_roots':roots,'solve_options':options})
    final_wall=perf_counter()-started
    if not final.usable:
        terminal.line('FAILED: full-sequence evaluation', 'ERROR')
        raise RuntimeError(final.report)
    processing=dict(active_frames=active_frames,boundary_frames=BOUNDARY_FRAMES,refined=False,
        frame_weights=weights.tolist(),max_iterations=max_iterations,function_tolerance=function_tolerance,windows=trace,
        initial_function_tolerance=initial_function_tolerance,
        window_solve_seconds=window_seconds,final_pass_seconds=final.seconds,final_pass_wall_seconds=final_wall,
        final_pass_report=final.full_report,wall_seconds=perf_counter()-wall_start,
        evaluation_scope='Final assembled trajectory scored once under the full-sequence objective; window costs are not summed.',
        initialization='Saved segment/root poses for first window; overlapping states retained; new frames copy the preceding fitted root, quaternions, lengths and shoulder displacements.',
        boundary='Two preceding committed frames fixed with SetParameterBlockConstant; active frames remain adjustable. No marginalization or averaging.',
        latency='At least active_frames minus one intervals of pose lookahead. This offline prototype uses full-recording timestamp weights, prepared scale and initialization; it is not an end-to-end live pipeline.')
    if fixed_length_segments:processing['fixed_length_segments']=list(fixed_length_segments)
    converged=all(w['converged'] for w in trace)
    terminal.finish(trace, final_wall)
    report=f'{len(trace)} sequential windows; {sum(w["converged"] for w in trace)} converged. '+'Final full-sequence evaluation only; no global optimization.'
    return WindowSequenceFit(final,processing,window_seconds+final.seconds,converged,report,
                             initial_q,initial_t,inspections)


def refine_window_result(arguments, window_result, *, max_iterations=DEFAULT_MAX_ITERATIONS, function_tolerance=DEFAULT_FUNCTION_TOLERANCE):
    """One post-hoc solve initialized by every windowed parameter; no repeated windows."""
    source=window_result.final
    options=_native.ChainSolveOptions()
    arguments=dict(arguments)
    options.landmark_position_priors=arguments.pop('landmark_position_priors', [])
    options.landmark_huber_scale_mm=arguments.pop('landmark_huber_scale_mm',0.)
    options.fixed_length_segments=window_result.processing.get('fixed_length_segments',[])
    options.initial_lengths=source.lengths;options.initial_linkage_displacements=source.linkage_displacements
    options.frame_weights=frame_weights(arguments['times']).tolist();options.max_iterations=max_iterations
    options.function_tolerance=function_tolerance
    started=perf_counter()
    final=_native.fit_chain_sequence(**{**arguments,'initial_quaternions':source.quaternions,
        'initial_roots':source.roots,'solve_options':options})
    elapsed=perf_counter()-started
    if not final.usable:raise RuntimeError(final.report)
    processing=deepcopy(window_result.processing)
    processing.update(refined=True,final_pass_seconds=final.seconds,final_pass_wall_seconds=elapsed,
                      final_pass_report=final.full_report,wall_seconds=processing['wall_seconds']+elapsed,
                      refinement_max_iterations=max_iterations,refinement_function_tolerance=function_tolerance)
    return WindowSequenceFit(final,processing,window_result.seconds+final.seconds,final.converged,
        window_result.report+' Post-hoc refinement: '+final.report,final.initial_quaternions,final.initial_translations,window_result.inspected_windows)
