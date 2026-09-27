"""Sequential fixed-lag experiments using the existing native Ceres objective.

No pose averaging or post-fit smoothing. Two committed frames remain constant
in each problem so acceleration residuals cross the moving boundary.
"""
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from skellyforge import _native

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

    def __getattr__(self,name):
        return getattr(self.final,name)


def fit_windows(arguments, *, active_frames=3, max_iterations=DEFAULT_MAX_ITERATIONS, function_tolerance=DEFAULT_FUNCTION_TOLERANCE, initial_function_tolerance=None, progress=None):
    if active_frames<3 or not isinstance(active_frames,int):
        raise ValueError('Active window must contain at least three frames')
    if arguments.get('allow_displacement'):
        raise ValueError('Window experiment uses general shoulder linkages, not legacy axial displacement')
    if 'solve_options' in arguments:
        raise ValueError('Window controller owns solve_options')
    args=dict(arguments);times=np.asarray(args['times'],dtype=float);n=len(times)
    if active_frames>n:raise ValueError('Window exceeds recording length')
    if not args.get('initial_quaternions') or not args.get('initial_roots'):
        raise ValueError('Window fitting requires explicit segment/root initialization')
    weights=frame_weights(times)
    quaternions=deepcopy(args['initial_quaternions']);roots=deepcopy(args['initial_roots'])
    bodies=len(args['local']);references=args.get('axial_reference_lengths') or [0.]*bodies
    lengths=[list(references) for _ in times]
    displacements=[[[0.,0.,0.] for _ in range(bodies)] for _ in times]
    prior=args.get('landmark_line_prior')
    # This native vector also copies on access; slice a single Python snapshot.
    line_frames=prior.frames if prior is not None else None
    initial_q=[None]*n;initial_t=[None]*n;introduced=0
    trace=[];window_seconds=0.;wall_start=perf_counter()
    for first in range(n-active_frames+1):
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
        options=_native.ChainSolveOptions()
        options.initial_lengths=lengths[lo:hi];options.initial_linkage_displacements=displacements[lo:hi]
        options.frame_weights=weights[lo:hi].tolist();options.fixed_prefix_frames=first-lo
        options.function_tolerance=initial_function_tolerance if first==0 and initial_function_tolerance is not None else function_tolerance
        options.max_iterations=max_iterations;window['solve_options']=options
        started=perf_counter();result=_native.fit_chain_sequence(**window);elapsed=perf_counter()-started
        if not result.usable:raise RuntimeError(f"Window {first}: {result.report}")
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
        if progress:progress(trace[-1],n-active_frames+1)
    options=_native.ChainSolveOptions()
    options.initial_lengths=lengths;options.initial_linkage_displacements=displacements
    options.frame_weights=weights.tolist();options.max_iterations=max_iterations
    options.function_tolerance=function_tolerance
    options.evaluate_only=True
    started=perf_counter()
    final=_native.fit_chain_sequence(**{**args,'initial_quaternions':quaternions,'initial_roots':roots,'solve_options':options})
    final_wall=perf_counter()-started
    if not final.usable:raise RuntimeError(final.report)
    processing=dict(active_frames=active_frames,boundary_frames=BOUNDARY_FRAMES,refined=False,
        frame_weights=weights.tolist(),max_iterations=max_iterations,function_tolerance=function_tolerance,windows=trace,
        initial_function_tolerance=initial_function_tolerance,
        window_solve_seconds=window_seconds,final_pass_seconds=final.seconds,final_pass_wall_seconds=final_wall,
        final_pass_report=final.full_report,wall_seconds=perf_counter()-wall_start,
        evaluation_scope='Final assembled trajectory scored once under the full-sequence objective; window costs are not summed.',
        initialization='Saved segment/root poses for first window; overlapping states retained; new frames copy the preceding fitted root, quaternions, lengths and shoulder displacements.',
        boundary='Two preceding committed frames fixed with SetParameterBlockConstant; active frames remain adjustable. No marginalization or averaging.',
        latency='At least active_frames minus one intervals of pose lookahead. This offline prototype uses full-recording timestamp weights, prepared scale and initialization; it is not an end-to-end live pipeline.')
    converged=all(w['converged'] for w in trace)
    report=f'{len(trace)} sequential windows; {sum(w["converged"] for w in trace)} converged. '+'Final full-sequence evaluation only; no global optimization.'
    return WindowSequenceFit(final,processing,window_seconds+final.seconds,converged,report,
                             initial_q,initial_t)


def refine_window_result(arguments, window_result, *, max_iterations=DEFAULT_MAX_ITERATIONS, function_tolerance=DEFAULT_FUNCTION_TOLERANCE):
    """One post-hoc solve initialized by every windowed parameter; no repeated windows."""
    source=window_result.final
    options=_native.ChainSolveOptions()
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
        window_result.report+' Post-hoc refinement: '+final.report,final.initial_quaternions,final.initial_translations)
