"""Analytic full-mesh framing for fixed camera orientation and desired placement.

Input coordinates are object-frame millimetres. Camera axes are right/down/forward,
column-vector camera_to_object transforms; pixels use image-edge origin. Supply
ALL mesh vertices, never only selected training features. No rendering occurs.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeshFraming:
    camera_to_object: np.ndarray
    target_position_mm: np.ndarray
    requested_center_pixels: np.ndarray
    distance_mm: float
    required_distance_mm: float
    projected_bounds_pixels: np.ndarray
    depth_bounds_mm: np.ndarray
    bbox_fraction_xy: np.ndarray
    bbox_area_fraction: float
    minimum_edge_margin_fraction: np.ndarray
    small_occupancy: bool


def fit_mesh_camera(mesh_vertices_mm: np.ndarray, camera_rotation: np.ndarray,
                    camera_intrinsics: np.ndarray, image_size: tuple[int,int],
                    minimum_distance_mm: float, image_center_fraction: tuple[float,float] = (.5,.5),
                    margin_fraction: float = .1, near_clip_mm: float = .01,
                    far_clip_mm: float | None = None, target_position_mm: np.ndarray | None = None,
                    small_occupancy_threshold: float = .15) -> MeshFraming:
    """Increase target depth only as necessary to contain every mesh vertex.

    The desired target projection and orientation/roll stay fixed. Off-centre
    placements can force a smaller object; report that via bounding-box occupancy
    instead of silently changing placement. `small_occupancy` flags when BOTH
    projected dimensions occupy less than the configured fraction of the image.
    For perspective triangles with positive depth, projected edges are straight,
    so containing every mesh vertex also contains every triangle.

    `minimum_distance_mm` is target depth along camera +Z (not Euclidean camera
    distance when placement is off-centre). No far plane is silently extended.
    """
    vertices=np.asarray(mesh_vertices_mm,dtype=np.float64)
    rotation=np.asarray(camera_rotation,dtype=np.float64)
    intrinsics=np.asarray(camera_intrinsics,dtype=np.float64)
    dimensions=np.asarray(image_size,dtype=np.float64)
    desired_fraction=np.asarray(image_center_fraction,dtype=np.float64)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<3 or not np.isfinite(vertices).all():
        raise ValueError('Supply finite full mesh vertices with shape [vertex,3]')
    if rotation.shape!=(3,3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6):
        raise ValueError('Camera rotation must be a proper orthonormal matrix')
    # Renderer transforms may arrive as float32. Remove only their tiny numerical
    # orthogonality error before solving distance, without accepting arbitrary scale.
    left_axes,_,right_axes=np.linalg.svd(rotation)
    rotation=left_axes@right_axes
    if intrinsics.shape!=(3,3) or not np.isfinite(intrinsics).all() or not np.allclose(intrinsics[2],[0,0,1]) or np.linalg.det(intrinsics)<=0 or np.any(np.diag(intrinsics)[:2]<=0):
        raise ValueError('Camera intrinsics must be finite invertible perspective intrinsics')
    if dimensions.shape!=(2,) or not np.isfinite(dimensions).all() or np.any(dimensions<=0):
        raise ValueError('Image dimensions must be positive width and height')
    if not np.isfinite(margin_fraction) or not 0<=margin_fraction<.5:
        raise ValueError('Margin fraction must be between zero and one half')
    if desired_fraction.shape!=(2,) or not np.isfinite(desired_fraction).all() or np.any(desired_fraction<=margin_fraction) or np.any(desired_fraction>=1-margin_fraction):
        raise ValueError('Requested centre must lie strictly inside the allowed margin rectangle')
    if not np.isfinite(minimum_distance_mm) or minimum_distance_mm<0 or not np.isfinite(near_clip_mm) or near_clip_mm<=0:
        raise ValueError('Minimum distance must be nonnegative and near plane positive')
    if far_clip_mm is not None and (not np.isfinite(far_clip_mm) or far_clip_mm<=near_clip_mm):
        raise ValueError('Far plane must be finite and beyond the near plane')
    if not np.isfinite(small_occupancy_threshold) or not 0<=small_occupancy_threshold<=1:
        raise ValueError('Small occupancy threshold must be a fraction in [0,1]')
    target=(vertices.min(0)+vertices.max(0))*.5 if target_position_mm is None else np.asarray(target_position_mm,dtype=np.float64)
    if target.shape!=(3,) or not np.isfinite(target).all():
        raise ValueError('Target position must be a finite object-frame point')
    desired_pixels=desired_fraction*dimensions
    target_ray=np.linalg.solve(intrinsics,np.r_[desired_pixels,1.])
    relative_camera=(vertices-target)@rotation
    homogeneous=relative_camera@intrinsics.T
    numerators=homogeneous[:,:2]-relative_camera[:,2,None]*desired_pixels
    lower=margin_fraction*dimensions;upper=(1-margin_fraction)*dimensions
    room=np.where(numerators>=0,upper-desired_pixels,desired_pixels-lower)
    per_vertex_distance=np.abs(numerators)/room-relative_camera[:,2,None]
    numerical_guard=max(float(np.ptp(vertices,axis=0).max()),near_clip_mm,1.)*1e-7
    required=max(float(per_vertex_distance.max()),near_clip_mm-float(relative_camera[:,2].min()))+numerical_guard
    distance=max(float(minimum_distance_mm),required)
    camera=np.eye(4);camera[:3,:3]=rotation;camera[:3,3]=target-rotation@(target_ray*distance)
    camera_points=(vertices-camera[:3,3])@rotation
    depth_bounds=np.array([camera_points[:,2].min(),camera_points[:,2].max()])
    if depth_bounds[0]<=near_clip_mm or (far_clip_mm is not None and depth_bounds[1]>=far_clip_mm):
        raise ValueError(f'Whole-mesh framing cannot fit clipping planes: depth bounds {depth_bounds.tolist()} mm')
    pixels_homogeneous=camera_points@intrinsics.T
    pixels=pixels_homogeneous[:,:2]/pixels_homogeneous[:,2,None]
    bounds=np.stack((pixels.min(0),pixels.max(0)))
    if not np.isfinite(pixels).all() or np.any(bounds[0]<lower-1e-5) or np.any(bounds[1]>upper+1e-5):
        raise ValueError('Numerical full-mesh framing verification failed')
    occupancy=(bounds[1]-bounds[0])/dimensions
    edge_margin=np.minimum(bounds[0]/dimensions,1-bounds[1]/dimensions)
    return MeshFraming(camera,target.copy(),desired_pixels,distance,required,bounds,depth_bounds,occupancy,
                       float(np.prod(occupancy)),edge_margin,bool(occupancy.max()<small_occupancy_threshold))
