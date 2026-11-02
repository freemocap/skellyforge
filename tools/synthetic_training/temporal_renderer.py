"""Render calibrated fixed cameras observing one smooth moving rigid object."""
import argparse
from contextlib import suppress
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from randomized_renderer import render_preview
from render_dataset import FeatureDefinition, file_hash, project_features


def temporal_geometry(feature_positions: np.ndarray, frame_count: int = 240,
                      frame_rate: int = 30, random_seed: int = 2026110207) -> dict:
    timestamps=np.arange(frame_count,dtype=np.float64)/frame_rate
    phases=np.random.default_rng(random_seed).uniform(-np.pi,np.pi,6)
    angles=np.deg2rad([25,30,45])*np.sin(timestamps[:,None]*[.73,.91,.57]+phases[:3])
    rotations=[]
    for axis_index in range(3):
        angle_values=angles[:,axis_index]
        cosine,sine=np.cos(angle_values),np.sin(angle_values)
        axis_rotation=np.broadcast_to(np.eye(3),(frame_count,3,3)).copy()
        first_axis,second_axis=(axis_index+1)%3,(axis_index+2)%3
        axis_rotation[:,first_axis,first_axis]=cosine
        axis_rotation[:,second_axis,second_axis]=cosine
        axis_rotation[:,first_axis,second_axis]=-sine
        axis_rotation[:,second_axis,first_axis]=sine
        rotations.append(axis_rotation)
    object_to_world=np.broadcast_to(np.eye(4),(frame_count,4,4)).copy()
    object_to_world[:,:3,:3]=rotations[2]@rotations[1]@rotations[0]
    object_center=(feature_positions.min(axis=0)+feature_positions.max(axis=0))*.5
    center_motion=np.array([8,6,5])*np.sin(timestamps[:,None]*[.67,.83,1.03]+phases[3:])
    object_to_world[:,:3,3]=center_motion-np.einsum('fij,j->fi',object_to_world[:,:3,:3],object_center)
    azimuth=np.deg2rad([0,90,180,270]); elevation=np.deg2rad([25,35,45,30])
    camera_count=len(azimuth)
    camera_origins=240*np.stack([np.cos(elevation)*np.cos(azimuth),np.cos(elevation)*np.sin(azimuth),np.sin(elevation)],axis=-1)
    forward_axes=-camera_origins/np.linalg.norm(camera_origins,axis=-1,keepdims=True)
    right_axes=np.cross(forward_axes,np.array([0,0,1]))
    right_axes/=np.linalg.norm(right_axes,axis=-1,keepdims=True)
    down_axes=np.cross(forward_axes,right_axes)
    camera_to_world=np.broadcast_to(np.eye(4),(camera_count,4,4)).copy()
    camera_to_world[:,:3,:3]=np.stack([right_axes,down_axes,forward_axes],axis=-1)
    camera_to_world[:,:3,3]=camera_origins
    world_to_object=np.linalg.inv(object_to_world)
    camera_to_object=world_to_object[:,None]@camera_to_world[None]
    focal_length=320/np.tan(np.deg2rad(19))
    intrinsics=np.broadcast_to([[focal_length,0,320],[0,focal_length,240],[0,0,1]],(camera_count,3,3)).copy()
    world_light=np.array([150.,-100.,200.])
    object_light=np.einsum('fij,j->fi',world_to_object[:,:3,:3],world_light)+world_to_object[:,:3,3]
    projected,valid=project_features(feature_positions,camera_to_object.reshape(-1,4,4),
                                     np.broadcast_to(intrinsics,(frame_count,camera_count,3,3)).reshape(-1,3,3),(640,480),.01,1000.)
    if not valid.all():
        raise ValueError("Temporal trajectory must keep every feature in every camera view")
    return dict(timestamps_seconds=timestamps,camera_ids=np.asarray([f'camera_{index:03d}' for index in range(camera_count)]),
                camera_to_world=camera_to_world,object_to_world=object_to_world,camera_to_object=camera_to_object,
                camera_intrinsics=intrinsics,projected_xy_pixels=projected.reshape(frame_count,camera_count,-1,2),
                projection_valid=valid.reshape(frame_count,camera_count,-1),light_positions_object_mm=object_light,
                light_position_world_mm=world_light,rotation_angles_radians=angles,object_center_positions_world_mm=center_motion)


def encode_lossless(image_paths: list[Path], destination: Path, frame_rate: int, ffmpeg: str) -> dict:
    """Verify every decoded RGB byte; no inference or GPU codec is used."""
    with Image.open(image_paths[0]) as first_image:
        width,height=first_image.size
    encoder=subprocess.Popen([ffmpeg,'-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
                              '-framerate',str(frame_rate),'-i','pipe:0','-an','-c:v','ffv1','-threads','4','-level','3','-pix_fmt','bgr0',str(destination)],stdin=subprocess.PIPE)
    expected_hash=hashlib.sha256()
    try:
        for image_path in image_paths:
            with Image.open(image_path) as image_value:
                rgb_bytes=np.asarray(image_value.convert('RGB')).tobytes()
            expected_hash.update(rgb_bytes)
            encoder.stdin.write(rgb_bytes)
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
    finally:
        with suppress(OSError):
            encoder.stdin.close()
    if encoder.wait()!=0:
        raise RuntimeError('Lossless video encoding failed')
    decoder=subprocess.Popen([ffmpeg,'-v','error','-threads','4','-i',str(destination),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
    decoded_hash=hashlib.sha256(); byte_count=0
    try:
        while chunk:=decoder.stdout.read(1024*1024):
            decoded_hash.update(chunk); byte_count+=len(chunk)
    except BaseException:
        decoder.kill()
        decoder.wait()
        raise
    finally:
        decoder.stdout.close()
    if decoder.wait()!=0 or byte_count!=len(image_paths)*width*height*3 or decoded_hash.digest()!=expected_hash.digest():
        raise ValueError('Lossless decoded video differs from original RGB image sequence')
    return dict(path=destination.name,frame_count=len(image_paths),width=width,height=height,fps=frame_rate,
                codec='FFV1',pixel_format='bgr0',decoded_rgb_sha256=decoded_hash.hexdigest(),sha256=file_hash(destination),pixel_identity_verified=True)


def render_temporal(mesh_path: Path, feature_path: Path, output_path: Path, cpu_threads: int = 4) -> dict:
    import drjit as dr
    dr.set_thread_count(cpu_threads)
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('Existing ffmpeg executable is required; no installation is performed')
    if output_path.exists():
        raise FileExistsError(output_path)
    definition=FeatureDefinition.load_file(feature_path)
    geometry=temporal_geometry(definition.feature_positions_mm)
    frame_count,camera_count=geometry['camera_to_object'].shape[:2]
    parameter_rows=[dict(azimuth_radians=0.,elevation_radians=.5,roll_radians=0.,distance_mm=240.,image_center_fraction=[.5,.5],roughness=.2) for _ in range(frame_count*camera_count)]
    for camera_index in range(camera_count):
        for frame_index in range(frame_count):
            image_index=camera_index*frame_count+frame_index
            parameter_rows[image_index].update(pose_group=f'temporal_{frame_index:06d}',background_group=f'temporal_camera_{camera_index}',
                split='temporal_replay',color_rgb=[.13,.43,.68],material_kind='diffuse',ambient_rgb=[.35]*3,
                light_position_mm=geometry['light_positions_object_mm'][frame_index].tolist(),light_rgb=[18000]*3,
                background_kind='gradient_noise',background_seed=2026110300+camera_index,render_seed=2026110400+camera_index)
    output_path.mkdir(parents=True)
    snapshot=output_path/'source_snapshot';snapshot.mkdir()
    for name in ['temporal_renderer.py','randomized_renderer.py','render_dataset.py']:
        (snapshot/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (snapshot/'features_50.json').write_bytes(feature_path.read_bytes())
    started=time.perf_counter()
    rendered=render_preview(mesh_path,feature_path,output_path/'rendered',parameters_override=parameter_rows,create_review=False,
                            camera_to_object_override=geometry['camera_to_object'].transpose(1,0,2,3).reshape(-1,4,4))
    with np.load(output_path/'rendered/labels.npz',allow_pickle=False) as labels:
        expected_pixels=labels['projected_xy_pixels'].reshape(camera_count,frame_count,-1,2).transpose(1,0,2,3)
        np.testing.assert_allclose(expected_pixels,geometry['projected_xy_pixels'],atol=1e-10,rtol=0)
        geometry['visibility']=labels['visibility'].reshape(camera_count,frame_count,-1).transpose(1,0,2)
        geometry['image_names']=np.array([f'rendered/{name}' for name in labels['image_names']]).reshape(camera_count,frame_count).T
    geometry.update(feature_names=definition.feature_names,feature_positions_mm=definition.feature_positions_mm)
    np.savez_compressed(output_path/'ground_truth.npz',**geometry)
    videos=[]
    for camera_index,camera_id in enumerate(geometry['camera_ids']):
        paths=[output_path/str(name) for name in geometry['image_names'][:,camera_index]]
        video=encode_lossless(paths,output_path/f'{camera_id}.mkv',30,ffmpeg);video['camera_id']=str(camera_id);videos.append(video)
    manifest=dict(purpose='Deterministic held-out temporal replay; not part of reserved final dataset',random_seed=2026110207,
                  frame_count=frame_count,camera_count=camera_count,feature_count=len(definition.feature_names),videos=videos,
                  ground_truth_path='ground_truth.npz',ground_truth_sha256=file_hash(output_path/'ground_truth.npz'),
                  mesh_sha256=file_hash(mesh_path),feature_definition_sha256=file_hash(feature_path),
                  source_snapshot_hashes={path.name:file_hash(path) for path in snapshot.iterdir()},
                  frame_convention='column vectors; camera axes right/down/forward; camera_to_object=inv(object_to_world)@camera_to_world',
                  world_frame='right-handed x/y horizontal,z up; initial object frame follows source feature definition',units='mm',
                  pixel_convention=rendered['pixel_convention'],timestamps='frame_index/30 seconds; 240frames; no exposure timing simulation',
                  light_convention='fixed world point light transformed to object frame; isotropic fixed ambient illumination',
                  limitations=['No motion blur, sensor noise or rolling shutter','No physical interaction with composited backgrounds','Geometry feature visibility distinct from projection validity'],
                  generation_seconds=time.perf_counter()-started,all_projection_valid=bool(geometry['projection_valid'].all()))
    (output_path/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path,required=True);parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args()
    print(json.dumps(render_temporal(arguments.mesh,arguments.features,arguments.output),indent=2))
