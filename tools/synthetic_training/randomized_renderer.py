"""CPU-only domain-randomized preview; generic mesh/feature input, no training.

Foreground coverage uses Mitsuba RGBA; depth uses compiled Embree arrays.
The much slower AOV alternative was checked [@mitsubaForegroundAov].
"""
from __future__ import annotations

import argparse
import colorsys
from importlib.metadata import version
from functools import lru_cache
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from render_dataset import FeatureDefinition, SurfaceVisibility, file_hash, project_features, read_binary_stl
from mesh_framing import fit_mesh_camera


def gradient_noise(image_shape: tuple[int, int], grid_size: int, random_seed: int) -> np.ndarray:
    """Vectorized gradient-lattice noise with quintic interpolation; no pixel loops."""
    random_generator = np.random.default_rng(random_seed)
    image_height, image_width = image_shape
    grid_angles = random_generator.uniform(0, 2*np.pi, (grid_size+1, grid_size+1))
    gradient_vectors = np.stack((np.cos(grid_angles), np.sin(grid_angles)), axis=-1)
    pixel_y, pixel_x = np.meshgrid(np.linspace(0, grid_size, image_height, endpoint=False),
                                  np.linspace(0, grid_size, image_width, endpoint=False), indexing="ij")
    cell_x, cell_y = pixel_x.astype(int), pixel_y.astype(int)
    local_x, local_y = pixel_x-cell_x, pixel_y-cell_y
    smooth_x = local_x**3*(local_x*(local_x*6-15)+10)
    smooth_y = local_y**3*(local_y*(local_y*6-15)+10)
    corner_offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    corner_gradients = gradient_vectors[cell_y[..., None]+corner_offsets[:, 1], cell_x[..., None]+corner_offsets[:, 0]]
    local_offsets = np.stack((local_x, local_y), axis=-1)[..., None, :]-corner_offsets
    corner_values = np.sum(corner_gradients*local_offsets, axis=-1)
    upper_values = corner_values[..., 0]*(1-smooth_x)+corner_values[..., 1]*smooth_x
    lower_values = corner_values[..., 2]*(1-smooth_x)+corner_values[..., 3]*smooth_x
    return upper_values*(1-smooth_y)+lower_values*smooth_y


def procedural_background(image_shape: tuple[int, int], background_kind: str, random_seed: int) -> np.ndarray:
    """Return linear RGB. Generated scenes are synthetic graphics, not photographs."""
    random_generator = np.random.default_rng(random_seed)
    image_height, image_width = image_shape
    color_first, color_second = random_generator.uniform(.02, .8, (2, 3))
    if background_kind == "solid_reference":
        return np.full((*image_shape,3),.35)
    if background_kind == "white_noise":
        texture_values = random_generator.uniform(0, 1, (*image_shape, 1))
        return color_first + texture_values*(color_second-color_first)
    if background_kind == "gradient_noise":
        texture_values = (gradient_noise(image_shape, 4, random_seed)+.5*gradient_noise(image_shape, 11, random_seed+1))
        texture_values = np.clip(.5+texture_values, 0, 1)[..., None]
        return color_first+texture_values*(color_second-color_first)
    if background_kind == "generated_scene":
        # A seeded perspective floor plus repeated cabinets/panels supplies scene-like edges.
        pixel_y, pixel_x = np.indices(image_shape)
        horizon_row = int(random_generator.uniform(.25, .65)*image_height)
        image_array = np.broadcast_to(color_first, (*image_shape, 3)).copy()
        floor_region = pixel_y > horizon_row
        floor_grid = (((pixel_x-image_width/2)/(np.maximum(pixel_y-horizon_row, 1)/image_height+.1)/80).astype(int)
                      + (np.log1p(np.maximum(pixel_y-horizon_row, 0))/0.5).astype(int))%2
        image_array[floor_region] = (color_second*(.45+.5*floor_grid[..., None]))[floor_region]
        panel_mask = ((pixel_x//max(1,image_width//8))%2 == 0) & (pixel_y>image_height*.08) & (pixel_y<horizon_row*.85)
        image_array[panel_mask] = color_second*.6
        return image_array
    raise ValueError(f"Unknown background kind: {background_kind}")


def composite_foreground(premultiplied_rgb: np.ndarray, surface_coverage: np.ndarray,
                         surface_depth_mm: np.ndarray, background_rgb: np.ndarray) -> np.ndarray:
    """Composite linear foreground only where the render reports surface depth.

    Depth is measured at pixel centres, coverage integrates subpixel samples.
    Thus fractional coverage can coexist with a zero centre-ray depth at edges.
    Background is infinitely far; no external occlusion is added.
"""
    if premultiplied_rgb.shape != background_rgb.shape or surface_coverage.shape != background_rgb.shape[:2] or surface_depth_mm.shape != surface_coverage.shape:
        raise ValueError("Foreground, coverage, depth and background layouts disagree")
    if not all(np.isfinite(array).all() for array in (premultiplied_rgb, surface_coverage, surface_depth_mm, background_rgb)):
        raise ValueError("Compositing arrays must be finite")
    if np.any(surface_coverage < -1e-6) or np.any(surface_coverage > 1+1e-6):
        raise ValueError("Coverage must be in [0,1]")
    if np.any(surface_depth_mm < 0):
        raise ValueError("Surface depth must be nonnegative; zero means no centre-ray hit")
    coverage_values = np.clip(surface_coverage, 0, 1)
    return premultiplied_rgb + (1-coverage_values[..., None])*background_rgb


def pixel_depth(visibility_scene: SurfaceVisibility, camera_matrix: np.ndarray,
                intrinsic_matrix: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """One native array query for physical camera-to-surface distances in mm."""
    image_height,image_width=image_shape
    pixel_y,pixel_x=np.indices(image_shape,dtype=np.float64)
    camera_rays=np.stack(((pixel_x+.5-intrinsic_matrix[0,2])/intrinsic_matrix[0,0],
                          (pixel_y+.5-intrinsic_matrix[1,2])/intrinsic_matrix[1,1],np.ones(image_shape)),axis=-1).reshape(-1,3)
    object_rays=camera_rays@camera_matrix[:3,:3].T
    object_rays/=np.linalg.norm(object_rays,axis=1,keepdims=True)
    ray_origins=np.ascontiguousarray(np.broadcast_to(camera_matrix[:3,3],object_rays.shape),dtype=np.float32)
    ray_distances=visibility_scene.embree_scene.run(ray_origins,np.ascontiguousarray(object_rays,dtype=np.float32),query="DISTANCE")
    return np.where(np.isfinite(ray_distances)&(ray_distances>0)&(ray_distances<1000),ray_distances,0).reshape(image_height,image_width)


def srgb_bytes(linear_rgb: np.ndarray) -> np.ndarray:
    bounded_rgb = np.clip(linear_rgb, 0, 1)
    srgb_values = np.where(bounded_rgb <= .0031308, bounded_rgb*12.92, 1.055*bounded_rgb**(1/2.4)-.055)
    return np.round(srgb_values*255).astype(np.uint8)


def load_photos(photo_directory: Path, source_manifest: Path) -> tuple[list[dict], dict[str, Path]]:
    source_rows=json.loads(source_manifest.read_text(encoding="utf-8"))
    source_owners={}
    family_owners={}
    image_hashes=set()
    photo_images={}
    for source_row in source_rows:
        if source_row["license"] not in {"CC0-1.0","CC-BY-4.0"} or not source_row.get("source_url") or not source_row.get("attribution"):
            raise ValueError("Photo source needs reviewed permission, source URL and attribution")
        source_identity=source_row["source_id"]
        if source_identity in source_owners:
            raise ValueError("Photo source identities must be unique and assigned to one split")
        source_owners[source_identity]=source_row["split"]
        for family_field in ("source_family","scene_group"):
            family_identity=(family_field,source_row.get(family_field,source_identity))
            if family_identity in family_owners and family_owners[family_identity]!=source_row["split"]:
                raise ValueError("Related photographic source families cannot cross splits")
            family_owners[family_identity]=source_row["split"]
        if source_row["sha256"] in image_hashes:
            raise ValueError("Duplicate photographic image bytes cannot count as distinct sources")
        image_hashes.add(source_row["sha256"])
        photo_path=photo_directory/source_row["file_name"]
        if photo_path.resolve().parent != photo_directory.resolve() or file_hash(photo_path) != source_row["sha256"]:
            raise ValueError("Photo path or hash does not match the reviewed source")
        with Image.open(photo_path) as source_image:
            if list(source_image.size)!=source_row["dimensions"]:
                raise ValueError("Photo dimensions do not match provenance")
            photo_images[source_identity]=photo_path
    return source_rows,photo_images


@lru_cache(maxsize=4)
def cached_photo(photo_path: Path) -> Image.Image:
    """Bound decoded source memory independently of the source-pool size."""
    with Image.open(photo_path) as source_image:
        return source_image.convert("RGB")


def photo_background(source_image: Image.Image, image_shape: tuple[int,int], random_seed: int) -> tuple[np.ndarray,list[int]]:
    random_generator=np.random.default_rng(random_seed)
    image_height,image_width=image_shape
    source_width,source_height=source_image.size
    crop_scale=random_generator.uniform(.65,1)
    crop_width=max(1,int(min(source_width,source_height*image_width/image_height)*crop_scale))
    crop_height=max(1,int(crop_width*image_height/image_width))
    crop_left=int(random_generator.integers(0,source_width-crop_width+1))
    crop_top=int(random_generator.integers(0,source_height-crop_height+1))
    crop_box=[crop_left,crop_top,crop_left+crop_width,crop_top+crop_height]
    srgb_values=np.asarray(source_image.crop(crop_box).resize((image_width,image_height),Image.Resampling.LANCZOS),dtype=np.float64)/255
    linear_values=np.where(srgb_values<=.04045,srgb_values/12.92,((srgb_values+.055)/1.055)**2.4)
    return linear_values,crop_box


def preview_parameters(image_count: int, random_seed: int, object_extent_mm: float) -> list[dict]:
    if image_count < 36 or image_count%36:
        raise ValueError("Use a multiple of 36 images to balance all three full-sphere preview splits")
    random_generator = np.random.default_rng(random_seed)
    parameter_rows = []
    # Each independent pose/background group belongs wholly to one split. Every
    # split includes below/side/above strata; no elevation range is held out.
    split_names = ("training", "development", "final_evaluation")
    for image_index in range(image_count):
        stratum_index = image_index%12
        split_index = image_index//12%3
        azimuth_angle = (stratum_index%4)*np.pi/2 + random_generator.uniform(-np.pi/4, np.pi/4)
        elevation_band = stratum_index//4
        elevation_angle = np.arcsin(random_generator.uniform(-1+elevation_band*2/3, -1+(elevation_band+1)*2/3))
        roll_angle = random_generator.uniform(-np.pi, np.pi)
        color_rgb = colorsys.hsv_to_rgb(random_generator.uniform(0, 1), random_generator.uniform(.25, .95), random_generator.uniform(.12, .85))
        parameter_rows.append({"pose_group": f"pose_{image_index:06d}", "background_group": f"background_{image_index:06d}",
                               "split": split_names[split_index], "azimuth_radians": float(azimuth_angle), "elevation_radians": float(elevation_angle),
                               "roll_radians": float(roll_angle), "distance_mm": float(object_extent_mm*random_generator.uniform(1.5, 5.5)),
                               "image_center_fraction": random_generator.uniform(.30, .70, 2).tolist(), "color_rgb": color_rgb,
                               "material_kind": ("diffuse", "roughplastic")[image_index%2], "roughness": float(random_generator.uniform(.03,.45)),
                               "ambient_rgb": random_generator.uniform(.05, .7, 3).tolist(),
                               "light_position_mm": (random_generator.normal(size=3)*object_extent_mm*3).tolist(),
                               "light_rgb": (random_generator.uniform(.3, 1, 3)*random_generator.uniform(1, 8)*object_extent_mm**2).tolist(),
                               "background_kind": ("white_noise", "gradient_noise", "generated_scene")[image_index%3],
                               "background_seed": int(random_generator.integers(0, 2**30)), "render_seed": int(random_generator.integers(0, 2**30))})
    return parameter_rows


def group_splits(parameter_rows: list[dict]) -> dict:
    membership = {split_name: [] for split_name in ("training", "development", "final_evaluation")}
    for group_field in ("pose_group", "background_group"):
        group_owners = {}
        for parameter_row in parameter_rows:
            group_identity, split_name = parameter_row[group_field], parameter_row["split"]
            if group_identity in group_owners and group_owners[group_identity] != split_name:
                raise ValueError(f"{group_field} leaks across dataset splits")
            group_owners[group_identity] = split_name
    for image_index, parameter_row in enumerate(parameter_rows):
        membership.setdefault(parameter_row["split"],[]).append(image_index)
    return membership


def resolve_camera_frame(object_vertices: np.ndarray, parameters: dict, camera_intrinsics: np.ndarray,
                         image_size: tuple[int,int], override: np.ndarray | None = None,
                         margin_fraction: float = .1) -> tuple[np.ndarray,dict]:
    """Fit random camera geometry; report but never relocate calibrated overrides."""
    object_center=(object_vertices.min(0)+object_vertices.max(0))*.5
    if override is not None:
        camera_matrix=np.asarray(override,dtype=np.float64).copy()
        camera_points=(object_vertices-camera_matrix[:3,3])@camera_matrix[:3,:3]
        positive=bool(np.all(camera_points[:,2]>.01) and np.all(camera_points[:,2]<1000.))
        homogeneous=camera_points@camera_intrinsics.T
        if np.any(camera_points[:,2]<=0):
            bounds=None;occupancy=None;inside=False
        else:
            projected=homogeneous[:,:2]/homogeneous[:,2,None]
            bounds=np.stack((projected.min(0),projected.max(0)))
            dimensions=np.asarray(image_size)
            occupancy=(bounds[1]-bounds[0])/dimensions
            inside=bool(positive and np.all(bounds[0]>=dimensions*margin_fraction) and np.all(bounds[1]<=dimensions*(1-margin_fraction)))
        return camera_matrix,dict(policy='fixed_camera_preserved_without_reframing',margin_fraction=margin_fraction,
            full_mesh_inside_margin=inside,projected_bounds_pixels=None if bounds is None else bounds.tolist(),
            bbox_fraction_xy=None if occupancy is None else occupancy.tolist(),depth_bounds_mm=[float(camera_points[:,2].min()),float(camera_points[:,2].max())])
    import mitsuba as mi
    mi.set_variant('scalar_rgb')
    azimuth,elevation=parameters['azimuth_radians'],parameters['elevation_radians']
    view_direction=np.array([np.cos(elevation)*np.cos(azimuth),np.cos(elevation)*np.sin(azimuth),np.sin(elevation)])
    initial=mi.ScalarTransform4f().look_at(origin=object_center+view_direction,target=object_center,up=[0,0,1])
    sensor=np.asarray((initial@mi.ScalarTransform4f().rotate([0,0,1],np.rad2deg(parameters['roll_radians']))).matrix,dtype=np.float64)
    camera_rotation=(sensor@np.diag([-1.,-1.,1.,1.]))[:3,:3]
    requested_distance=float(parameters['distance_mm'])
    fitted=fit_mesh_camera(object_vertices,camera_rotation,camera_intrinsics,image_size,requested_distance,
                           parameters['image_center_fraction'],margin_fraction,.01,1000.,object_center)
    return fitted.camera_to_object,dict(policy='full_mesh_distance_fit',margin_fraction=margin_fraction,
        requested_distance_mm=requested_distance,actual_distance_mm=fitted.distance_mm,required_distance_mm=fitted.required_distance_mm,
        projected_bounds_pixels=fitted.projected_bounds_pixels.tolist(),depth_bounds_mm=fitted.depth_bounds_mm.tolist(),
        bbox_fraction_xy=fitted.bbox_fraction_xy.tolist(),bbox_area_fraction=fitted.bbox_area_fraction,
        minimum_edge_margin_fraction=fitted.minimum_edge_margin_fraction.tolist(),small_occupancy=fitted.small_occupancy,full_mesh_inside_margin=True)


def render_preview(mesh_path: Path, feature_path: Path, output_path: Path, image_count: int = 36,
                   random_seed: int = 20261102, samples_per_pixel: int = 16,
                   photo_directory: Path | None = None, photo_manifest: Path | None = None,
                   parameters_override: list[dict] | None = None, create_review: bool = True,
                   camera_to_object_override: np.ndarray | None = None, framing_margin_fraction: float = .1) -> dict:
    import mitsuba as mi
    started_time = time.perf_counter()
    if output_path.exists():
        raise FileExistsError(f"Preserve existing output: {output_path}")
    feature_definition = FeatureDefinition.load_file(feature_path)
    if file_hash(mesh_path) != feature_definition.mesh_sha256:
        raise ValueError("Mesh differs from the approved feature definition")
    source_vertices, triangle_indices = read_binary_stl(mesh_path)
    object_vertices = (source_vertices-feature_definition.source_origin_mm)@feature_definition.source_rotation.T
    object_center = (object_vertices.min(axis=0)+object_vertices.max(axis=0))/2
    object_extent = float(np.ptp(object_vertices, axis=0).max())
    parameter_rows = preview_parameters(image_count, random_seed, object_extent) if parameters_override is None else parameters_override
    image_count=len(parameter_rows)
    photo_sources,photo_images=[],{}
    if (photo_directory is None) != (photo_manifest is None):
        raise ValueError("Photo directory and reviewed source manifest must be supplied together")
    if photo_directory is not None:
        photo_sources,photo_images=load_photos(photo_directory,photo_manifest)
        for image_index,parameters in enumerate(parameter_rows):
            if parameters_override is None:
                parameters["background_kind"]=("white_noise","gradient_noise","generated_scene","photograph")[image_index%4]
            if parameters["background_kind"]=="photograph":
                split_sources=[source for source in photo_sources if source["split"]==parameters["split"]]
                if not split_sources:
                    raise ValueError("Every split needs its own photographic source")
                source=split_sources[(image_index//4)%len(split_sources)]
                parameters["photo_source_id"]=source["source_id"]
                parameters["background_group"]=source["source_id"]
    split_indices = group_splits(parameter_rows)
    mi.set_variant("scalar_rgb")
    visibility_scene = SurfaceVisibility(object_vertices, triangle_indices)
    image_width, image_height = 640, 480
    focal_length = image_width*.5/np.tan(np.deg2rad(38/2))
    camera_intrinsics = np.broadcast_to([[focal_length,0,image_width/2],[0,focal_length,image_height/2],[0,0,1]], (image_count,3,3)).copy()
    camera_transforms = np.empty((image_count,4,4))
    if camera_to_object_override is not None:
        camera_to_object_override=np.asarray(camera_to_object_override,dtype=np.float64)
        if camera_to_object_override.shape!=(image_count,4,4) or not np.isfinite(camera_to_object_override).all():
            raise ValueError("Camera overrides must be finite image-by-4-by-4 transforms")
        np.testing.assert_allclose(camera_to_object_override[:,3],np.broadcast_to([0,0,0,1],(image_count,4)),atol=1e-10)
        rotations=camera_to_object_override[:,:3,:3]
        np.testing.assert_allclose(rotations.transpose(0,2,1)@rotations,np.broadcast_to(np.eye(3),rotations.shape),atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(rotations),1.,atol=1e-10)
    image_names = np.asarray([f"images/image_{image_index:06d}.png" for image_index in range(image_count)])
    output_path.mkdir(parents=True)
    (output_path/"images").mkdir()
    (output_path/"surface_arrays").mkdir()
    (output_path/"generation_parameters.json").write_text(json.dumps(parameter_rows,indent=2),encoding="utf-8")
    progress_path=output_path/"progress.json"
    progress_path.write_text(json.dumps({"pid":os.getpid(),"status":"running","completed":0,"total":image_count}),encoding="utf-8")
    image_hashes, surface_hashes = {}, {}
    render_seconds = []
    for image_index, parameters in enumerate(parameter_rows):
        override=None if camera_to_object_override is None else camera_to_object_override[image_index]
        camera_matrix,framing=resolve_camera_frame(object_vertices,parameters,camera_intrinsics[image_index],(image_width,image_height),override,framing_margin_fraction)
        parameters['framing']=framing
        if override is None:
            parameters['requested_distance_mm']=framing['requested_distance_mm']
            parameters['distance_mm']=framing['actual_distance_mm']
        sensor_matrix=camera_matrix@np.diag([-1.,-1.,1.,1.])
        camera_transforms[image_index] = camera_matrix
        material_definition = {"type": parameters["material_kind"]}
        if parameters["material_kind"] == "diffuse":
            material_definition["reflectance"] = {"type":"rgb","value":parameters["color_rgb"]}
        else:
            material_definition.update({"diffuse_reflectance":{"type":"rgb","value":parameters["color_rgb"]},"alpha":parameters["roughness"]})
        mesh_properties = mi.Properties()
        mesh_properties["bsdf"] = mi.load_dict(material_definition)
        render_mesh = mi.Mesh("object_mesh", len(object_vertices), len(triangle_indices), props=mesh_properties)
        mesh_parameters = mi.traverse(render_mesh)
        mesh_parameters["vertex_positions"] = object_vertices.astype(np.float32).ravel()
        mesh_parameters["faces"] = triangle_indices.astype(np.uint32).ravel()
        mesh_parameters.update()
        render_scene = mi.load_dict({"type":"scene","integrator":{"type":"path","max_depth":4,"hide_emitters":True},
                                    "object_mesh":render_mesh,"sensor":{"type":"perspective","to_world":mi.ScalarTransform4f(sensor_matrix),"fov":38.,"fov_axis":"x","near_clip":.01,"far_clip":1000.,
                                    "film":{"type":"hdrfilm","width":image_width,"height":image_height,"pixel_format":"rgba","rfilter":{"type":"box"}},"sampler":{"type":"independent","sample_count":samples_per_pixel}},
                                    "ambient_light":{"type":"constant","radiance":{"type":"rgb","value":parameters["ambient_rgb"]}},
                                    "point_light":{"type":"point","position":parameters["light_position_mm"],"intensity":{"type":"rgb","value":parameters["light_rgb"]}}})
        render_started = time.perf_counter()
        render_values = np.asarray(mi.render(render_scene, seed=parameters["render_seed"], spp=samples_per_pixel))
        render_seconds.append(time.perf_counter()-render_started)
        if render_values.shape != (image_height,image_width,4):
            raise ValueError("Expected rendered RGBA channels")
        surface_depth=pixel_depth(visibility_scene,camera_matrix,camera_intrinsics[image_index],(image_height,image_width))
        if parameters["background_kind"]=="photograph":
            background_rgb,crop_box=photo_background(cached_photo(photo_images[parameters["photo_source_id"]]),(image_height,image_width),parameters["background_seed"])
            parameters["photo_crop_box"]=crop_box
        else:
            background_rgb = procedural_background((image_height,image_width), parameters["background_kind"], parameters["background_seed"])
        linear_rgb = composite_foreground(render_values[...,:3], render_values[...,3], surface_depth, background_rgb)
        Image.fromarray(srgb_bytes(linear_rgb)).save(output_path/str(image_names[image_index]))
        surface_path = output_path/f"surface_arrays/image_{image_index:06d}.npz"
        np.savez_compressed(surface_path, coverage=render_values[...,3], depth_mm=surface_depth)
        image_hashes[str(image_names[image_index])] = file_hash(output_path/str(image_names[image_index]))
        surface_hashes[str(surface_path.relative_to(output_path)).replace("\\","/")] = file_hash(surface_path)
        if (image_index+1)%25==0 or image_index+1==image_count:
            progress={"pid":os.getpid(),"status":"running","completed":image_index+1,"total":image_count,"elapsed_seconds":time.perf_counter()-started_time}
            progress_path.write_text(json.dumps(progress),encoding="utf-8")
            print(json.dumps(progress),flush=True)
            np.savez_compressed(output_path/"camera_checkpoint.npz",camera_to_object=camera_transforms[:image_index+1])
    (output_path/"generation_parameters.json").write_text(json.dumps(parameter_rows,indent=2),encoding="utf-8")
    projected_pixels, projection_valid = project_features(feature_definition.feature_positions_mm,camera_transforms,camera_intrinsics,(image_width,image_height),.01,1000.)
    feature_visibility = visibility_scene.query_visibility(camera_transforms[:,:3,3],feature_definition.feature_positions_mm,.03)
    np.savez_compressed(output_path/"labels.npz",image_names=image_names,feature_names=feature_definition.feature_names,projected_xy_pixels=projected_pixels,
                        projection_valid=projection_valid,visibility=feature_visibility,camera_to_object=camera_transforms,camera_intrinsics=camera_intrinsics,
                        feature_positions_mm=feature_definition.feature_positions_mm,pose_groups=np.asarray([row["pose_group"] for row in parameter_rows]),
                        background_groups=np.asarray([row["background_group"] for row in parameter_rows]))
    manifest = {"schema_version":2,"purpose":"domain-randomized preview; not a trained model", "mesh_sha256":file_hash(mesh_path),"feature_definition_sha256":file_hash(feature_path),
                "feature_definition":json.loads(feature_path.read_text(encoding="utf-8-sig")),"image_size":[image_width,image_height],"image_count":image_count,"random_seed":random_seed,
                "samples_per_pixel":samples_per_pixel,"pixel_convention":"origin at top-left image edge; pixel centres at half-integers", "units":"mm",
                "camera_axes":"x right, y down, z forward","camera_to_object_convention":"right-handed column-vector R@camera_point+translation; object stays fixed",
                "endpoint_tolerance_mm":.03,"image_sha256":image_hashes,"surface_sha256":surface_hashes,"label_sha256":file_hash(output_path/"labels.npz"),
                "image_parameters":parameter_rows,"split_indices":split_indices,"split_policy":"diagnostic preview only; independent pose/background seed groups spanning below/side/above and roll; full experiment needs fresh reserved seeds",
                "framing_policy":"Full mesh fits a 10% margin by increasing random-camera distance; explicit camera overrides are preserved and reported" if framing_margin_fraction==.1 else "Full mesh random-camera distance fit; explicit camera overrides preserved",
                "background_provenance":photo_sources if photo_sources else "locally generated procedural graphics only",
                "surface_convention":"linear premultiplied foreground plus (1-coverage)*background; depth is pixel-centre camera-ray distance (0 means miss), coverage is subpixel-integrated; background infinitely far",
                "limitations":[f"Photographic source pool contains {len(photo_sources)} reviewed sources; real-camera transfer remains unverified" if photo_sources else "No photographs in this run","Generated scene backgrounds are graphics, not photorealistic rooms","No external occluders, shadows on composited backgrounds, camera distortion or motion blur","No empirical tracking-quality claim"],
                "dependencies":{package:version(package) for package in ("mitsuba","drjit","embreex","numpy","pillow")},"render_seconds":render_seconds,
                "generation_seconds_before_review":time.perf_counter()-started_time}
    (output_path/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    if create_review:
        contact_sheet(output_path,manifest,projected_pixels,projection_valid,feature_visibility)
    progress_path.write_text(json.dumps({"pid":os.getpid(),"status":"complete","completed":image_count,"total":image_count,"elapsed_seconds":time.perf_counter()-started_time}),encoding="utf-8")
    return manifest


def contact_sheet(output_path: Path, manifest: dict, projected_pixels: np.ndarray, projection_valid: np.ndarray, feature_visibility: np.ndarray) -> None:
    tile_width,tile_height = 320,264
    image_count=manifest["image_count"]
    review_image = Image.new("RGB",(tile_width*6,tile_height*((image_count+5)//6)),"white")
    for image_index,parameters in enumerate(manifest["image_parameters"]):
        with Image.open(output_path/f"images/image_{image_index:06d}.png") as image_file:
            tile_image=image_file.copy()
        tile_draw=ImageDraw.Draw(tile_image)
        for feature_index,pixel_position in enumerate(projected_pixels[image_index]):
            if not projection_valid[image_index,feature_index]:
                continue
            pixel_x,pixel_y=pixel_position
            point_color="red" if feature_visibility[image_index,feature_index] else "yellow"
            tile_draw.ellipse((pixel_x-3,pixel_y-3,pixel_x+3,pixel_y+3),fill=point_color,outline="black")
            tile_draw.text((pixel_x+4,pixel_y-9),str(feature_index+1),fill="black",stroke_width=1,stroke_fill="white")
        tile_image=tile_image.resize((320,240))
        tile_x,tile_y=image_index%6*tile_width,image_index//6*tile_height
        review_image.paste(tile_image,(tile_x,tile_y))
        ImageDraw.Draw(review_image).text((tile_x+2,tile_y+241),f"{image_index:02d} {parameters['split']} elev{np.rad2deg(parameters['elevation_radians']):.0f} roll{np.rad2deg(parameters['roll_radians']):.0f}",fill="black")
        ImageDraw.Draw(review_image).text((tile_x+2,tile_y+252),f"{parameters['material_kind']} / {parameters['background_kind']}",fill="black")
    review_image.save(output_path/"label-review.png")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh",type=Path,required=True)
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--image-count",type=int,default=36)
    parser.add_argument("--seed",type=int,default=20261102)
    parser.add_argument("--samples-per-pixel",type=int,default=16)
    parser.add_argument("--photo-directory",type=Path)
    parser.add_argument("--photo-manifest",type=Path)
    arguments=parser.parse_args()
    output_manifest=render_preview(arguments.mesh,arguments.features,arguments.output,arguments.image_count,arguments.seed,arguments.samples_per_pixel,arguments.photo_directory,arguments.photo_manifest)
    print(json.dumps({"output":str(arguments.output),"images":output_manifest["image_count"],"generation_seconds":output_manifest["generation_seconds_before_review"]}))

