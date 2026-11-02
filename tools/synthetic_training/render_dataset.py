"""Optional CPU synthetic image/geometry export; no Tracker or Forge runtime imports.

Mitsuba sensor/mesh APIs: [@mitsuba391Mesh] [@mitsubaPerspectiveSensor].
Compiled batch visibility: [@embreexArrayQueries].
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_binary_stl(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw_bytes = mesh_path.read_bytes()
    if len(raw_bytes) < 84:
        raise ValueError("Binary STL header is incomplete")
    triangle_count = int.from_bytes(raw_bytes[80:84], "little")
    if triangle_count == 0 or len(raw_bytes) != 84 + 50 * triangle_count:
        raise ValueError("Expected exact binary STL triangle payload")
    triangle_dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    triangle_data = np.frombuffer(raw_bytes, dtype=triangle_dtype, offset=84)
    vertex_positions, vertex_indices = np.unique(triangle_data["vertices"].reshape(-1, 3), axis=0, return_inverse=True)
    if not np.isfinite(vertex_positions).all():
        raise ValueError("STL contains nonfinite coordinates")
    return vertex_positions.astype(np.float64), vertex_indices.reshape(-1, 3).astype(np.int32)


@dataclass(frozen=True)
class FeatureDefinition:
    feature_names: np.ndarray
    feature_positions_mm: np.ndarray
    source_rotation: np.ndarray
    source_origin_mm: np.ndarray
    reference_unit_mm: float
    mesh_sha256: str

    @classmethod
    def load_file(cls, feature_path: Path) -> FeatureDefinition:
        feature_data = json.loads(feature_path.read_text(encoding="utf-8-sig"))
        if feature_data["status"] not in {"approved_initial_baseline", "generated_object_definition"} or feature_data["units"] != "mm":
            raise ValueError("A supported millimetre feature definition is required")
        feature_names = np.asarray(feature_data["feature_names"], dtype=str)
        feature_positions = np.asarray(feature_data["object_positions_mm"], dtype=np.float64)
        source_rotation = np.asarray(feature_data["source_to_object_rotation"], dtype=np.float64)
        source_origin = np.asarray(feature_data["origin_source_mm"], dtype=np.float64)
        if feature_names.ndim != 1 or len(feature_names) == 0 or len(np.unique(feature_names)) != len(feature_names):
            raise ValueError("Feature identities must be nonempty and unique")
        if feature_positions.shape != (len(feature_names), 3) or not np.isfinite(feature_positions).all():
            raise ValueError("Feature positions must be finite K by 3 coordinates")
        if source_rotation.shape != (3, 3) or source_origin.shape != (3,):
            raise ValueError("Source frame dimensions are incorrect")
        if not np.allclose(source_rotation.T @ source_rotation, np.eye(3)) or not np.isclose(np.linalg.det(source_rotation), 1):
            raise ValueError("Source frame must use a proper rotation")
        reference_unit = float(feature_data["reference_unit_mm"])
        if not np.isfinite(reference_unit) or reference_unit <= 0:
            raise ValueError("Reference unit must be positive millimetres")
        source_positions = np.asarray(feature_data["source_positions_mm"], dtype=np.float64)
        if source_positions.shape != feature_positions.shape or not np.allclose((source_positions-source_origin) @ source_rotation.T, feature_positions, atol=1e-8):
            raise ValueError("Source and object feature positions disagree")
        return cls(feature_names, feature_positions, source_rotation, source_origin, reference_unit, feature_data["source_mesh_sha256"])


def project_features(feature_positions_mm: np.ndarray, camera_to_object: np.ndarray, camera_intrinsics: np.ndarray,
                     image_size: tuple[int, int], near_clip_mm: float, far_clip_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV camera axes: x right, y down, z forward; coordinates at pixel edges."""
    camera_positions = np.einsum("nki,nij->nkj", feature_positions_mm[None] - camera_to_object[:, None, :3, 3], camera_to_object[:, :3, :3])
    homogeneous_pixels = np.einsum("nij,nkj->nki", camera_intrinsics, camera_positions)
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_pixels = homogeneous_pixels[..., :2] / homogeneous_pixels[..., 2, None]
    depth_values = camera_positions[..., 2]
    image_width, image_height = image_size
    projection_valid = (np.isfinite(projected_pixels).all(axis=-1) & (depth_values > near_clip_mm) & (depth_values < far_clip_mm)
                        & (projected_pixels[..., 0] >= 0) & (projected_pixels[..., 0] < image_width)
                        & (projected_pixels[..., 1] >= 0) & (projected_pixels[..., 1] < image_height))
    return projected_pixels, projection_valid


class SurfaceVisibility:
    """Reuse one native BVH for dense camera-by-feature queries in object mm."""
    def __init__(self, vertex_positions_mm: np.ndarray, triangle_indices: np.ndarray):
        from embreex.mesh_construction import TriangleMesh
        from embreex.rtcore_scene import EmbreeScene
        self.embree_scene = EmbreeScene()
        self.mesh_geometry = TriangleMesh(self.embree_scene, np.ascontiguousarray(vertex_positions_mm, dtype=np.float32),
                                          np.ascontiguousarray(triangle_indices, dtype=np.int32))

    def query_visibility(self, camera_positions_mm: np.ndarray, feature_positions_mm: np.ndarray, endpoint_tolerance_mm: float) -> np.ndarray:
        if not np.isfinite(endpoint_tolerance_mm) or endpoint_tolerance_mm <= 0:
            raise ValueError("Endpoint tolerance must be positive millimetres")
        ray_deltas = feature_positions_mm[None] - camera_positions_mm[:, None]
        ray_distances = np.linalg.norm(ray_deltas, axis=-1)
        if not np.isfinite(ray_distances).all() or np.any(ray_distances <= endpoint_tolerance_mm):
            raise ValueError("Camera and feature positions must form finite nonzero rays")
        ray_origins = np.ascontiguousarray(np.broadcast_to(camera_positions_mm[:, None], ray_deltas.shape).reshape(-1, 3), dtype=np.float32)
        ray_directions = np.ascontiguousarray((ray_deltas / ray_distances[..., None]).reshape(-1, 3), dtype=np.float32)
        ray_endpoints = np.ascontiguousarray((ray_distances-endpoint_tolerance_mm).ravel(), dtype=np.float32)
        # OCCLUDED returns -1 for a clear segment, a triangle index otherwise.
        hit_indices = self.embree_scene.run(ray_origins, ray_directions, dists=ray_endpoints, query="OCCLUDED")
        return (hit_indices < 0).reshape(ray_distances.shape)


def generate_dataset(mesh_path: Path, feature_path: Path, output_path: Path, image_count: int = 64,
                     random_seed: int = 20261101, image_width: int = 640, image_height: int = 480,
                     samples_per_pixel: int = 16, endpoint_tolerance_mm: float = 0.03) -> dict:
    generation_started = time.perf_counter()
    import mitsuba as mi
    from PIL import Image
    if min(image_count, image_width, image_height, samples_per_pixel) <= 0:
        raise ValueError("Image counts, dimensions and samples must be positive")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output_path}")
    feature_definition = FeatureDefinition.load_file(feature_path)
    if file_hash(mesh_path) != feature_definition.mesh_sha256:
        raise ValueError("Mesh hash does not match the approved feature definition")
    vertex_positions, triangle_indices = read_binary_stl(mesh_path)
    vertex_positions = (vertex_positions-feature_definition.source_origin_mm) @ feature_definition.source_rotation.T
    visibility_scene = SurfaceVisibility(vertex_positions, triangle_indices)
    mi.set_variant("scalar_rgb")
    mesh_properties = mi.Properties()
    mesh_properties["bsdf"] = mi.load_dict({"type": "diffuse", "reflectance": {"type": "rgb", "value": [.24, .48, .68]}})
    render_mesh = mi.Mesh("approved_object", len(vertex_positions), len(triangle_indices), props=mesh_properties)
    mesh_parameters = mi.traverse(render_mesh)
    mesh_parameters["vertex_positions"] = vertex_positions.astype(np.float32).ravel()
    mesh_parameters["faces"] = triangle_indices.astype(np.uint32).ravel()
    mesh_parameters.update()
    random_generator = np.random.default_rng(random_seed)
    azimuth_angles = random_generator.uniform(-np.pi, np.pi, image_count)
    elevation_angles = random_generator.uniform(np.deg2rad(8), np.deg2rad(70), image_count)
    camera_distances = random_generator.uniform(125, 190, image_count)
    view_targets = np.broadcast_to([0., 0., 23.], (image_count, 3)).copy()
    view_targets[:, :2] += random_generator.uniform(-5, 5, (image_count, 2))
    camera_positions = view_targets + camera_distances[:, None] * np.column_stack((np.cos(elevation_angles)*np.cos(azimuth_angles), np.cos(elevation_angles)*np.sin(azimuth_angles), np.sin(elevation_angles)))
    ambient_values = random_generator.uniform(.35, 1.1, image_count)
    light_intensities = random_generator.uniform(6000, 22000, image_count)
    image_seeds = random_generator.integers(0, 2**31, image_count, dtype=np.int64)
    image_names = np.asarray([f"images/image_{image_index:06d}.png" for image_index in range(image_count)])
    camera_transforms = np.empty((image_count, 4, 4), dtype=np.float64)
    intrinsic_matrices = np.broadcast_to(np.eye(3), (image_count, 3, 3)).copy()
    focal_length = image_width*.5/np.tan(np.deg2rad(38/2))
    intrinsic_matrices[:, 0, 0] = intrinsic_matrices[:, 1, 1] = focal_length
    intrinsic_matrices[:, 0, 2] = image_width*.5
    intrinsic_matrices[:, 1, 2] = image_height*.5
    render_timings = np.empty(image_count)
    output_path.mkdir(parents=True)
    (output_path/"images").mkdir()
    # Images are independent render jobs; feature assembly and visibility remain dense.
    for image_index in range(image_count):
        sensor_transform = mi.ScalarTransform4f().look_at(origin=camera_positions[image_index], target=view_targets[image_index], up=[0, 0, 1])
        camera_transforms[image_index] = np.asarray(sensor_transform.matrix) @ np.diag([-1., -1., 1., 1.])
        scene_definition = {"type": "scene", "integrator": {"type": "path", "max_depth": 3}, "object_mesh": render_mesh,
                            "sensor": {"type": "perspective", "to_world": sensor_transform, "fov": 38., "fov_axis": "x", "near_clip": .01, "far_clip": 1000.,
                                       "film": {"type": "hdrfilm", "width": image_width, "height": image_height, "rfilter": {"type": "box"}},
                                       "sampler": {"type": "independent", "sample_count": samples_per_pixel}},
                            "ambient_light": {"type": "constant", "radiance": float(ambient_values[image_index])},
                            "point_light": {"type": "point", "position": [60, -90, 140], "intensity": float(light_intensities[image_index])}}
        render_scene = mi.load_dict(scene_definition)
        start_time = time.perf_counter()
        rendered_image = mi.render(render_scene, seed=int(image_seeds[image_index]), spp=samples_per_pixel)
        render_timings[image_index] = time.perf_counter()-start_time
        image_bitmap = mi.Bitmap(rendered_image).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)
        Image.fromarray(np.asarray(image_bitmap)).save(output_path/str(image_names[image_index]))
    projected_pixels, projection_valid = project_features(feature_definition.feature_positions_mm, camera_transforms, intrinsic_matrices, (image_width, image_height), .01, 1000.)
    feature_visibility = visibility_scene.query_visibility(camera_positions, feature_definition.feature_positions_mm, endpoint_tolerance_mm)
    np.savez_compressed(output_path/"labels.npz", image_names=image_names, feature_names=feature_definition.feature_names,
                        projected_xy_pixels=projected_pixels, visibility=feature_visibility, projection_valid=projection_valid,
                        camera_to_object=camera_transforms, camera_intrinsics=intrinsic_matrices,
                        feature_positions_mm=feature_definition.feature_positions_mm, image_seeds=image_seeds,
                        ambient_values=ambient_values, light_intensities=light_intensities,
                        view_targets_mm=view_targets, azimuth_radians=azimuth_angles, elevation_radians=elevation_angles,
                        camera_distances_mm=camera_distances)
    manifest = {"schema_version": 1, "status": "generated_geometry_not_trained", "mesh_sha256": file_hash(mesh_path),
                "feature_definition_sha256": file_hash(feature_path), "feature_definition": json.loads(feature_path.read_text(encoding="utf-8-sig")),
                "random_seed": random_seed, "image_count": image_count, "image_size": [image_width, image_height], "samples_per_pixel": samples_per_pixel,
                "endpoint_tolerance_mm": endpoint_tolerance_mm, "camera_axes": "x right, y down, z forward", "pixel_convention": "origin at top-left image edge; pixel centres at half-integers",
                "units": "mm", "object_pose": "identity; cameras vary in the fixed object frame",
                "camera_to_object_convention": "Proper right-handed camera-to-object transform; column vectors; R @ camera_point + translation",
                "near_clip_mm": .01, "far_clip_mm": 1000., "renderer_variant": "scalar_rgb",
                "dependencies": {package_name: version(package_name) for package_name in ["mitsuba", "drjit", "embreex", "numpy", "pillow"]},
                "render_seconds": render_timings.tolist(), "label_sha256": file_hash(output_path/"labels.npz"),
                "image_sha256": {str(image_name): file_hash(output_path/str(image_name)) for image_name in image_names},
                "limitations": ["Fixed material and simple backgrounds; no real-camera accuracy claim", "No external occluders; self-occlusion only", "Visibility does not determine DLC supervision masking", "Bitwise reproducibility checked only within the same pinned environment"]}
    manifest["generation_seconds_before_manifest_write"] = time.perf_counter()-generation_started
    (output_path/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--mesh", type=Path, required=True)
    argument_parser.add_argument("--features", type=Path, required=True)
    argument_parser.add_argument("--output", type=Path, required=True)
    argument_parser.add_argument("--image-count", type=int, default=64)
    argument_parser.add_argument("--seed", type=int, default=20261101)
    argument_parser.add_argument("--samples-per-pixel", type=int, default=16)
    parsed_arguments = argument_parser.parse_args()
    result_manifest = generate_dataset(parsed_arguments.mesh, parsed_arguments.features, parsed_arguments.output,
                                       image_count=parsed_arguments.image_count, random_seed=parsed_arguments.seed,
                                       samples_per_pixel=parsed_arguments.samples_per_pixel)
    print(json.dumps({"output": str(parsed_arguments.output), "image_count": result_manifest["image_count"], "render_seconds": sum(result_manifest["render_seconds"]),
                      "generation_seconds_before_manifest_write": result_manifest["generation_seconds_before_manifest_write"]}))
