"""Run with the isolated renderer Python: -B -m unittest discover -s tools/synthetic_training."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from render_dataset import FeatureDefinition, SurfaceVisibility, project_features, read_binary_stl
from review_dataset import split_membership


class ProjectionTests(unittest.TestCase):
    def test_axes_depth_bounds_and_feature_order(self):
        feature_positions = np.array([[0, 0, 10], [1, 2, 10], [-1, -2, 10], [0, 0, -10], [9, 0, 10]], dtype=float)
        intrinsic_matrix = np.array([[[100, 0, 50], [0, 100, 40], [0, 0, 1]]], dtype=float)
        projected_pixels, projection_valid = project_features(feature_positions, np.eye(4)[None], intrinsic_matrix, (100, 80), .01, 100)
        np.testing.assert_allclose(projected_pixels[0, :3], [[50, 40], [60, 60], [40, 20]])
        np.testing.assert_array_equal(projection_valid, [[True, True, True, False, False]])

    def test_object_transform_and_units_preserve_projection(self):
        feature_positions = np.array([[1, 2, 10], [-2, 1, 15]], dtype=float)
        intrinsic_matrix = np.array([[[100, 0, 50], [0, 100, 40], [0, 0, 1]]], dtype=float)
        base_pixels, _ = project_features(feature_positions, np.eye(4)[None], intrinsic_matrix, (100, 80), .01, 100)
        proper_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        frame_translation = np.array([5, 3, 7])
        camera_transform = np.eye(4)
        camera_transform[:3, :3] = proper_rotation
        camera_transform[:3, 3] = frame_translation/60
        transformed_pixels, _ = project_features((feature_positions@proper_rotation.T+frame_translation)/60, camera_transform[None], intrinsic_matrix, (100, 80), .01/60, 100/60)
        np.testing.assert_allclose(transformed_pixels, base_pixels)

    def test_sensor_rays_and_raster_feature_centroid(self):
        import mitsuba as mi
        mi.set_variant("scalar_rgb")
        image_width, image_height = 320, 240
        focal_length = image_width*.5/np.tan(np.deg2rad(19))
        sensor_transform = mi.ScalarTransform4f().look_at(origin=[0, 0, -10], target=[0, 0, 0], up=[0, 1, 0])
        camera_transform = np.asarray(sensor_transform.matrix)@np.diag([-1., -1., 1., 1.])
        feature_positions = np.array([[1., .5, 0.]])
        intrinsic_matrix = np.array([[[focal_length, 0, image_width/2], [0, focal_length, image_height/2], [0, 0, 1]]])
        projected_pixels, projection_valid = project_features(feature_positions, camera_transform[None], intrinsic_matrix, (image_width, image_height), .01, 100)
        self.assertTrue(projection_valid.all())
        scene_definition = {"type": "scene", "integrator": {"type": "path"},
                            "sensor": {"type": "perspective", "to_world": sensor_transform, "fov": 38, "fov_axis": "x",
                                       "film": {"type": "hdrfilm", "width": image_width, "height": image_height, "rfilter": {"type": "box"}},
                                       "sampler": {"type": "independent", "sample_count": 64}},
                            "feature_marker": {"type": "sphere", "center": feature_positions[0], "radius": .08,
                                               "emitter": {"type": "area", "radiance": 1.}}}
        render_scene = mi.load_dict(scene_definition)
        sensor_ray, _ = render_scene.sensors()[0].sample_ray(0, .5, mi.Point2f(projected_pixels[0, 0]/[image_width, image_height]), mi.Point2f([.5, .5]))
        expected_direction = feature_positions[0]-camera_transform[:3, 3]
        expected_direction /= np.linalg.norm(expected_direction)
        np.testing.assert_allclose(np.asarray(sensor_ray.d), expected_direction, atol=1e-6)
        pixel_weights = np.asarray(mi.render(render_scene, seed=7, spp=64)).mean(axis=2)
        pixel_rows, pixel_columns = np.indices(pixel_weights.shape)
        centroid_pixels = np.array([np.sum(pixel_weights*(pixel_columns+.5)), np.sum(pixel_weights*(pixel_rows+.5))])/pixel_weights.sum()
        np.testing.assert_allclose(centroid_pixels, projected_pixels[0, 0], atol=.2)


class VisibilityTests(unittest.TestCase):
    def test_surface_endpoint_and_occluder_distance(self):
        vertex_positions = np.array([[-2, -2, 0], [2, -2, 0], [0, 2, 0]], dtype=float)
        visibility_scene = SurfaceVisibility(vertex_positions, np.array([[0, 1, 2]]))
        feature_positions = np.array([[0, 0, 2], [0, 0, 0], [0, 0, -.02], [0, 0, -.04], [0, 0, -2]])
        camera_positions = np.array([[0, 0, 10]])
        expected_visibility = [[True, True, True, False, False]]
        np.testing.assert_array_equal(visibility_scene.query_visibility(camera_positions, feature_positions, .03), expected_visibility)
        normalized_scene = SurfaceVisibility(vertex_positions/60, np.array([[0, 1, 2]]))
        np.testing.assert_array_equal(normalized_scene.query_visibility(camera_positions/60, feature_positions/60, .03/60), expected_visibility)

    def test_invalid_tolerance_and_zero_ray_rejected(self):
        visibility_scene = SurfaceVisibility(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]))
        with self.assertRaises(ValueError):
            visibility_scene.query_visibility(np.zeros((1, 3)), np.ones((1, 3)), 0)
        with self.assertRaises(ValueError):
            visibility_scene.query_visibility(np.zeros((1, 3)), np.zeros((1, 3)), .03)


class DefinitionTests(unittest.TestCase):
    def test_split_groups_pose_duplicates_and_excludes_guard_band(self):
        elevation_values = np.deg2rad([10, 10, 45, 50, 55, 55, 65])
        membership = split_membership(elevation_values)
        np.testing.assert_array_equal(membership["training"], [0, 1, 2])
        np.testing.assert_array_equal(membership["evaluation"], [4, 5, 6])
        np.testing.assert_array_equal(membership["excluded_guard_band"], [3])
        self.assertEqual(len(np.unique(np.concatenate(list(membership.values())))), len(elevation_values))

    def test_approved_feature_order_and_frame(self):
        feature_definition = FeatureDefinition.load_file(Path(__file__).with_name("benchy_features.json"))
        self.assertEqual(len(feature_definition.feature_names), 15)
        self.assertEqual(feature_definition.feature_names[0], "roof_fore_port")
        self.assertEqual(feature_definition.feature_names[-1], "chimney_rim_starboard")
        self.assertEqual(feature_definition.reference_unit_mm, 60)

    def test_duplicate_identity_rejected(self):
        feature_data = json.loads(Path(__file__).with_name("benchy_features.json").read_text(encoding="utf-8-sig"))
        feature_data["feature_names"][1] = feature_data["feature_names"][0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            feature_path = Path(temporary_directory)/"features.json"
            feature_path.write_text(json.dumps(feature_data), encoding="utf-8")
            with self.assertRaises(ValueError):
                FeatureDefinition.load_file(feature_path)

    def test_truncated_stl_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            mesh_path = Path(temporary_directory)/"mesh.stl"
            mesh_path.write_bytes(b"invalid")
            with self.assertRaises(ValueError):
                read_binary_stl(mesh_path)


if __name__ == "__main__":
    unittest.main()
