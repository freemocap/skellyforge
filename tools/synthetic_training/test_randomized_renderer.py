import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from randomized_renderer import composite_foreground, gradient_noise, group_splits, preview_parameters, procedural_background, pixel_depth, photo_background, srgb_bytes, load_photos
from render_dataset import SurfaceVisibility, file_hash


class RandomizedRendererTests(unittest.TestCase):
    def test_related_photo_families_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            sources=[]
            for index,split_name in enumerate(("training","final_evaluation")):
                image_path=root/f"photo_{index}.png"
                Image.new("RGB",(8,8),(index*100,0,0)).save(image_path)
                sources.append({"file_name":image_path.name,"source_id":str(index),"source_family":"same_room",
                                "split":split_name,"license":"CC0-1.0","source_url":"https://example.invalid/fixture",
                                "attribution":"Synthetic unit fixture","sha256":file_hash(image_path),"dimensions":[8,8]})
            manifest_path=root/"manifest.json"
            manifest_path.write_text(json.dumps(sources))
            with self.assertRaisesRegex(ValueError,"families"):
                load_photos(root,manifest_path)

    def test_photographic_crop_and_color_roundtrip(self):
        source_image=Image.new("RGB",(80,100),(30,100,210))
        first,crop_box=photo_background(source_image,(48,64),7)
        second,repeated_box=photo_background(source_image,(48,64),7)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(crop_box,repeated_box)
        self.assertGreaterEqual(crop_box[0],0)
        self.assertLessEqual(crop_box[2],80)
        self.assertLessEqual(crop_box[3],100)
        np.testing.assert_array_equal(srgb_bytes(first)[0,0],[30,100,210])

    def test_photo_source_cannot_cross_splits(self):
        parameters=preview_parameters(36,42,60)
        parameters[0]["background_group"]="photo:one-source"
        parameters[12]["background_group"]="photo:one-source"
        with self.assertRaises(ValueError):
            group_splits(parameters)

    def test_backgrounds_are_reproducible_and_distinct(self):
        background_arrays=[]
        for background_kind in ("white_noise","gradient_noise","generated_scene"):
            first=procedural_background((48,64),background_kind,12)
            second=procedural_background((48,64),background_kind,12)
            np.testing.assert_array_equal(first,second)
            self.assertEqual(first.shape,(48,64,3))
            self.assertTrue(np.isfinite(first).all())
            self.assertGreater(first.std(),.01)
            background_arrays.append(first)
        self.assertFalse(np.array_equal(background_arrays[0],background_arrays[1]))

    def test_gradient_noise_has_spatial_coherence(self):
        texture_values=gradient_noise((100,100),4,7)
        adjacent_difference=np.abs(np.diff(texture_values,axis=1)).mean()
        distant_difference=np.abs(texture_values[:,:50]-texture_values[:,50:]).mean()
        self.assertLess(adjacent_difference,distant_difference/3)

    def test_compositing_preserves_foreground_holes_and_edges(self):
        coverage_values=np.array([[0.,.5,1.]])
        depth_values=np.array([[0.,10.,10.]])
        foreground_values=np.broadcast_to(coverage_values[...,None]*np.array([1.,0.,0.]),(1,3,3)).copy()
        background_values=np.broadcast_to([0.,0.,1.],(1,3,3))
        result=composite_foreground(foreground_values,coverage_values,depth_values,background_values)
        np.testing.assert_allclose(result,[[[0,0,1],[.5,0,.5],[1,0,0]]])
        with self.assertRaises(ValueError):
            composite_foreground(foreground_values,coverage_values,-np.ones((1,3)),background_values)

    def test_dense_pixel_depth_matches_known_plane(self):
        plane_scene=SurfaceVisibility(np.array([[-100,-100,10],[100,-100,10],[0,100,10]]),np.array([[0,1,2]]))
        intrinsic_matrix=np.array([[100,0,1.5],[0,100,1.5],[0,0,1]])
        depth_values=pixel_depth(plane_scene,np.eye(4),intrinsic_matrix,(3,3))
        self.assertAlmostEqual(float(depth_values[1,1]),10,places=5)
        self.assertGreater(depth_values[0,0],depth_values[1,1])

    def test_grouped_split_and_broad_domain(self):
        parameter_rows=preview_parameters(36,42,60)
        self.assertEqual(parameter_rows,preview_parameters(36,42,60))
        membership=group_splits(parameter_rows)
        self.assertEqual([len(indices) for indices in membership.values()],[12,12,12])
        for image_indices in membership.values():
            elevations=np.array([parameter_rows[index]["elevation_radians"] for index in image_indices])
            self.assertLess(elevations.min(),-np.arcsin(1/3))
            self.assertGreater(elevations.max(),np.arcsin(1/3))
            self.assertEqual(len({parameter_rows[index]["background_kind"] for index in image_indices}),3)
            self.assertGreater(np.ptp([parameter_rows[index]["roll_radians"] for index in image_indices]),3)
        parameter_rows[12]["pose_group"]=parameter_rows[0]["pose_group"]
        with self.assertRaises(ValueError):
            group_splits(parameter_rows)

    def test_renderer_depth_coverage_and_premultiplication(self):
        import mitsuba as mi
        mi.set_variant("scalar_rgb")
        scene=mi.load_dict({"type":"scene","integrator":{"type":"aov","aovs":"surface_depth:depth","beauty":{"type":"path","hide_emitters":True}},
                           "sensor":{"type":"perspective","to_world":mi.ScalarTransform4f().look_at(origin=[0,0,-5],target=[0,0,0],up=[0,1,0]),
                                     "film":{"type":"hdrfilm","width":48,"height":48,"pixel_format":"rgba","rfilter":{"type":"box"}}},
                           "object_shape":{"type":"sphere"},"environment_light":{"type":"constant"}})
        render_values=np.asarray(mi.render(scene,spp=16,seed=11))
        self.assertEqual(render_values.shape,(48,48,5))
        self.assertEqual(render_values[0,0,3],0)
        self.assertGreater(render_values[24,24,4],0)
        self.assertTrue(np.any((render_values[...,3]>0)&(render_values[...,3]<1)))
        result=composite_foreground(render_values[...,:3],render_values[...,3],render_values[...,4],np.ones((48,48,3))*.2)
        np.testing.assert_allclose(result[0,0],[.2,.2,.2])


if __name__ == "__main__":
    unittest.main()
