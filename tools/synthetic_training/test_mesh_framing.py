import unittest
import numpy as np

from mesh_framing import fit_mesh_camera


class MeshFramingTests(unittest.TestCase):
    def setUp(self):
        self.vertices=np.array([[x,y,z] for x in [-30,30] for y in [-20,20] for z in [-15,15]],dtype=float)
        self.intrinsics=np.array([[900.,0,320],[0,900.,240],[0,0,1.]])

    def verify(self,result,vertices,rotation,intrinsics=None):
        intrinsics=self.intrinsics if intrinsics is None else intrinsics
        points=(vertices-result.camera_to_object[:3,3])@rotation
        projection=points@intrinsics.T;projection=projection[:,:2]/projection[:,2,None]
        self.assertTrue(np.all(points[:,2]>.01))
        self.assertTrue(np.all(projection>=[64-1e-5,48-1e-5]))
        self.assertTrue(np.all(projection<=[576+1e-5,432+1e-5]))
        target=(result.target_position_mm-result.camera_to_object[:3,3])@rotation
        target=intrinsics@target
        np.testing.assert_allclose(target[:2]/target[2],result.requested_center_pixels,atol=1e-8)

    def test_camera_near_or_inside_mesh_moves_back(self):
        result=fit_mesh_camera(self.vertices,np.eye(3),self.intrinsics,(640,480),1.)
        self.verify(result,self.vertices,np.eye(3))
        self.assertGreater(result.distance_mm,100)

    def test_off_center_extremes_and_arbitrary_orientation_roll(self):
        rng=np.random.default_rng(19)
        for _ in range(24):
            rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
            if np.linalg.det(rotation)<0:rotation[:,0]*=-1
            for center in ((.16,.16),(.84,.84),(.16,.84),(.5,.5)):
                result=fit_mesh_camera(self.vertices,rotation,self.intrinsics,(640,480),20.,center)
                self.verify(result,self.vertices,rotation)
                self.assertTrue(np.all(result.minimum_edge_margin_fraction>=.1-1e-8))

    def test_feature_subset_is_not_sufficient_to_frame_mesh(self):
        feature_subset=self.vertices*.1
        subset=fit_mesh_camera(feature_subset,np.eye(3),self.intrinsics,(640,480),1.)
        full=fit_mesh_camera(self.vertices,np.eye(3),self.intrinsics,(640,480),1.)
        self.assertGreater(full.distance_mm,subset.distance_mm*5)
        points=self.vertices-subset.camera_to_object[:3,3]
        pixels=points@self.intrinsics.T;pixels=pixels[:,:2]/pixels[:,2,None]
        self.assertTrue(np.any(pixels<0) or np.any(pixels>[640,480]))
        self.verify(full,self.vertices,np.eye(3))

    def test_existing_distance_scale_and_occupancy_are_preserved(self):
        result=fit_mesh_camera(self.vertices,np.eye(3),self.intrinsics,(640,480),1500.)
        self.assertEqual(result.distance_mm,1500.)
        np.testing.assert_allclose(result.bbox_fraction_xy,np.diff(result.projected_bounds_pixels,axis=0)[0]/[640,480])
        self.assertAlmostEqual(result.bbox_area_fraction,np.prod(result.bbox_fraction_xy))
        self.assertTrue(result.small_occupancy)

    def test_far_plane_failure_is_not_silently_clipped(self):
        with self.assertRaisesRegex(ValueError,'clipping planes'):
            fit_mesh_camera(self.vertices,np.eye(3),self.intrinsics,(640,480),1.,far_clip_mm=100.)
        with self.assertRaises(ValueError):
            fit_mesh_camera(self.vertices,np.eye(3),self.intrinsics,(640,480),1.,(.1,.5))

    def test_translated_geometry_and_skew_intrinsics(self):
        vertices=self.vertices+[1000,-2000,3000]
        intrinsics=self.intrinsics.copy();intrinsics[0,1]=35.
        result=fit_mesh_camera(vertices,np.eye(3),intrinsics,(640,480),1.,(.8,.3))
        self.verify(result,vertices,np.eye(3),intrinsics)

    def test_float32_renderer_rotation_retains_proper_frame(self):
        rotation,_=np.linalg.qr(np.random.default_rng(93).normal(size=(3,3)))
        rotation=rotation.astype(np.float32)
        result=fit_mesh_camera(self.vertices,rotation,self.intrinsics,(640,480),20.,(.16,.84))
        self.verify(result,self.vertices,result.camera_to_object[:3,:3])
        np.testing.assert_allclose(result.camera_to_object[:3,:3],rotation,atol=1e-7)


if __name__=='__main__':
    unittest.main()
