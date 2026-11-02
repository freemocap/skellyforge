import unittest
import numpy as np

from randomized_renderer import preview_parameters,resolve_camera_frame
from production_dataset import production_parameters


class FramingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.vertices=np.array([[x,y,z] for x in [-30,30] for y in [-20,20] for z in [-15,15]],dtype=float)
        focal=320/np.tan(np.deg2rad(19))
        self.intrinsics=np.array([[focal,0,320],[0,focal,240],[0,0,1.]])

    def test_random_camera_increases_distance_and_preserves_roll_and_target(self):
        parameters=dict(azimuth_radians=.3,elevation_radians=.2,roll_radians=1.4,distance_mm=20.,image_center_fraction=[.3,.7])
        camera,record=resolve_camera_frame(self.vertices,parameters,self.intrinsics,(640,480))
        self.assertEqual(record['requested_distance_mm'],20)
        self.assertGreater(record['actual_distance_mm'],20)
        self.assertTrue(record['full_mesh_inside_margin'])
        points=(self.vertices-camera[:3,3])@camera[:3,:3]
        pixels=points@self.intrinsics.T;pixels=pixels[:,:2]/pixels[:,2,None]
        self.assertTrue(np.all(pixels>=[64-1e-5,48-1e-5]))
        self.assertTrue(np.all(pixels<=[576+1e-5,432+1e-5]))

    def test_fixed_camera_is_unchanged_even_when_object_is_clipped(self):
        override=np.eye(4);override[2,3]=-30
        camera,record=resolve_camera_frame(self.vertices,{},self.intrinsics,(640,480),override)
        np.testing.assert_array_equal(camera,override)
        self.assertEqual(record['policy'],'fixed_camera_preserved_without_reframing')
        self.assertFalse(record['full_mesh_inside_margin'])

    def test_sampled_centres_retain_variation_without_extreme_edges(self):
        for rows in (preview_parameters(36,21,60),production_parameters('training',22,60)):
            centers=np.array([row['image_center_fraction'] for row in rows])
            self.assertTrue(np.all(centers>=.3));self.assertTrue(np.all(centers<=.7))
            self.assertGreater(np.ptp(centers),.3)


if __name__=='__main__':
    unittest.main()
