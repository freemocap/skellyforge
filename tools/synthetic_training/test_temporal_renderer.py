import shutil
import tempfile
import unittest
from unittest.mock import patch
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from temporal_renderer import encode_lossless, temporal_geometry


class TemporalGeometryTests(unittest.TestCase):
    def test_world_and_object_projection_agree_for_fixed_cameras(self):
        points=np.array([[-10,-20,0],[10,20,40],[0,0,20]],dtype=float)
        values=temporal_geometry(points)
        poses=values['object_to_world']; cameras=values['camera_to_world']
        np.testing.assert_allclose(poses[:,None]@values['camera_to_object'],np.broadcast_to(cameras,(240,4,4,4)),atol=1e-10)
        world_points=np.einsum('fij,kj->fki',poses[:,:3,:3],points)+poses[:,None,:3,3]
        camera_points=np.einsum('fckj,cji->fcki',world_points[:,None]-cameras[None,:,None,:3,3],cameras[:,:3,:3])
        pixel_homogeneous=np.einsum('cij,fckj->fcki',values['camera_intrinsics'],camera_points)
        np.testing.assert_allclose(pixel_homogeneous[...,:2]/pixel_homogeneous[...,2,None],values['projected_xy_pixels'],atol=1e-10)
        self.assertTrue(values['projection_valid'].all())
        self.assertTrue(np.all(np.ptp(poses[:,:3,3],axis=0)>1))
        self.assertTrue(np.all(np.ptp(values['rotation_angles_radians'],axis=0)>.1))
        np.testing.assert_allclose(np.linalg.det(cameras[:,:3,:3]),1)
        np.testing.assert_array_equal(values['timestamps_seconds'],np.arange(240)/30)

    def test_fixed_world_light_is_transformed_with_object_pose(self):
        values=temporal_geometry(np.array([[-10,-20,0],[10,20,40]],dtype=float))
        poses=values['object_to_world']
        recovered=np.einsum('fij,fj->fi',poses[:,:3,:3],values['light_positions_object_mm'])+poses[:,:3,3]
        np.testing.assert_allclose(recovered,np.broadcast_to(values['light_position_world_mm'],(240,3)),atol=1e-10)

    @unittest.skipUnless(shutil.which('ffmpeg'),'Existing ffmpeg required for lossless codec test')
    def test_lossless_video_roundtrip_preserves_rgb_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=[]
            rng=np.random.default_rng(12)
            for frame_index in range(3):
                path=root/f'frame_{frame_index}.png';Image.fromarray(rng.integers(0,256,(24,32,3),dtype=np.uint8)).save(path);paths.append(path)
            report=encode_lossless(paths,root/'camera.mkv',30,shutil.which('ffmpeg'))
            self.assertTrue(report['pixel_identity_verified']);self.assertEqual(report['frame_count'],3)

    @unittest.skipUnless(shutil.which('ffmpeg'),'Existing ffmpeg required for failure cleanup test')
    def test_missing_frame_terminates_owned_encoder(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); first=root/'first.png'
            Image.fromarray(np.zeros((24,32,3),dtype=np.uint8)).save(first)
            processes=[]; real_popen=subprocess.Popen
            def capture_process(*args,**kwargs):
                process=real_popen(*args,**kwargs);processes.append(process);return process
            with patch('temporal_renderer.subprocess.Popen',side_effect=capture_process):
                with self.assertRaises(FileNotFoundError):
                    encode_lossless([first,root/'missing.png'],root/'failed.mkv',30,shutil.which('ffmpeg'))
            self.assertEqual(len(processes),1)
            self.assertIsNotNone(processes[0].poll())


if __name__=='__main__':
    unittest.main()
