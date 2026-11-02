import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from object_definition import create_object_definition,read_stl,write_binary_stl
from render_dataset import FeatureDefinition,file_hash,read_binary_stl


def tetrahedron():
    return np.array([[10,20,30],[30,20,30],[10,40,30],[10,20,50]],dtype=float),np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])


def write_ascii(path,vertices,faces):
    lines=['solid fixture']
    for face in faces:
        lines.extend(['facet normal 0 0 0','outer loop'])
        lines.extend('vertex '+' '.join(format(value,'.17g') for value in vertices[index]) for index in face)
        lines.extend(['endloop','endfacet'])
    path.write_text('\n'.join(lines+['endsolid fixture']))


class ObjectDefinitionTests(unittest.TestCase):
    def test_smooth_mesh_can_select_features_without_sharp_creases(self):
        latitude_count,longitude_count=20,40
        latitudes=np.arange(1,latitude_count)*np.pi/latitude_count
        longitudes=np.arange(longitude_count)*2*np.pi/longitude_count
        latitude,longitude=np.meshgrid(latitudes,longitudes,indexing='ij')
        rings=np.stack((np.sin(latitude)*np.cos(longitude),np.sin(latitude)*np.sin(longitude),np.cos(latitude)),axis=-1).reshape(-1,3)
        vertices=np.vstack(([0,0,1],rings,[0,0,-1]))*20
        faces=[]
        for column in range(longitude_count):
            following=(column+1)%longitude_count
            faces.append([0,1+column,1+following])
            for row in range(latitude_count-2):
                first=1+row*longitude_count;second=first+longitude_count
                faces.extend([[first+column,second+column,first+following],[first+following,second+column,second+following]])
            last=1+(latitude_count-2)*longitude_count
            faces.append([len(vertices)-1,last+following,last+column])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);mesh=root/'smooth.stl';write_binary_stl(mesh,vertices,np.asarray(faces))
            result=create_object_definition(mesh,root/'out','mm',30)
            self.assertLess(result['selection']['crease_candidate_count'],60)
            self.assertEqual(result['feature_count'],30)

    def test_stl_only_sparse_mesh_has_deterministic_surface_identities(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);mesh=root/'shape.stl';vertices,faces=tetrahedron();write_binary_stl(mesh,vertices,faces)
            first=create_object_definition(mesh,root/'first','mm',24)
            second=create_object_definition(mesh,root/'second','mm',24)
            self.assertEqual(first['definition_sha256'],second['definition_sha256'])
            definition=json.loads(Path(first['definition_path']).read_text())
            self.assertTrue(definition['selection']['surface_sampling_fallback'])
            self.assertEqual(len(definition['feature_names']),24)
            self.assertTrue(any(index==-1 for index in definition['source_vertex_indices']))
            source_vertices,source_faces=read_stl(mesh)
            correspondence=np.einsum('ni,nij->nj',definition['barycentric_coordinates'],source_vertices[source_faces[definition['source_triangle_indices']]])
            np.testing.assert_allclose(correspondence,definition['original_positions_source_units'])
            transform=np.array(definition['source_mesh_to_object']);inverse=np.array(definition['object_to_source_mesh'])
            np.testing.assert_allclose(transform@inverse,np.eye(4))
            np.testing.assert_allclose((correspondence@transform[:3,:3].T+transform[:3,3]),definition['object_positions_mm'],atol=1e-6)
            loaded=FeatureDefinition.load_file(Path(first['definition_path']))
            self.assertEqual(loaded.mesh_sha256,file_hash(Path(first['mesh_path'])))
            np.testing.assert_array_equal(loaded.source_origin_mm,[0,0,0])
            self.assertEqual(definition['original_mesh_sha256'],file_hash(mesh))

    def test_ascii_binary_and_unit_equivalence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);vertices,faces=tetrahedron()
            write_ascii(root/'mm.stl',vertices,faces);write_ascii(root/'m.stl',vertices/1000,faces)
            write_binary_stl(root/'binary.stl',vertices,faces)
            av,af=read_stl(root/'mm.stl');bv,bf=read_stl(root/'binary.stl')
            np.testing.assert_array_equal(av,bv);np.testing.assert_array_equal(af,bf)
            a=create_object_definition(root/'mm.stl',root/'a','mm',12);b=create_object_definition(root/'m.stl',root/'b','m',12)
            ad=json.loads(Path(a['definition_path']).read_text());bd=json.loads(Path(b['definition_path']).read_text())
            self.assertEqual(ad['feature_names'],bd['feature_names']);self.assertEqual(ad['source_triangle_indices'],bd['source_triangle_indices'])
            np.testing.assert_allclose(ad['object_positions_mm'],bd['object_positions_mm'],atol=1e-5)
            self.assertEqual(bd['original_units_to_mm'],1000)

    def test_optional_authored_order_and_correspondence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);vertices,faces=tetrahedron();mesh=root/'shape.stl';write_binary_stl(mesh,vertices,faces)
            authored=root/'authored.json';authored.write_text(json.dumps({'feature_names':['front_corner','side_center'],
                'source_triangle_indices':[0,1],'barycentric_coordinates':[[1,0,0],[.2,.3,.5]]}))
            result=create_object_definition(mesh,root/'out','mm',10,authored)
            data=json.loads(Path(result['definition_path']).read_text())
            self.assertEqual(data['feature_names'][:2],['front_corner','side_center'])
            np.testing.assert_allclose(data['barycentric_coordinates'][:2],[[1,0,0],[.2,.3,.5]])

    def test_flat_mesh_uses_source_axes_without_a_degenerate_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);mesh=root/'flat.stl';write_binary_stl(mesh,np.array([[0,0,4],[20,0,4],[0,20,4]]),np.array([[0,1,2]]))
            result=create_object_definition(mesh,root/'out','mm',10)
            data=json.loads(Path(result['definition_path']).read_text())
            self.assertEqual(np.linalg.det(np.array(data['source_mesh_to_object'])[:3,:3]),1)
            np.testing.assert_allclose(np.array(data['object_positions_mm'])[:,2],0)

    def test_rejects_unknown_units_degenerate_facets_and_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);vertices,faces=tetrahedron();mesh=root/'shape.stl';write_binary_stl(mesh,vertices,faces)
            with self.assertRaises(ValueError):create_object_definition(mesh,root/'bad','unknown',8)
            create_object_definition(mesh,root/'out','mm',8)
            with self.assertRaises(FileExistsError):create_object_definition(mesh,root/'out','mm',8)
            write_ascii(root/'bad.stl',np.array([[0,0,0],[1,0,0],[2,0,0]]),np.array([[0,1,2]]))
            with self.assertRaises(ValueError):create_object_definition(root/'bad.stl',root/'flat','mm',8)

    def test_degenerate_facets_are_explicitly_mapped_and_preserved_in_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);vertices,faces=tetrahedron();mesh=root/'mixed.stl'
            write_ascii(mesh,vertices,np.vstack(([0,0,0],faces)))
            original_hash=file_hash(mesh)
            result=create_object_definition(mesh,root/'out','mm',9)
            data=json.loads(Path(result['definition_path']).read_text())
            self.assertEqual(data['discarded_degenerate_source_triangle_indices'],[0])
            self.assertEqual(data['normalized_to_original_triangle_indices'],[1,2,3,4])
            np.testing.assert_array_equal(np.array(data['normalized_triangle_indices'])+1,data['source_triangle_indices'])
            self.assertEqual(original_hash,file_hash(mesh))


if __name__=='__main__':
    unittest.main()
