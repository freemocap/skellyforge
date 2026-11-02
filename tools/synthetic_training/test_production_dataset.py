import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from production_dataset import production_parameters
from select_geometry_features import canonical_directions, crease_strength
from finalize_production import finalize_production
from render_dataset import file_hash


class ProductionDatasetTests(unittest.TestCase):
    def test_finalizer_preserves_split_paths_and_excludes_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            split_names=("training","development","final_evaluation")
            (root/"production_plan.json").write_text(json.dumps({"status":"complete","counts":{name:1 for name in split_names},"split_seeds":{name:index for index,name in enumerate(split_names)}}))
            for split_name in split_names:
                split_path=root/split_name
                split_path.mkdir()
                np.savez_compressed(split_path/"labels.npz",image_names=np.array(["images/image_000000.png"]),feature_names=np.array(["feature_one"]),
                                    feature_positions_mm=np.zeros((1,3)),projected_xy_pixels=np.ones((1,1,2)),projection_valid=np.ones((1,1),dtype=bool),visibility=np.ones((1,1),dtype=bool))
                (split_path/"manifest.json").write_text(json.dumps({"label_sha256":file_hash(split_path/"labels.npz"),"image_parameters":[{"split":split_name}],
                                                                    "image_sha256":{"images/image_000000.png":"fixture"},"surface_sha256":{},"render_seconds":[1]}))
            result=finalize_production(root)
            self.assertEqual(result["images"],3)
            self.assertTrue(result["canonical_excluded"])
            with np.load(root/"labels.npz",allow_pickle=False) as labels:
                np.testing.assert_array_equal(labels["image_names"],[f"{name}/images/image_000000.png" for name in split_names])
            membership=json.loads((root/"split_membership.json").read_text())
            self.assertEqual(membership["splits"]["final_evaluation"]["image_indices"],[2])
            with self.assertRaises(FileExistsError):
                finalize_production(root)

    def test_canonical_directions_are_upper_equal_area_bands(self):
        directions=canonical_directions(20)
        np.testing.assert_allclose(np.linalg.norm(directions,axis=1),1)
        self.assertTrue(np.all(directions[:,2]>0))
        np.testing.assert_allclose(np.diff(directions[:,2]),1/20)

    def test_production_counts_and_split_identity(self):
        training=production_parameters("training",2,60)
        development=production_parameters("development",3,60)
        final=production_parameters("final_evaluation",4,60)
        reference=production_parameters("canonical_reference",1,60)
        self.assertEqual([len(reference),len(training),len(development),len(final)],[20,4000,400,400])
        np.testing.assert_array_equal(np.bincount([row["canonical_anchor_index"] for row in training]),np.full(20,200))
        self.assertEqual(sum(row["supplementary_full_orientation"] for row in training),400)
        group_sets=[{row["pose_group"] for row in rows} for rows in (reference,training,development,final)]
        self.assertEqual(len(set.union(*group_sets)),4820)
        self.assertTrue(all(row["background_kind"]=="solid_reference" for row in reference))

    def test_crease_strength_excludes_coplanar_diagonal(self):
        vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],dtype=float)
        strength=crease_strength(vertices,np.array([[0,1,2],[0,2,3]]))
        # Boundary vertices remain eligible even though the internal diagonal is flat.
        np.testing.assert_allclose(strength,np.pi)


if __name__=="__main__":
    unittest.main()
