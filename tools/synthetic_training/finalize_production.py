"""Expose one dense dataset manifest without copying images or including references."""
import argparse
import json
from pathlib import Path

import numpy as np

from render_dataset import file_hash


def finalize_production(dataset_path: Path) -> dict:
    if (dataset_path/"labels.npz").exists() or (dataset_path/"manifest.json").exists():
        raise FileExistsError("Combined dataset already exists; preserve its identity")
    production_plan=json.loads((dataset_path/"production_plan.json").read_text())
    if production_plan["status"]!="complete":
        raise ValueError("Cannot finalize incomplete production")
    split_names=("training","development","final_evaluation")
    shared_arrays={}
    combined_arrays={}
    split_manifest={"policy":"Independent pose and original-background-source groups; canonical reference images excluded; final_evaluation reserved",
                    "splits":{},"canonical_reference_directory":"canonical_reference","canonical_reference_in_learning_rows":False}
    combined_parameters=[]
    image_hashes={}
    surface_hashes={}
    row_offset=0
    manifests=[]
    for split_name in split_names:
        split_path=dataset_path/split_name
        manifest=json.loads((split_path/"manifest.json").read_text())
        if file_hash(split_path/"labels.npz")!=manifest["label_sha256"]:
            raise ValueError("Split label hash changed")
        with np.load(split_path/"labels.npz",allow_pickle=False) as source_labels:
            labels={array_name:source_labels[array_name] for array_name in source_labels.files}
        image_names=np.asarray([f"{split_name}/{image_name}" for image_name in labels["image_names"]])
        row_count=len(image_names)
        if row_count!=production_plan["counts"][split_name]:
            raise ValueError("Split count differs from the production plan")
        labels["image_names"]=image_names
        for array_name,array_values in labels.items():
            if array_name in {"feature_names","feature_positions_mm"}:
                if array_name in shared_arrays:
                    np.testing.assert_array_equal(shared_arrays[array_name],array_values)
                else:
                    shared_arrays[array_name]=array_values
            else:
                if len(array_values)!=row_count:
                    raise ValueError(f"Unknown non-image array layout: {array_name}")
                combined_arrays.setdefault(array_name,[]).append(array_values)
        split_manifest["splits"][split_name]={"image_indices":list(range(row_offset,row_offset+row_count)),"image_names":image_names.tolist(),
                                              "image_count":row_count,"projection_valid_count":int(labels["projection_valid"].sum()),
                                              "ray_visible_count":int(labels["visibility"].sum()),"ray_occluded_count":int((~labels["visibility"]).sum())}
        row_offset+=row_count
        combined_parameters.extend(manifest["image_parameters"])
        image_hashes.update({f"{split_name}/{key}":value for key,value in manifest["image_sha256"].items()})
        surface_hashes.update({f"{split_name}/{key}":value for key,value in manifest["surface_sha256"].items()})
        manifests.append(manifest)
    dense_arrays={array_name:np.concatenate(array_parts,axis=0) for array_name,array_parts in combined_arrays.items()}
    dense_arrays.update(shared_arrays)
    np.savez_compressed(dataset_path/"labels.npz",**dense_arrays)
    split_manifest["labels_sha256"]=file_hash(dataset_path/"labels.npz")
    (dataset_path/"split_membership.json").write_text(json.dumps(split_manifest,indent=2),encoding="utf-8")
    manifest={key:value for key,value in manifests[0].items() if key not in {"image_parameters","image_sha256","surface_sha256","label_sha256","split_indices","image_count","random_seed","render_seconds","generation_seconds_before_review"}}
    manifest.update({"purpose":"Production synthetic dataset with explicit development and reserved final groups","image_count":row_offset,
                     "image_parameters":combined_parameters,"image_sha256":image_hashes,"surface_sha256":surface_hashes,
                     "label_sha256":split_manifest["labels_sha256"],"split_membership_sha256":file_hash(dataset_path/"split_membership.json"),
                     "split_seeds":production_plan["split_seeds"],"canonical_reference_directory":"canonical_reference",
                     "canonical_reference_in_learning_rows":False,"final_images_displayed":False,
                     "split_policy":split_manifest["policy"],"production_plan_sha256":file_hash(dataset_path/"production_plan.json"),
                     "render_seconds":[value for source_manifest in manifests for value in source_manifest["render_seconds"]]})
    # The frozen production renderer retained an obsolete preview-only prose string.
    # Preserve its split manifests, but correct the combined artifact's description
    # from actual provenance rather than repeating the three-source preview claim.
    manifest["limitations"]=[value for value in manifest.get("limitations",[]) if value!="Photographic source pool is only three preview sources"]
    if "photo_source_count" in production_plan:
        manifest["limitations"].append(f"Photographic source pool contains {production_plan['photo_source_count']} reviewed sources; real-camera transfer remains unverified")
        manifest["metadata_corrections"]=["Replaced obsolete three-source preview limitation with the verified production photo count; source split manifests remain unchanged"]
    (dataset_path/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return {"images":row_offset,"split_counts":{name:data["image_count"] for name,data in split_manifest["splits"].items()},
            "labels_sha256":manifest["label_sha256"],"canonical_excluded":True}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",type=Path)
    arguments=parser.parse_args()
    print(json.dumps(finalize_production(arguments.dataset),indent=2))
