"""Validate production artifacts without displaying reserved final images."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from render_dataset import file_hash, project_features


def training_gallery(split_path: Path, labels: dict, parameters: list[dict]) -> None:
    # Pair anchor perturbations (cycling gradient/scene/photo backgrounds) with
    # supplementary full-orientation views (white-noise backgrounds).
    selected_indices=np.concatenate([np.array([index*200+1+index%3,index*200+100]) for index in range(20)])
    tile_width,tile_height=320,260
    gallery=Image.new("RGB",(tile_width*5,tile_height*8),"white")
    for tile_index,image_index in enumerate(selected_indices):
        with Image.open(split_path/str(labels["image_names"][image_index])) as saved_image:
            annotated_image=saved_image.convert("RGB")
        annotation_draw=ImageDraw.Draw(annotated_image)
        for feature_index,pixel in enumerate(labels["projected_xy_pixels"][image_index]):
            if not labels["projection_valid"][image_index,feature_index]:
                continue
            pixel_x,pixel_y=pixel
            color_value="red" if labels["visibility"][image_index,feature_index] else "yellow"
            annotation_draw.ellipse((pixel_x-3,pixel_y-3,pixel_x+3,pixel_y+3),fill=color_value,outline="black")
        tile_x,tile_y=tile_index%5*tile_width,tile_index//5*tile_height
        gallery.paste(annotated_image.resize((320,240)),(tile_x,tile_y))
        ImageDraw.Draw(gallery).text((tile_x+2,tile_y+242),f"Training {image_index}; anchor {parameters[image_index]['canonical_anchor_index']}",fill="black")
    gallery.save(split_path/"training-gallery.png")


def validate_production(dataset_path: Path) -> dict:
    plan=json.loads((dataset_path/"production_plan.json").read_text())
    if plan["status"]!="complete":
        raise ValueError("Production is still running; do not certify partial output")
    report={"splits":{},"final_images_displayed":False,"pose_group_isolation":True,"photo_source_isolation":True}
    pose_owners={}
    photo_owners={}
    reference_features=None
    for split_name,expected_count in plan["counts"].items():
        split_path=dataset_path/split_name
        manifest=json.loads((split_path/"manifest.json").read_text())
        if file_hash(split_path/"labels.npz")!=manifest["label_sha256"]:
            raise ValueError("Labels no longer match saved hash")
        with np.load(split_path/"labels.npz",allow_pickle=False) as saved_labels:
            labels={array_name:saved_labels[array_name] for array_name in saved_labels.files}
        if labels["projected_xy_pixels"].shape!=(expected_count,50,2):
            raise ValueError("Incorrect dense projection dimensions")
        if reference_features is None:
            reference_features=labels["feature_names"].copy()
        np.testing.assert_array_equal(reference_features,labels["feature_names"])
        recomputed_pixels,recomputed_valid=project_features(labels["feature_positions_mm"],labels["camera_to_object"],labels["camera_intrinsics"],(640,480),.01,1000.)
        np.testing.assert_allclose(recomputed_pixels,labels["projected_xy_pixels"],rtol=0,atol=1e-10)
        np.testing.assert_array_equal(recomputed_valid,labels["projection_valid"])
        image_bytes=0
        for image_index,image_name in enumerate(labels["image_names"]):
            image_path=split_path/str(image_name)
            if file_hash(image_path)!=manifest["image_sha256"][str(image_name)]:
                raise ValueError(f"Image hash mismatch: {image_path}")
            # Header/dimensions only for final images; no final gallery or visual selection.
            with Image.open(image_path) as saved_image:
                if saved_image.size!=(640,480):
                    raise ValueError("Image dimensions differ from native model input")
            image_bytes+=image_path.stat().st_size
            surface_name=f"surface_arrays/image_{image_index:06d}.npz"
            if file_hash(split_path/surface_name)!=manifest["surface_sha256"][surface_name]:
                raise ValueError("Surface array hash mismatch")
            parameters=manifest["image_parameters"][image_index]
            pose_identity=parameters["pose_group"]
            if pose_identity in pose_owners and pose_owners[pose_identity]!=split_name:
                raise ValueError("Pose group leaked across splits")
            pose_owners[pose_identity]=split_name
            if "photo_source_id" in parameters:
                source_identity=parameters["photo_source_id"]
                if source_identity in photo_owners and photo_owners[source_identity]!=split_name:
                    raise ValueError("Photo source leaked across splits")
                photo_owners[source_identity]=split_name
        if split_name=="training":
            anchor_counts=np.bincount([row["canonical_anchor_index"] for row in manifest["image_parameters"]])
            np.testing.assert_array_equal(anchor_counts,np.full(20,200))
            training_gallery(split_path,labels,manifest["image_parameters"])
        used_photo_count=len({row["photo_source_id"] for row in manifest["image_parameters"] if "photo_source_id" in row})
        if split_name!="canonical_reference" and used_photo_count!=plan["photo_source_split_counts"][split_name]:
            raise ValueError("Not every assigned photographic source was exercised")
        report["splits"][split_name]={"images":expected_count,"features_per_image":50,"projection_valid":int(labels["projection_valid"].sum()),
                                      "ray_visible":int(labels["visibility"].sum()),"ray_occluded":int((~labels["visibility"]).sum()),
                                      "image_bytes":image_bytes,"render_seconds":sum(manifest["render_seconds"]),
                                      "generation_seconds":manifest["generation_seconds_before_review"],"all_image_surface_hashes_verified":True,
                                      "source_photo_count":used_photo_count}
    report["total_images"]=sum(row["images"] for row in report["splits"].values())
    (dataset_path/"validation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",type=Path)
    arguments=parser.parse_args()
    print(json.dumps(validate_production(arguments.dataset),indent=2))
