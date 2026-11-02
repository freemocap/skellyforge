"""Write explicit pose-heldout membership and inspectable projected feature overlays."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from render_dataset import file_hash


def split_membership(elevation_radians: np.ndarray, train_max_degrees: float = 45., evaluation_min_degrees: float = 55.) -> dict[str, np.ndarray]:
    if not np.isfinite(elevation_radians).all() or not train_max_degrees < evaluation_min_degrees:
        raise ValueError("Finite elevations and a positive held-out angular gap are required")
    elevation_degrees = np.rad2deg(elevation_radians)
    return {"training": np.flatnonzero(elevation_degrees <= train_max_degrees),
            "evaluation": np.flatnonzero(elevation_degrees >= evaluation_min_degrees),
            "excluded_guard_band": np.flatnonzero((elevation_degrees > train_max_degrees) & (elevation_degrees < evaluation_min_degrees))}


def review_dataset(dataset_path: Path) -> dict:
    label_path = dataset_path/"labels.npz"
    manifest = json.loads((dataset_path/"manifest.json").read_text())
    if file_hash(label_path) != manifest["label_sha256"]:
        raise ValueError("Label bytes no longer match the generation manifest")
    with np.load(label_path, allow_pickle=False) as saved_labels:
        labels = {array_name: saved_labels[array_name] for array_name in saved_labels.files}
    membership = split_membership(labels["elevation_radians"])
    if not len(membership["training"]) or not len(membership["evaluation"]):
        raise ValueError("Both training and evaluation need examples")
    image_width, image_height = manifest["image_size"]
    pixels = labels["projected_xy_pixels"]
    strict_inbounds = (np.isfinite(pixels).all(axis=-1) & (pixels[..., 0] > 0) & (pixels[..., 0] < image_width)
                       & (pixels[..., 1] > 0) & (pixels[..., 1] < image_height))
    split_report = {"policy": "training elevations <=45deg; evaluation >=55deg; intermediate poses excluded",
                    "labels_sha256": manifest["label_sha256"], "pose_grouping": "one image per sampled camera pose; membership depends only on pose elevation",
                    "minimum_elevation_gap_degrees": float(np.rad2deg(labels["elevation_radians"][membership["evaluation"]].min()-labels["elevation_radians"][membership["training"]].max())),
                    "splits": {}}
    for split_name, image_indices in membership.items():
        split_report["splits"][split_name] = {
            "image_indices": image_indices.tolist(), "image_names": labels["image_names"][image_indices].tolist(),
            "image_count": len(image_indices), "feature_count": int(len(image_indices)*len(labels["feature_names"])),
            "ray_visible_count": int(labels["visibility"][image_indices].sum()),
            "ray_occluded_count": int((~labels["visibility"][image_indices]).sum()),
            "projection_valid_count": int(labels["projection_valid"][image_indices].sum()),
            "strict_finite_inbounds_count": int(strict_inbounds[image_indices].sum())}
    # Contact sheet loops only for diagnostic drawing, never numerical label assembly.
    selected_indices = np.concatenate([membership["training"][:3], membership["evaluation"][:3]])
    review_image = Image.new("RGB", (image_width*3, (image_height+30)*2), "white")
    for tile_index, image_index in enumerate(selected_indices):
        image_name = str(labels["image_names"][image_index])
        if file_hash(dataset_path/image_name) != manifest["image_sha256"][image_name]:
            raise ValueError(f"Image bytes changed: {image_name}")
        with Image.open(dataset_path/image_name) as source_image:
            tile_image = source_image.convert("RGB")
        tile_draw = ImageDraw.Draw(tile_image)
        for feature_index, pixel_position in enumerate(pixels[image_index]):
            if not labels["projection_valid"][image_index, feature_index]:
                continue
            pixel_x, pixel_y = pixel_position
            point_color = "red" if labels["visibility"][image_index, feature_index] else "yellow"
            tile_draw.ellipse((pixel_x-3, pixel_y-3, pixel_x+3, pixel_y+3), fill=point_color, outline="black")
            tile_draw.text((pixel_x+4, pixel_y-9), str(feature_index+1), fill="black", stroke_width=1, stroke_fill="white")
        tile_x, tile_y = tile_index%3*image_width, tile_index//3*(image_height+30)
        review_image.paste(tile_image, (tile_x, tile_y))
        ImageDraw.Draw(review_image).text((tile_x+5, tile_y+image_height+5), f"{image_name} | elevation {np.rad2deg(labels['elevation_radians'][image_index]):.1f} deg | {'training' if tile_index<3 else 'evaluation'}", fill="black")
    review_image.save(dataset_path/"label-review.png")
    (dataset_path/"split_membership.json").write_text(json.dumps(split_report, indent=2), encoding="utf-8")
    return split_report


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("dataset", type=Path)
    parsed_arguments = argument_parser.parse_args()
    split_report = review_dataset(parsed_arguments.dataset)
    print(json.dumps({split_name: {key: value for key, value in split_data.items() if key not in {"image_indices", "image_names"}} for split_name, split_data in split_report["splits"].items()}, indent=2))
