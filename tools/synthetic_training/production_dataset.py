"""Generate the approved canonical-anchor production dataset on bounded CPU threads."""
import argparse
import colorsys
import json
import os
import sys
from pathlib import Path
import time

import numpy as np

from randomized_renderer import render_preview, load_photos
from render_dataset import FeatureDefinition, file_hash, read_binary_stl
from select_geometry_features import canonical_directions


def production_parameters(split_name: str, random_seed: int, object_extent_mm: float,
                          direction_count: int = 20, images_per_direction: int = 200) -> list[dict]:
    random_generator=np.random.default_rng(random_seed)
    anchor_directions=canonical_directions(direction_count)
    image_count=direction_count if split_name=="canonical_reference" else direction_count*images_per_direction if split_name=="training" else 400
    parameter_rows=[]
    for image_index in range(image_count):
        anchor_index=image_index//images_per_direction if split_name=="training" else image_index%direction_count
        supplementary=split_name!="canonical_reference" and image_index%10==0
        if split_name=="canonical_reference":
            view_direction=anchor_directions[image_index]
        elif split_name=="training" and not supplementary:
            view_direction=anchor_directions[anchor_index]+random_generator.normal(0,.17,3)
            view_direction/=np.linalg.norm(view_direction)
        else:
            view_direction=random_generator.normal(size=3)
            if not supplementary:
                view_direction[2]=abs(view_direction[2])
            view_direction/=np.linalg.norm(view_direction)
        canonical=split_name=="canonical_reference"
        color_rgb=[.5,.5,.5] if canonical else colorsys.hsv_to_rgb(random_generator.random(),random_generator.uniform(.1,.95),random_generator.uniform(.12,.85))
        parameter_rows.append({"pose_group":f"{split_name}_{random_seed}_{image_index:06d}","background_group":f"procedural_{split_name}_{random_seed}_{image_index:06d}",
                               "split":split_name,"canonical_anchor_index":int(anchor_index),"supplementary_full_orientation":supplementary,
                               "azimuth_radians":float(np.arctan2(view_direction[1],view_direction[0])),"elevation_radians":float(np.arcsin(view_direction[2])),
                               "roll_radians":0. if canonical else float(random_generator.uniform(-np.pi,np.pi)),
                               "distance_mm":float(object_extent_mm*(3.5 if canonical else random_generator.uniform(1.6,5.5))),
                               "image_center_fraction":[.5,.5] if canonical else random_generator.uniform(.30,.70,2).tolist(),
                               "color_rgb":color_rgb,"material_kind":"diffuse" if canonical else ("diffuse","roughplastic")[int(random_generator.integers(2))],
                               "roughness":float(random_generator.uniform(.03,.45)),"ambient_rgb":[.5,.5,.5] if canonical else random_generator.uniform(.05,.7,3).tolist(),
                               "light_position_mm":[60,-90,140] if canonical else (random_generator.normal(size=3)*object_extent_mm*3).tolist(),
                               "light_rgb":[12000]*3 if canonical else (random_generator.uniform(.3,1,3)*random_generator.uniform(1,8)*object_extent_mm**2).tolist(),
                               "background_kind":"solid_reference" if canonical else ("white_noise","gradient_noise","generated_scene","photograph")[image_index%4],
                               "background_seed":int(random_generator.integers(0,2**30)),"render_seed":int(random_generator.integers(0,2**30))})
    return parameter_rows


def run_production(mesh_path: Path, feature_path: Path, output_path: Path, photo_directory: Path,
                   photo_manifest: Path, cpu_threads: int = 4, seed_offset: int = 0) -> None:
    import drjit as dr
    dr.set_thread_count(cpu_threads)
    if seed_offset<0:
        raise ValueError("Seed offset must be nonnegative")
    if output_path.exists():
        raise FileExistsError(output_path)
    feature_definition=FeatureDefinition.load_file(feature_path)
    if len(feature_definition.feature_names)!=50:
        raise ValueError("This approved experiment requires the fixed 50-feature manifest")
    source_vertices,_=read_binary_stl(mesh_path)
    object_vertices=(source_vertices-feature_definition.source_origin_mm)@feature_definition.source_rotation.T
    object_extent_mm=float(np.ptp(object_vertices,axis=0).max())
    photo_sources,_=load_photos(photo_directory,photo_manifest)
    source_counts={split_name:sum(row["split"]==split_name for row in photo_sources) for split_name in ("training","development","final_evaluation")}
    if len(photo_sources)<100 or source_counts["training"]<80 or source_counts["development"]<10 or source_counts["final_evaluation"]<10:
        raise ValueError("Production requires at least 100 distinct verified photo sources, split at least 80/10/10")
    output_path.mkdir(parents=True)
    snapshot_path=output_path/"source_snapshot"
    snapshot_path.mkdir()
    for source_name in ("production_dataset.py","randomized_renderer.py","render_dataset.py","select_geometry_features.py","mesh_framing.py"):
        (snapshot_path/source_name).write_bytes(Path(__file__).with_name(source_name).read_bytes())
    (snapshot_path/"photo_sources.json").write_bytes(photo_manifest.read_bytes())
    photo_manifest=snapshot_path/"photo_sources.json"
    (snapshot_path/"features_50.json").write_bytes(feature_path.read_bytes())
    plan={"status":"running","pid":os.getpid(),"cpu_threads":cpu_threads,"feature_sha256":file_hash(feature_path),
          "mesh_sha256":file_hash(mesh_path),"counts":{"canonical_reference":20,"training":4000,"development":400,"final_evaluation":400},
          "seed_offset":seed_offset,
          "split_seeds":{name:seed+seed_offset for name,seed in {"canonical_reference":2026110201,"training":2026110202,"development":2026110203,"final_evaluation":2026110204}.items()},
          "final_images_must_not_be_previewed":True,"supplementary_full_orientation_fraction":.1,
          "photo_source_count":len(photo_sources),"photo_source_split_counts":source_counts,"photo_manifest_sha256":file_hash(photo_manifest),
          "source_snapshot_hashes":{path.name:file_hash(path) for path in snapshot_path.iterdir()},"command_arguments":sys.argv,"working_directory":str(Path.cwd()),
          "photo_pool_limitation":"Verified source-disjoint photographs diversify backgrounds but do not prove real-camera transfer quality.",
          "canonical_directions":canonical_directions().tolist(),"completed_splits":[]}
    (output_path/"production_plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    print(json.dumps({"pid":os.getpid(),"status":"started","output":str(output_path),"counts":plan["counts"]}),flush=True)
    all_parameters={split:production_parameters(split,seed,object_extent_mm) for split,seed in plan["split_seeds"].items()}
    (output_path/"planned_parameters.json").write_text(json.dumps(all_parameters,indent=2),encoding="utf-8")
    started_time=time.perf_counter()
    for split_name,parameters in all_parameters.items():
        manifest=render_preview(mesh_path,feature_path,output_path/split_name,image_count=len(parameters),
                                random_seed=plan["split_seeds"][split_name],samples_per_pixel=16,
                                photo_directory=None if split_name=="canonical_reference" else photo_directory,
                                photo_manifest=None if split_name=="canonical_reference" else photo_manifest,
                                parameters_override=parameters,create_review=split_name=="canonical_reference")
        manifest["purpose"]="canonical reference" if split_name=="canonical_reference" else "production synthetic dataset"
        manifest["split_policy"]="Independent split seeds and pose groups; photograph source identity confined to one split; canonical references excluded from learning/evaluation; final images reserved"
        (output_path/split_name/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        plan["completed_splits"].append(split_name)
        plan["elapsed_seconds"]=time.perf_counter()-started_time
        (output_path/"production_plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    plan["status"]="complete"
    (output_path/"production_plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    print(json.dumps(plan),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh",type=Path,required=True)
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--photo-directory",type=Path,required=True)
    parser.add_argument("--photo-manifest",type=Path,required=True)
    parser.add_argument("--cpu-threads",type=int,default=4)
    parser.add_argument("--seed-offset",type=int,default=0)
    arguments=parser.parse_args()
    run_production(arguments.mesh,arguments.features,arguments.output,arguments.photo_directory,arguments.photo_manifest,arguments.cpu_threads,arguments.seed_offset)
