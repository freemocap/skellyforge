"""Deterministic crease candidates ranked by spread and multi-view visibility."""
import argparse
import json
from pathlib import Path

import numpy as np



def canonical_directions(direction_count: int = 20) -> np.ndarray:
    """Fibonacci hemisphere: evenly spaced equal-area height bands, positive Z."""
    height_values=(np.arange(direction_count)+.5)/direction_count
    azimuth_values=np.arange(direction_count)*np.pi*(3-np.sqrt(5))
    horizontal_radius=np.sqrt(1-height_values**2)
    return np.column_stack((horizontal_radius*np.cos(azimuth_values),horizontal_radius*np.sin(azimuth_values),height_values))


def crease_strength(vertex_positions: np.ndarray, triangle_indices: np.ndarray) -> np.ndarray:
    triangle_positions=vertex_positions[triangle_indices]
    face_normals=np.cross(triangle_positions[:,1]-triangle_positions[:,0],triangle_positions[:,2]-triangle_positions[:,0])
    normal_lengths=np.linalg.norm(face_normals,axis=1)
    face_normals/=np.maximum(normal_lengths[:,None],1e-15)
    edge_vertices=np.sort(triangle_indices[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    face_ids=np.repeat(np.arange(len(triangle_indices)),3)
    unique_edges,edge_inverse,edge_counts=np.unique(edge_vertices,axis=0,return_inverse=True,return_counts=True)
    first_faces=np.full(len(unique_edges),len(triangle_indices),dtype=np.int64)
    last_faces=np.zeros(len(unique_edges),dtype=np.int64)
    np.minimum.at(first_faces,edge_inverse,face_ids)
    np.maximum.at(last_faces,edge_inverse,face_ids)
    edge_angles=np.arccos(np.clip(np.sum(face_normals[first_faces]*face_normals[last_faces],axis=1),-1,1))
    edge_angles[edge_counts!=2]=np.pi
    vertex_strength=np.zeros(len(vertex_positions))
    np.maximum.at(vertex_strength,unique_edges.ravel(),np.repeat(edge_angles,2))
    return vertex_strength


def select_features(mesh_path: Path, output_directory: Path, units: str,
                    feature_count: int = 50, original_path: Path | None = None) -> dict:
    """Build a generic object definition; authored seed identities are optional."""
    from object_definition import create_object_definition
    return create_object_definition(mesh_path,output_directory,units,feature_count,original_path)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh",type=Path,required=True)
    parser.add_argument("--original",type=Path,help="Optional authored vertex or triangle/barycentric identities")
    parser.add_argument("--units",choices=("mm","cm","m","in"),required=True)
    parser.add_argument("--feature-count",type=int,default=50)
    parser.add_argument("--output",type=Path,required=True)
    arguments=parser.parse_args()
    print(json.dumps(select_features(arguments.mesh,arguments.output,arguments.units,arguments.feature_count,arguments.original)))
