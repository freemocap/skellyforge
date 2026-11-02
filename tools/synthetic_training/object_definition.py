"""Create persistent geometry identities from an STL and explicit source units."""
import json
from pathlib import Path
import re

import numpy as np

from render_dataset import SurfaceVisibility, file_hash, read_binary_stl


UNIT_SCALE_MM={'mm':1.,'cm':10.,'m':1000.,'in':25.4}


def read_stl(mesh_path: Path) -> tuple[np.ndarray,np.ndarray]:
    raw=mesh_path.read_bytes()
    if len(raw)>=84 and len(raw)==84+50*int.from_bytes(raw[80:84],'little'):
        return read_binary_stl(mesh_path)
    try:
        text=raw.decode('ascii')
    except UnicodeDecodeError as error:
        raise ValueError('Expected ordinary binary or ASCII STL') from error
    if not text.lstrip().lower().startswith('solid') or 'endsolid' not in text.lower():
        raise ValueError('ASCII STL requires solid and endsolid records')
    facets=re.findall(r'facet\s+normal\s+[^\n]+\s+outer\s+loop\s+(.*?)\s+endloop\s+endfacet',text,re.I|re.S)
    if len(facets)!=len(re.findall(r'\bfacet\s+normal\b',text,re.I)) or not facets:
        raise ValueError('Malformed ASCII STL facets')
    triangles=[]
    for facet in facets:
        rows=re.findall(r'\bvertex\s+(\S+)\s+(\S+)\s+(\S+)',facet,re.I)
        if len(rows)!=3:
            raise ValueError('Each STL facet must have three vertices')
        triangles.append(np.asarray(rows,dtype=np.float64))
    vertices,inverse=np.unique(np.asarray(triangles).reshape(-1,3),axis=0,return_inverse=True)
    if not np.isfinite(vertices).all():
        raise ValueError('STL coordinates must be finite')
    return vertices,inverse.reshape(-1,3).astype(np.int32)


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    triangles=np.asarray(vertices,dtype=np.float64)[faces]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    records=np.zeros(len(faces),dtype=dtype);records['normal']=normals;records['vertices']=triangles
    path.write_bytes(b'Generated object-frame millimetre geometry'.ljust(80,b'\0')+len(faces).to_bytes(4,'little')+records.tobytes())


def _authored_features(path: Path | None, vertices: np.ndarray, faces: np.ndarray,
                       scale_mm: float, source_hash: str) -> tuple[list[str],np.ndarray,np.ndarray]:
    if path is None:
        return [],np.empty(0,dtype=int),np.empty((0,3))
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('original_mesh_sha256',data.get('source_mesh_sha256',source_hash))!=source_hash:
        raise ValueError('Authored definition belongs to a different original mesh')
    names=data['feature_names']
    if not names or len(set(names))!=len(names) or any(not isinstance(name,str) or not name.strip() for name in names):
        raise ValueError('Authored feature identities must be nonempty unique strings')
    if 'source_triangle_indices' in data:
        indices=np.asarray(data['source_triangle_indices'],dtype=int)
        if not np.array_equal(indices,np.asarray(data['source_triangle_indices'])):
            raise ValueError('Source triangle indices must be integers')
        barycentric=np.asarray(data['barycentric_coordinates'],dtype=float)
        if indices.shape!=(len(names),) or np.any(indices<0) or np.any(indices>=len(faces)) or barycentric.shape!=(len(names),3):
            raise ValueError('Invalid authored triangle correspondences')
        if not np.isfinite(barycentric).all() or np.any(barycentric<0) or not np.allclose(barycentric.sum(1),1,atol=1e-10):
            raise ValueError('Authored barycentric weights must be finite, nonnegative and sum to one')
        return names,indices,barycentric
    if 'source_vertex_indices' in data:
        vertex_ids=np.asarray(data['source_vertex_indices'],dtype=int)
        if not np.array_equal(vertex_ids,np.asarray(data['source_vertex_indices'])):
            raise ValueError('Source vertex indices must be integers')
        if vertex_ids.shape!=(len(names),) or np.any(vertex_ids<0) or np.any(vertex_ids>=len(vertices)):
            raise ValueError('Invalid authored source vertex indices')
    elif 'source_positions_mm' in data:
        positions=np.asarray(data['source_positions_mm'],dtype=float)/scale_mm
        if positions.shape!=(len(names),3) or not np.isfinite(positions).all():
            raise ValueError('Invalid authored source positions')
        # Optional authored features are few; candidate distances remain vectorized.
        vertex_ids=np.array([np.argmin(np.linalg.norm(vertices-position,axis=1)) for position in positions])
        if not np.allclose(vertices[vertex_ids],positions,rtol=0,atol=max(np.ptp(vertices,axis=0).max()*1e-7,1e-12)):
            raise ValueError('Authored positions must match mesh vertices; use explicit triangle/barycentric coordinates for surface interiors')
    else:
        raise ValueError('Authored features require vertex or triangle/barycentric correspondences')
    face_ids=np.full(len(vertices),len(faces),dtype=int)
    face_vertices=vertices[faces]
    valid_faces=np.linalg.norm(np.cross(face_vertices[:,1]-face_vertices[:,0],face_vertices[:,2]-face_vertices[:,0]),axis=1)>0
    np.minimum.at(face_ids,faces[valid_faces].ravel(),np.repeat(np.flatnonzero(valid_faces),3))
    indices=face_ids[vertex_ids]
    if np.any(indices==len(faces)):
        raise ValueError('Authored vertices must belong to a nondegenerate facet')
    barycentric=(faces[indices]==vertex_ids[:,None]).astype(float)
    return names,indices,barycentric


def create_object_definition(mesh_path: Path, output_directory: Path, units: str,
                             feature_count: int = 50, authored_path: Path | None = None) -> dict:
    from select_geometry_features import canonical_directions, crease_strength
    if output_directory.exists():
        raise FileExistsError(output_directory)
    if units not in UNIT_SCALE_MM or isinstance(feature_count,bool) or not isinstance(feature_count,(int,np.integer)) or feature_count<3:
        raise ValueError('Explicit supported units and at least three features are required')
    source_vertices,original_faces=read_stl(mesh_path)
    scale=UNIT_SCALE_MM[units];source_hash=file_hash(mesh_path)
    center=(source_vertices.min(0)+source_vertices.max(0))*.5
    object_vertices=(source_vertices-center)*scale
    extent=np.ptp(object_vertices,axis=0).max()
    triangles=object_vertices[original_faces]
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)*.5
    valid_facets=area>extent**2*1e-14
    if not np.isfinite(extent) or extent<=0 or not valid_facets.any():
        raise ValueError('Mesh must contain finite nondegenerate triangles')
    retained_facets=np.flatnonzero(valid_facets)
    faces=original_faces[valid_facets];triangles=triangles[valid_facets];area=area[valid_facets]
    names,authored_faces,authored_weights=_authored_features(authored_path,source_vertices,original_faces,scale,source_hash)
    if not valid_facets[authored_faces].all():
        raise ValueError('Authored features cannot reference degenerate source facets')
    original_to_normalized=np.full(len(original_faces),-1,dtype=int);original_to_normalized[retained_facets]=np.arange(len(faces))
    authored_faces=original_to_normalized[authored_faces]
    if len(names)>feature_count:
        raise ValueError('Requested feature count is smaller than supplied identity count')
    strength=crease_strength(object_vertices,faces)
    crease_ids=np.flatnonzero(strength>=np.deg2rad(20))
    # Creases are preferred, but smooth/coarse meshes also need stable surface candidates.
    vertex_ids=crease_ids if len(crease_ids)>=feature_count*2 else np.unique(faces)
    face_for_vertex=np.full(len(object_vertices),len(faces),dtype=int)
    np.minimum.at(face_for_vertex,faces.ravel(),np.repeat(np.arange(len(faces)),3))
    candidate_faces=face_for_vertex[vertex_ids]
    candidate_weights=(faces[candidate_faces]==vertex_ids[:,None]).astype(float)
    fallback=len(vertex_ids)<feature_count*2
    if fallback:
        sample_count=max(1024,feature_count*32)
        sample_index=np.arange(sample_count)+.5
        surface_faces=np.searchsorted(np.cumsum(area)/area.sum(),sample_index/sample_count)
        radial=np.sqrt(np.mod(sample_index*(np.sqrt(5)-1)/2,1))
        lateral=np.mod(sample_index*(np.sqrt(2)-1),1)
        weights=np.column_stack((1-radial,radial*(1-lateral),radial*lateral))
        candidate_faces=np.concatenate((candidate_faces,surface_faces))
        candidate_weights=np.concatenate((candidate_weights,weights))
    candidate_positions=np.einsum('ni,nij->nj',candidate_weights,triangles[candidate_faces])
    directions=canonical_directions(20);directions=np.concatenate((directions,directions*[1,1,-1]))
    visibility=SurfaceVisibility(object_vertices,faces).query_visibility(directions*extent*4,candidate_positions,extent*1e-5)
    visibility_counts=visibility.sum(0)
    eligible=visibility_counts>=2
    if eligible.sum()<feature_count-len(names):
        raise ValueError('Not enough externally visible surface candidates for the requested feature count')
    candidate_positions=candidate_positions[eligible];candidate_faces=candidate_faces[eligible];candidate_weights=candidate_weights[eligible]
    quality=.6+.4*visibility_counts[eligible]/len(directions)
    authored_positions=np.einsum('ni,nij->nj',authored_weights,triangles[authored_faces])
    minimum_distance=(np.linalg.norm(candidate_positions[:,None]-authored_positions[None],axis=-1).min(1)
                      if len(names) else np.linalg.norm(candidate_positions,axis=1))
    selected=[]
    for _ in range(feature_count-len(names)):
        scores=minimum_distance*quality
        # Treat floating-point ties consistently across equivalent unit encodings.
        chosen=int(np.flatnonzero(np.isclose(scores,scores.max(),rtol=1e-7,atol=extent*1e-12))[0])
        if minimum_distance[chosen]<=extent*1e-7:
            raise ValueError('Requested features exceed distinct surface candidates')
        selected.append(chosen)
        minimum_distance=np.minimum(minimum_distance,np.linalg.norm(candidate_positions-candidate_positions[chosen],axis=1))
    selected=np.asarray(selected,dtype=int)
    selected_faces=np.concatenate((authored_faces,candidate_faces[selected]))
    selected_weights=np.concatenate((authored_weights,candidate_weights[selected]))
    positions=np.einsum('ni,nij->nj',selected_weights,triangles[selected_faces])
    if len(np.unique(positions,axis=0))!=len(positions):
        raise ValueError('Supplied feature coordinates must be distinct')
    for feature_index in range(len(names),feature_count):
        candidate_name=f'geometry_feature_{feature_index+1:04d}'
        if candidate_name in names:
            raise ValueError('Supplied feature identity collides with generated identity')
        names.append(candidate_name)
    transform=np.eye(4);transform[:3,:3]*=scale;transform[:3,3]=-center*scale
    output_directory.mkdir(parents=True)
    normalized_path=output_directory/'object_mesh_mm.stl'
    write_binary_stl(normalized_path,object_vertices,faces)
    normalized_vertices,normalized_faces=read_binary_stl(normalized_path)
    # Freeze labels on the exact float32 STL surface consumed by the renderer.
    rendered_positions=np.einsum('ni,nij->nj',selected_weights,normalized_vertices[normalized_faces[selected_faces]])
    definition={'schema_version':1,'status':'generated_object_definition','units':'mm','feature_names':names,
        'object_positions_mm':rendered_positions.tolist(),'source_positions_mm':rendered_positions.tolist(),
        'source_to_object_rotation':np.eye(3).tolist(),'origin_source_mm':[0.,0.,0.],'reference_unit_mm':float(extent),
        'source_mesh_sha256':file_hash(normalized_path),'normalized_mesh_path':normalized_path.name,
        'original_mesh_sha256':source_hash,'original_mesh_units':units,'original_units_to_mm':scale,
        'source_mesh_to_object':transform.tolist(),'object_to_source_mesh':np.linalg.inv(transform).tolist(),
        'frame_policy':'Object axes retain STL source axes; origin is source bounding-box centre; full invertible unit-scale/translation transform preserves original source origin',
        'source_vertex_order':'lexicographically sorted unique source STL coordinates','source_triangle_order':'original STL facet order',
        'source_triangle_indices':retained_facets[selected_faces].tolist(),'barycentric_coordinates':selected_weights.tolist(),
        'normalized_triangle_indices':selected_faces.tolist(),'normalized_to_original_triangle_indices':retained_facets.tolist(),
        'discarded_degenerate_source_triangle_indices':np.flatnonzero(~valid_facets).tolist(),
        'degenerate_area_threshold_mm_squared':float(extent**2*1e-14),
        'source_vertex_indices':[int(faces[face,np.argmax(weight)]) if np.count_nonzero(weight)==1 else -1 for face,weight in zip(selected_faces,selected_weights)],
        'original_positions_source_units':np.einsum('ni,nij->nj',selected_weights,source_vertices[faces[selected_faces]]).tolist(),
        'normalization_rounding_max_mm':float(np.max(np.abs(rendered_positions-positions))),
        'selection':{'authored_count':len(authored_faces),'crease_candidate_count':len(crease_ids),'candidate_count':len(candidate_positions),'surface_sampling_fallback':fallback,
                     'candidate_mode':'vertices_and_triangle_samples' if fallback else 'crease_vertices' if len(crease_ids)>=feature_count*2 else 'surface_vertices',
                     'method':'20-degree creases preferred; deterministic area-weighted triangle samples when sparse; visibility-weighted farthest-point ordering','visibility_probes':40,'minimum_visible_probes':2},
        'ambiguity_warning':'Fixed geometric identities may be visually ambiguous on symmetric or textureless surfaces; selection does not prove visual trackability or enforce rigidity'}
    path=output_directory/'object_definition.json';path.write_text(json.dumps(definition,indent=2),encoding='utf-8')
    return {'definition_path':str(path),'mesh_path':str(normalized_path),'feature_count':feature_count,'definition_sha256':file_hash(path),'original_mesh_sha256':source_hash,'selection':definition['selection']}
