"""Static presentation geometry. Never consulted by reconstruction or scale fitting.

Contours use segment-local authored units; world = origin + rotation @ (local *
segment_length_mm / authored_segment_length). Compile once per definition.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import re

import numpy as np

from skellyforge.core.skeleton.components.rigid_body_segment import RigidBodySegment


@dataclass(frozen=True, slots=True)
class PresentationMaterial:
    color: str
    opacity: float

    def __post_init__(self):
        if not re.fullmatch(r'#[0-9a-fA-F]{6}', self.color) or not np.isfinite(self.opacity) or not 0 <= self.opacity <= 1:
            raise ValueError('Presentation material requires a hex RGB color and opacity in [0, 1]')


@dataclass(frozen=True, slots=True)
class ConnectionMaterial:
    group: str
    material: PresentationMaterial

@dataclass(frozen=True, slots=True)
class SuperquadricContours:
    segment: str
    crown: str
    chin: str
    left_ear: str
    right_ear: str
    left_eye: str
    right_eye: str
    vertical_exponent: float = 1.0
    horizontal_exponent: float = 0.85
    posterior_depth: float = 0.8
    samples: int = 64
    # Absent in older snapshots: preserve their volume-centered ring and no surfaces.
    nose: str | None = None
    line_material: PresentationMaterial | None = None
    shell_material: PresentationMaterial | None = None
    side_material: PresentationMaterial | None = None

@dataclass(frozen=True, slots=True)
class Presentation:
    contours: tuple[SuperquadricContours, ...] = ()
    excluded_origin_edges: tuple[tuple[str, str], ...] = ()
    excluded_bones: tuple[str, ...] = ()
    connection_materials: tuple[ConnectionMaterial, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping) -> Presentation:
        unknown = set(value) - {'contours', 'excluded_origin_edges', 'excluded_bones', 'connection_materials'}
        if unknown:
            raise ValueError(f'Unknown presentation fields: {sorted(unknown)}')
        def contour(item):
            fields = dict(item)
            for name in ('line_material', 'shell_material', 'side_material'):
                if isinstance(fields.get(name), Mapping):
                    fields[name] = PresentationMaterial(**fields[name])
            return SuperquadricContours(**fields)
        return cls(
            contours=tuple(contour(item) for item in value.get('contours', ())),
            excluded_origin_edges=tuple(tuple(edge) for edge in value.get('excluded_origin_edges', ())),
            excluded_bones=tuple(value.get('excluded_bones', ())),
            connection_materials=tuple(ConnectionMaterial(group=item['group'], material=PresentationMaterial(**item['material']))
                                       for item in value.get('connection_materials', ())),
        )

    def compile(self, segments: Mapping[str, RigidBodySegment]) -> tuple[AttachedLines, ...]:
        for edge in self.excluded_origin_edges:
            if len(edge) != 2 or any(name not in segments for name in edge):
                raise ValueError(f'Invalid presentation origin edge: {edge}')
        if any(name not in segments for name in self.excluded_bones):
            raise ValueError('Unknown presentation bone')
        return tuple(compile_contours(recipe=item, segment=segments[item.segment]) for item in self.contours)


@dataclass(frozen=True, slots=True)
class AttachedSurface:
    positions: tuple[float, ...]
    indices: tuple[int, ...]
    material: PresentationMaterial
    double_sided: bool = False


@dataclass(frozen=True, slots=True)
class AttachedLines:
    segment: str
    authored_length: float
    # Consecutive endpoint pairs, in segment-local coordinates (not rest-world).
    positions: tuple[float, ...]
    material: PresentationMaterial | None = None
    surfaces: tuple[AttachedSurface, ...] = ()


def compile_contours(*, recipe: SuperquadricContours, segment: RigidBodySegment) -> AttachedLines:
    """Five closed sections and optional static surfaces of an asymmetric superellipsoid.

    Signed-power parameterization: [@barr1981Superquadrics], [@mcgillSuperquadrics].
    Simplified construction lines inspired by [@prokoLoomisHead]; the fit and
    posterior asymmetry are project choices, not an anatomical head model.
    Local axes must be +X right, +Y anterior, +Z superior. Bilateral landmarks
    define symmetric spans; eyes/ears constrain the surface. Nose only orients
    the optional ear-to-nose section; it never changes the fitted volume.
    """
    e1, e2, back = recipe.vertical_exponent, recipe.horizontal_exponent, recipe.posterior_depth
    if not all(np.isfinite(v) and 0.2 <= v <= 2 for v in (e1, e2, back)):
        raise ValueError('Contour exponents and posterior depth must be finite in [0.2, 2]')
    if not isinstance(recipe.samples, int) or not 16 <= recipe.samples <= 256:
        raise ValueError('Contour samples must be an integer in [16, 256]')
    p = np.array([segment.landmarks[name].local_position.array for name in (
        recipe.crown, recipe.chin, recipe.left_ear, recipe.right_ear, recipe.left_eye, recipe.right_eye)])
    if not np.isfinite(p).all() or not np.isfinite(segment.length) or segment.length <= 0:
        raise ValueError('Contour landmarks and authored length must be finite')
    crown, chin, ear_l, ear_r, eye_l, eye_r = p
    cz = (crown[2] + chin[2]) / 2
    c = (crown[2] - chin[2]) / 2
    cx = (ear_l[0] + ear_r[0]) / 2
    cy = (ear_l[1] + ear_r[1]) / 2
    if c <= 0:
        raise ValueError('Contour crown must be above chin')
    def residual(z):
        return 1 - abs((z - cz) / c) ** (2 / e1)
    er, yr = residual((ear_l[2] + ear_r[2]) / 2), residual((eye_l[2] + eye_r[2]) / 2)
    if er <= 0 or yr <= 0:
        raise ValueError('Contour ears and eyes must lie between poles')
    a = abs(ear_r[0] - ear_l[0]) / 2 / er ** (e1 / 2)
    if a <= 0:
        raise ValueError('Contour ears must span a width')
    denominator = yr ** (e1 / e2) - (abs(eye_r[0] - eye_l[0]) / (2 * a)) ** (2 / e2)
    eye_depth = (eye_l[1] + eye_r[1]) / 2 - cy
    if denominator <= 0 or eye_depth <= 0:
        raise ValueError('Contour eye span/depth cannot fit the ear-constrained volume')
    b = eye_depth / denominator ** (e2 / 2)
    t = np.arange(recipe.samples) * (2 * np.pi / recipe.samples)
    def sp(values, exponent):
        return np.sign(values) * np.abs(values) ** exponent
    sin, cos = np.sin(t), np.cos(t)
    zero = np.zeros_like(t)
    depth = np.where(sin >= 0, b, b * back)
    rings = [
        np.column_stack((a * sp(cos, e1), zero, c * sp(sin, e1))),
        np.column_stack((zero, depth * sp(sin, e1), c * sp(cos, e1))),
        np.column_stack((a * sp(cos, e2), depth * sp(sin, e2), zero)),
    ]
    # True constant-X sections: solve the implicit surface along each YZ ray.
    # This bounded solve is definition-time only, never a frame operation.
    for x in (-0.82 * a, 0.82 * a):
        lo, hi = np.zeros_like(t), np.ones_like(t)
        for _ in range(48):
            r = (lo + hi) / 2
            value = ((abs(x / a) ** (2 / e2) + np.abs(r * sin) ** (2 / e2)) ** (e2 / e1)
                     + np.abs(r * cos) ** (2 / e1))
            lo = np.where(value <= 1, r, lo)
            hi = np.where(value > 1, r, hi)
        r = (lo + hi) / 2
        rings.append(np.column_stack((np.full_like(t, x), depth * r * sin, c * r * cos)))
    center = np.array([cx, cy, cz])
    def implicit(points):
        v = points - center
        return ((np.abs(v[..., 0] / a) ** (2 / e2)
                 + np.abs(v[..., 1] / np.where(v[..., 1] >= 0, b, b * back)) ** (2 / e2)) ** (e2 / e1)
                + np.abs(v[..., 2] / c) ** (2 / e1))

    if recipe.nose is not None:
        nose = segment.landmarks[recipe.nose].local_position.array
        origin = (ear_l + ear_r) / 2
        lateral = ear_r - ear_l
        lateral /= np.linalg.norm(lateral)
        forward = nose - origin
        forward -= lateral * np.dot(forward, lateral)
        if not np.isfinite(forward).all() or np.linalg.norm(forward) <= 1e-12 or implicit(origin) >= 1:
            raise ValueError('Ear/nose section requires an interior ear midpoint and a non-collinear nose')
        forward /= np.linalg.norm(forward)
        directions = cos[:, None] * lateral + sin[:, None] * forward
        lo, hi = np.zeros_like(t), np.full_like(t, 4 * max(a, b, b * back, c))
        for _ in range(55):
            r = (lo + hi) / 2
            inside = implicit(origin + r[:, None] * directions) <= 1
            lo, hi = np.where(inside, r, lo), np.where(inside, hi, r)
        rings[2] = origin + ((lo + hi) / 2)[:, None] * directions - center

    surfaces = []
    # Bounded definition-time tessellation: 32 longitude x 16 latitude, no normals
    # needed for the unlit translucent display. Both side caps share one mesh.
    def surface(vertices, indices, material, double_sided=False):
        vertices = np.asarray(vertices)
        if not np.isfinite(vertices).all():
            raise ValueError('Nonfinite presentation surface')
        surfaces.append(AttachedSurface(tuple(vertices.reshape(-1).tolist()), tuple(indices), material, double_sided))
    if recipe.shell_material is not None:
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 33), np.linspace(-np.pi / 2, np.pi / 2, 17))
        cv = np.maximum(0, np.cos(v)) ** e1
        vertices = np.stack((a * cv * sp(np.cos(u), e2),
            np.where(np.sin(u) >= 0, b, b * back) * cv * sp(np.sin(u), e2),
            c * sp(np.sin(v), e1)), axis=-1) + center
        indices = []
        for j in range(16):
            for i in range(32):
                k = j * 33 + i
                indices.extend((k, k + 1, k + 33, k + 1, k + 34, k + 33))
        surface(vertices, indices, recipe.shell_material)
    if recipe.side_material is not None:
        vertices, indices = [], []
        for side, boundary in zip((-1, 1), rings[3:], strict=True):
            boundary = boundary[::max(1, recipe.samples // 32)]
            n, start = len(boundary), len(vertices)
            for radius in np.linspace(0, 1, 9):
                yz = boundary[:, 1:] * radius
                budget = np.maximum(0, 1 - np.abs(yz[:, 1] / c) ** (2 / e1)) ** (e1 / e2)
                budget -= np.abs(yz[:, 0] / np.where(yz[:, 0] >= 0, b, b * back)) ** (2 / e2)
                x = side * a * np.maximum(0, budget) ** (e2 / 2)
                vertices.extend((np.column_stack((x, yz)) + center).tolist())
            for j in range(8):
                for i in range(n):
                    k, next_k = start + j * n + i, start + j * n + (i + 1) % n
                    indices.extend((k, next_k, k + n, next_k, next_k + n, k + n))
        surface(vertices, indices, recipe.side_material, True)
    edges = np.concatenate([np.stack((ring + center, np.roll(ring, -1, axis=0) + center), axis=1)
                            for ring in rings]).reshape(-1)
    if not np.isfinite(edges).all():
        raise ValueError('Contour compilation produced nonfinite geometry')
    return AttachedLines(segment=segment.name, authored_length=segment.length, positions=tuple(edges.tolist()),
                         material=recipe.line_material, surfaces=tuple(surfaces))
