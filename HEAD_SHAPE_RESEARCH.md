# Head construction references and fitting direction

2026-11-01 exploratory research and implemented viewer prototype. User selected
a hyper-simplified Loomis-inspired superquadric direction. No production geometry
change yet. This concerns a sparse head visualization, not recovery of
a person's underlying bony skull from facial markers.

## Art references

- Loomis: cranial ball with flattened sides plus a separate jaw block. Useful
  structural prior for proportions and readable orientation. [@prokoLoomisHead]
- Reilly: curved construction rhythms. Watts teaches front, three-quarter, and
  profile examples and combines rhythms with Asaro planes. [@wattsReillyRhythmStructure]
- Asaro: a planar head sculpture for viewing and lighting studies.
  [@nmaAsaroHead]

The user recognized Loomis as the intended reference; the exact chart remains
unidentified. The user explicitly rejected FLAME, Asaro, nose geometry and detailed
jaw geometry. These traditions provide visual
construction principles, not a verified off-the-shelf sparse-landmark solver.

## Mathematical references

Superquadrics provide compact, deformable shape primitives. This is a useful
family to investigate for cranial roundness and front flattening, not evidence
that a single superquadric accurately represents the whole head.
[@barr1981Superquadrics]

FLAME supplies learned head-shape and expression parameters and articulation.
It is a stronger statistical shape prior, but adopting it for this small
visualization would require a separate complexity, fitting, and asset review.
No model downloaded or fitting benchmark run. [@flameProject]

## Superseded experiments (retained rationale)

1. Nose-pole control: construct a surface whose anterior pole is the nose,
   fit remaining guide points, and compare whether it over-tapers the face.
   A mere change of parameterization leaves the surface unchanged; moving its
   geometric pole is a change to the shape family and constraints.
2. Cranial volume plus jaw and local nose wedge: Loomis-inspired separation
   with a smooth superquadric-like cranial mass, independently adjustable jaw
   taper/chin, and a small nose projection. Draw selected surface curves only.
3. A small planar cage inspired by Asaro, interpolated or smoothed while
   maintaining ear, eye, crown, chin and nose constraints.

The user rejected the nose/jaw/Asaro direction in favor of a single generic volume.
Sparse visible points cannot uniquely determine posterior skull shape, cheek
contours or jaw angles. Keep those explicit authored proportions. Fit shape in
the skull-local frame and move it with the existing skull pose; do not refit
all unconstrained proportions on every live frame. Keep detector keypoints,
mapped anatomical landmarks, and authored inferred landmarks distinct.

The earlier demo's ear-eye ring is tilted because eyes sit above ears in the
authored geometry. It is not a horizontal anatomical transverse section. The
earlier loft also is not a general superquadric: it uses a superellipse for
one section and custom interpolation to crown and chin. It was not derived
from the references above.

## Current prototype: cranial superellipsoid

The updated Three.js viewer uses a signed-power superellipsoid, inspired by
Loomis's cranial mass and construction guides. It does not reproduce a full
Loomis head, make literal planar cuts, or model a nose, jaw or facial features.
Superquadric exponents control vertical and horizontal roundness.
[@barr1981Superquadrics; @mcgillSuperquadrics]

Our parameterization and fit below are implementation derivations, not equations
claimed to be a head-fitting method from those sources. In skull-local axes,
define S(v,e) = sign(v)|v|^e. With latitude u and longitude v:

```
x = a cos(u)^e1 S(cos(v), e2)
y = cy + b_side cos(u)^e1 S(sin(v), e2)
z = cz + c S(sin(u), e1)
```

Here b_side is b anteriorly and b times the back-depth multiplier posteriorly.
This is a piecewise depth-scaled superellipsoid, continuous at the dividing
plane. Three principal rings and two x=+/-0.82a section curves are drawn from
the same surface. The side guides suggest Loomis's side planes without adding
anatomical detail or claiming hard planar truncation.

For the symmetric authored demo landmarks, the fit is closed-form:

```
cz = (crown_z + chin_z)/2
c = (crown_z - chin_z)/2
cy = mean(ear_y)
R(z) = 1 - abs((z-cz)/c)^(2/e1)
a = half_ear_span / R(ear_z)^(e1/2)
b = (eye_y-cy) /
    (R(eye_z)^(e1/e2) - abs(half_eye_span/a)^(2/e2))^(e2/2)
```

Crown, both ears and both eyes lie on the implicit surface. Chin contributes
only its height, so its forward position does not create a pointed jaw; the chin
marker is not promised to lie on the volume. Nose does not constrain the shape.
The fit assumes bilateral symmetry, a valid positive denominator, and an already
resolved skull frame. It is not an implemented noisy or missing-keypoint fitter.
Production integration must use Tracker's mapped observations and Forge's skull
pose, with explicit handling of insufficient evidence, rather than passing raw
detector layouts into a hard-coded UI solver.

Exponents and posterior depth remain design controls, not measurements inferred
from these points. The selected production settings are e1=1.0, e2=0.85, posterior
multiplier=0.8. The original exploratory viewer exposed e1=[0.65,1.25],
e2=[0.55,1.25], posterior=[0.8,1.6]; those ranges are not fitted parameters.

Validation: real Edge loaded the viewer, exercised all eight control-range
corners, and checked finite mesh coordinates. Maximum implicit residual at the
five constrained landmarks was 1.34e-15. Solid and ring screenshots inspected;
360px layout had no horizontal overflow. This validates authored demo geometry,
not live reconstruction, individual anatomical accuracy, or production speed.

## Accepted non-head choices

Pelvis rim plus hip socket struts; SC anterior offset 70%; xiphoid bridge plus
shoulder-to-chest-center links. These are now declared in the human definition.

## Production contract

`core/skeleton/presentation.py` compiles the authored skull-local geometry once
per `SkeletonDefinition`. `SkeletonSnapshot` preserves the declaration; old
snapshots without it restore with no presentation overlay. No detector fitting
or rendering code is added to reconstruction. The five closed contours have
64 edges each. The selected transverse section is the implicit surface's
intersection with the plane through both ears and the nose: its lateral axis
joins the ears and its forward axis points toward the nose after removing the
lateral component. The nose changes this section's orientation, not the volume.
An optional 32-by-16 translucent shell and two curved side caps add 2,048
triangles in two static batches. All vertices lie on the same implicit surface.
Old recipes without these optional declarations retain their original contours
and have no surfaces. The local axes are +X right, +Y anterior, +Z up;
positions use the segment's authored reference units. Consumers apply the
reconstructed world rotation and origin and scale local positions by the fitted
segment length divided by the authored segment length, without a second rest
rotation. Missing pose or measured size must not produce an invented default.

Numerical tests verify the ears, eyes, crown and all sampled sections against the
implicit surface, closed/finite contours, a nonidentity hydrated transform, and
snapshot round trips. Chin constrains vertical extent only; its anterior offset
and the nose do not define facial anatomy. This is a simplified display volume,
not a personalized anatomical surface.

The bilateral SC local anterior coordinate changes from 0.048593 to 0.0340151.
The notch now has the same depth; xiphoid anterior depth is separately reduced
to 60% of its original value. With the existing
rest-pose hierarchy, clavicle/acromion and downstream arm positions consequently
move posteriorly by 0.0145779 reference units. Tests make that propagation
explicit; no compensating arm edits or solver retuning are applied. Pelvis and
thorax line declarations and excluded glyph/origin edges affect presentation
only; the scientific hierarchy is retained.
