"""Viewer data contract. Rendering consumes experiments, runs, frames and segments.

Experiment generators own all fitting and diagnostics. The UI only selects runs.
"""

import itertools
import hashlib
from pathlib import Path
from skellyforge import _native
from scripts.solver_linkage_experiment import linkage_experiment
from scripts.solver_linked_sequence_experiment import linked_sequence_experiment
from scripts.solver_chain_experiment import chain_experiment
from scripts.solver_displacement_experiment import displacement_experiment
from scripts.solver_tree_experiment import tree_experiment
from scripts.solver_torso_experiment import torso_catalog

EDGES = [[i, j] for i in range(8) for j in range(i + 1, 8) if i ^ j in [1, 2, 4]]


def control(id, label, values):
    return dict(id=id, label=label, values=values)


def body_definition(id, label, attachment=None):
    return dict(
        id=id,
        label=label,
        edges=EDGES,
        landmark_names=[f"{label} landmark {i}" for i in range(8)],
        attachment=attachment,
    )


def cube_frame(frame):
    return {
        **{k: frame[k] for k in ["costs", "converged", "seconds", "report"]},
        "time": frame.get("time", 0.0),
        "bodies": [frame],
        "diagnostics": {"Rotation error (degrees)": frame["rotation_error_degrees"]},
    }


def experiments(cases, sequences):
    controls = [
        control("noise", "Landmark noise / mm per axis", [0, 1, 5, 15]),
        control("missing", "Missing landmarks", [0, 3]),
        control("outlier", "Landmark 0 outlier / mm +X", [0, 80]),
        control("seed", "Noise seed", [7, 42]),
    ]
    result = []
    for motion, label in [
        ("static", "01 · Cube — static"),
        ("stationary", "02 · Cube — stationary sequence"),
        ("moving", "03 · Cube — moving sequence"),
    ]:
        runs = []
        for case in (
            cases
            if motion == "static"
            else [s for s in sequences if s["motion"] == motion]
        ):
            methods = {}
            source = (
                {"independent": dict(frames=[case], settings=case["settings"])}
                if motion == "static"
                else case["modes"]
            )
            for name, method in source.items():
                methods[name] = dict(
                    frames=[cube_frame(f) for f in method["frames"]],
                    settings=method["settings"],
                    summary={
                        k: method[k]
                        for k in [
                            "mean_corner_error_mm",
                            "translation_x_range_mm",
                            "known_translation_x_range_mm",
                        ]
                        if k in method
                    },
                    objective=(
                        "Selected-frame squared landmark error / mm²."
                        if name == "independent"
                        else "Whole-sequence objective: time-weighted scaled landmark and motion residuals. Same history at every frame."
                    ),
                )
            runs.append(
                dict(
                    parameters={c["id"]: case[c["id"]] for c in controls},
                    times=case.get("times", [0.0]),
                    methods=methods,
                )
            )
        result.append(
            dict(
                id=motion,
                label=label,
                description="One rigid segment with eight landmarks, 200 mm sides. Compare fitted poses against known synthetic poses. Known poses are never supplied to the fit.",
                controls=controls,
                bodies=[body_definition("cube", "Cube")],
                runs=runs,
                methods=(
                    [
                        dict(id=k, label=v)
                        for k, v in [
                            ("independent", "Independent frames"),
                            ("sequence", "Velocity preference"),
                            ("strong", "Strong velocity preference"),
                            ("acceleration", "Acceleration preference"),
                        ]
                    ]
                    if motion != "static"
                    else [dict(id="independent", label="Independent fit")]
                ),
            )
        )
    result.append(
        dict(
            id="linkage",
            label="04 · Two segments — shared point",
            description="Two 200 mm rigid segments. Their attachment points coincide exactly in the connected fit; both segments can rotate freely. Independent frames, no smoothing or joint-angle limits.",
            controls=[
                controls[0],
                controls[-1],
                control("child_points", "Child observed landmarks", [8, 4, 2, 1]),
            ],
            bodies=[
                body_definition("parent", "Parent segment", [0, 0, 100]),
                body_definition("child", "Child segment", [0, 0, -100]),
            ],
            methods=[
                dict(id="independent", label="Independent segments"),
                dict(id="connected", label="Connected at one point"),
            ],
            runs=[
                linkage_experiment(noise=n, seed=s, child_points=count)
                for n, s, count in itertools.product(
                    [0, 1, 5, 15], [7, 42], [8, 4, 2, 1]
                )
            ],
        )
    )
    result.append(
        dict(
            id="linked_sequence",
            label="05 - Linked segments / observation gap",
            description="One Ceres Problem for the full sequence: two quaternion parameter blocks and one shared joint-position block per frame, connected by acceleration residual blocks. Child landmark residual blocks are omitted during the gap.",
            controls=[
                control("noise", "Landmark noise / mm per axis", [0, 1, 15]),
                control("gap", "Missing child frames", [5, 0, 11]),
            ],
            bodies=[
                body_definition("parent", "Parent segment", [0, 0, 100]),
                body_definition("child", "Child segment", [0, 0, -100]),
            ],
            methods=[
                dict(id="per_frame", label="Per-frame Problems"),
                dict(id="temporal", label="One temporal Problem"),
            ],
            runs=[
                linked_sequence_experiment(noise=n, gap=g)
                for n, g in itertools.product([0, 1, 15], [5, 0, 11])
            ],
        )
    )
    chain_bodies = [
        body_definition("root", "Root segment"),
        body_definition("middle", "Middle segment"),
        body_definition("distal", "Distal segment"),
    ]
    chain_bodies[0]["attachments"] = [
        dict(position=[0, 0, 80], label="first linkage / parent")
    ]
    chain_bodies[1]["attachments"] = [
        dict(position=[0, 0, -80], label="first linkage / child"),
        dict(position=[0, 0, 80], label="second linkage / parent"),
    ]
    chain_bodies[2]["attachments"] = [
        dict(position=[0, 0, -80], label="second linkage / child")
    ]
    result.append(
        dict(
            id="chain",
            label="06 - Three-segment chain",
            description="Three world-quaternion parameter blocks and one root-position block per frame. ChainLandmarkResidual blocks reference all upstream quaternions. Middle observations are withheld during the gap; root and distal observations remain.",
            controls=[
                control("noise", "Landmark noise / mm per axis", [0, 1, 15]),
                control("gap", "Missing middle frames", [5, 0, 11]),
            ],
            bodies=chain_bodies,
            methods=[dict(id="temporal", label="One chain sequence Problem")],
            runs=[
                chain_experiment(noise=n, gap=g)
                for n, g in itertools.product([0, 1, 15], [5, 0, 11])
            ],
        )
    )
    result.append(
        dict(
            id="displacement",
            label="07 - Bounded linkage displacement",
            description="All segments stay rigid and observed. Compare a fixed second linkage with a scalar displacement parameter along the middle segment local Z axis. The displacement has bounds and prior/acceleration residual blocks; these are synthetic settings, not anatomical defaults.",
            controls=[
                control("noise", "Landmark noise / mm per axis", [0, 1, 15]),
                control("extension", "Known displacement peak / mm", [20, 0]),
            ],
            bodies=chain_bodies,
            displacement_link=dict(
                parent=1,
                child=2,
                parent_attachment=[0, 0, 80],
                child_attachment=[0, 0, -80],
            ),
            methods=[
                dict(id="fixed", label="Fixed linkage"),
                dict(id="displacement", label="Bounded displacement"),
            ],
            runs=[
                displacement_experiment(noise=n, extension=e)
                for n, e in itertools.product([0, 1, 15], [20, 0])
            ],
        )
    )
    displacement_gap = dict(result[-1])
    displacement_gap.update(
        id="displacement_gap",
        label="08 - Displacement / observation gap",
        description="Same observed root and distal landmarks in both Problems; middle landmark residual blocks are omitted in the shaded gap. Endpoint distance constrains axial displacement, while middle roll remains unobserved. Compare residual costs, displacement bias and quaternion trajectories; convergence is not proof of recovering missing motion.",
        controls=[
            control("noise", "Landmark noise / mm per axis", [1, 0, 15]),
            control("extension", "Known displacement peak / mm", [20, 0]),
            control("gap", "Missing middle frames", [5, 11]),
        ],
        runs=[
            displacement_experiment(noise=n, extension=e, gap=g)
            for n, e, g in itertools.product([1, 0, 15], [20, 0], [5, 11])
        ],
    )
    result.append(displacement_gap)
    branch_bodies=[body_definition("parent", "Parent"), body_definition("a", "Branch A"), body_definition("b", "Branch B")]
    branch_bodies[0]["attachments"]=[dict(position=[0,-45,80],label="branch A attachment"),dict(position=[0,45,80],label="branch B attachment")]
    for body in branch_bodies[1:]:
        body["attachments"]=[dict(position=[0,0,-80],label="parent attachment")]
    result.append(dict(
        id="branching", label="09 - One parent / two branches",
        description="Both branches share the parent quaternion and root position. Fixed attachment equations; no displacement. Branch A can lose observations; Branch B remains observed. Its landmark residuals never reference Branch A's quaternion.",
        controls=[control("noise","Landmark noise / mm per axis",[1,0,15]),control("gap","Missing Branch A frames",[5,0,11])],
        bodies=branch_bodies, methods=[dict(id="temporal",label="One branching sequence Problem")],
        runs=[chain_experiment(noise=n,gap=g,branching=True) for n,g in itertools.product([1,0,15],[5,0,11])],
    ))
    result.append(tree_catalog())
    result.append(torso_catalog())
    result.append(axial_catalog())
    for item in result:
        for run in item["runs"]:
            for name, method in run["methods"].items():
                method.setdefault(
                    "problem",
                    dict(
                        connected=item["id"] == "linkage" and name == "connected",
                        temporal=item["id"] in ["stationary", "moving"]
                        and name != "independent",
                        acceleration=name == "acceleration",
                    ),
                )

        item["metadata"] = dict(
            schema_version=1,
            units="mm",
            quaternion_order="wxyz",
            ceres_version=_native.ceres_version,
            native_sha256=hashlib.sha256(
                Path(_native.__file__).read_bytes()
            ).hexdigest(),
        )
    return result


def tree_catalog(include_runs=True):
    tree_bodies=[body_definition(str(i),label) for i,label in enumerate(["Root", "Spine link 1", "Spine link 2", "Left branch", "Right branch"])]
    for body in tree_bodies: body["attachments"]=[]
    for b,p in enumerate([0,1,2,2],1):
        position=[0,0,40] if b<3 else [0,-35 if b==3 else 35,40]
        tree_bodies[p]["attachments"].append(dict(position=position,label=f"to segment {b}"))
        tree_bodies[b]["attachments"].append(dict(position=[0,0,-40],label=f"from segment {p}"))
    return dict(id="tree",label="10 - Five-segment tree / infrastructure check",
        description="Dense synthetic observations on every segment. Tests the torso-shaped topology and shared ancestor blocks, not sparse hip/shoulder reconstruction or human anatomy.",
        controls=[control("noise","Landmark noise / mm per axis",[1,0,15])],bodies=tree_bodies,
        methods=[dict(id="temporal",label="One tree sequence Problem")],runs=[tree_experiment(n) for n in [1,0,15]] if include_runs else [])


def axial_catalog():
    experiment=tree_catalog(include_runs=False)
    runs=[]
    for noise in [1,0]:
        rigid=tree_experiment(noise,deforming=True)
        flexible=tree_experiment(noise,deforming=True,axial=True)
        for run in [rigid,flexible]:
            method=run['methods']['temporal']
            method['objective']='Known axial contraction; compare fixed vs variable local-Z geometry. Native residuals and attachment equations use the same deformation rule.'
            for frame in method['frames']:
                frame['observability']='Dense synthetic observations identify lengths. This validates deformation geometry, not sparse real-data anatomical accuracy.'
        runs.append(dict(parameters=dict(noise=noise),times=rigid['times'],length_series=True,length_reference_label='known',methods=dict(rigid=rigid['methods']['temporal'],flexible=flexible['methods']['temporal'])))
    experiment.update(id='axial',label='13 - Known axial contraction',description='Synthetic spine-shaped tree contracts from 80 to 56/64 mm. Rigid and flexible models receive identical observations. Local X/Y stay fixed; connections use deformed attachments.',controls=[control('noise','Landmark noise / mm per axis',[1,0])],methods=[dict(id='rigid',label='Rigid segments'),dict(id='flexible',label='Axial spine lengths')],runs=runs)
    return experiment
