from skellymodels.models.aspect import Aspect, TrajectoryNames
from skellymodels.models.trajectory import Trajectory

from skellymodels.biomechanics.calculations.calculate_center_of_mass import calculate_center_of_mass
from skellymodels.biomechanics.calculations.enforce_rigid_bones import enforce_rigid_bones
from skellymodels.biomechanics.models.anatomical_calculation import AnatomicalCalculation, CalculationResult


class CenterOfMassCalculation(AnatomicalCalculation):
    """
    Calculates segment-level and total-body center of mass trajectories.

    Requires the aspect's AnatomicalStructure to define:
    - segment_connections
    - center_of_mass_definitions

    Adds two new trajectories to the aspect:
    - 'total_body_com' (Trajectory with a single virtual marker)
    - 'segment_com' (Trajectory with one marker per segment)
    """
    def calculate(self, aspect:Aspect) -> CalculationResult:
        if not aspect.anatomical_structure.center_of_mass_definitions:
            return CalculationResult(
                success = False,
                data = {},
                messages=[f'No COM definitions for aspect: {aspect.name}, skipping COM calculation']
            )

        trajectory = aspect.xyz #NOTE: maybe put this in a try/except loop where the except also returns a CalcResult? with success=False

        total_body_com, segment_com = calculate_center_of_mass(
            segment_positions=trajectory.segment_data(aspect.anatomical_structure.segment_connections),
            center_of_mass_definitions=aspect.anatomical_structure.center_of_mass_definitions,
            num_frames=trajectory.num_frames
        )

        return CalculationResult(
            success = True,
            data = {
                TrajectoryNames.TOTAL_BODY_COM.value: total_body_com,
                TrajectoryNames.SEGMENT_COM.value: segment_com
            },
            messages=[f'Successfully calculated COM for aspect: {aspect.name}'] 
        )
    
    def store(self, aspect: Aspect, results: CalculationResult):
        if not results.success:
            return
        
        tb_com_trajectory = Trajectory(
            name=TrajectoryNames.TOTAL_BODY_COM.value,
            array=results.data[TrajectoryNames.TOTAL_BODY_COM.value],
            landmark_names =[TrajectoryNames.TOTAL_BODY_COM.value],
        )

        segment_com_trajectory = Trajectory(
            name=TrajectoryNames.SEGMENT_COM.value,
            array=results.data[TrajectoryNames.SEGMENT_COM.value],
            landmark_names=list(aspect.anatomical_structure.center_of_mass_definitions.keys()),
        )

        aspect.add_trajectory(
            {TrajectoryNames.TOTAL_BODY_COM.value: tb_com_trajectory,
            TrajectoryNames.SEGMENT_COM.value: segment_com_trajectory}
        )

class RigidBonesEnforcement(AnatomicalCalculation):
    """
    Enforces rigid bone constraints using a joint hierarchy.

    Requires the aspect's AnatomicalStructure to define:
    - joint_hierarchy

    Adds one new trajectory to the aspect:
    - 'rigid_3d_xyz' (rigidified marker trajectories)
    """
    def calculate(self, aspect:Aspect) -> CalculationResult:
        if not aspect.anatomical_structure.joint_hierarchy:
            return CalculationResult(
                success = False,
                data = {},
                messages = [f'No joint hierarchy defined for aspect: {aspect.name}, skipping rigid bones enforcement']
            )
        
        trajectory = aspect.xyz

        rigid_marker_data = enforce_rigid_bones(
            marker_trajectories=trajectory.as_dict,
            joint_hierarchy= aspect.anatomical_structure.joint_hierarchy
        )

        return CalculationResult(
            success=True,
            data = {TrajectoryNames.RIGID_XYZ.value: rigid_marker_data},
            messages= [f'Successfully enforced rigid bones for aspect: {aspect.name}']
        )

    def store(self, aspect:Aspect, results: CalculationResult):
        if not results.success:
            return
        
        aspect.add_trajectory(
            {TrajectoryNames.RIGID_XYZ.value : Trajectory(
            name=TrajectoryNames.RIGID_XYZ.value,
            array=results.data[TrajectoryNames.RIGID_XYZ.value],
            landmark_names=aspect.anatomical_structure.landmark_names,
            segment_connections=aspect.anatomical_structure.segment_connections
        )})
        

class CalculationPipeline:
    """
    A sequence of anatomical calculations to run on an aspect.

    Each task must be a subclass of `AnatomicalCalculation`, implementing
    both `calculate()` and `store()` methods.

    Parameters
    ----------
    tasks : list of AnatomicalCalculation
        Calculation classes to apply in order.
    """
    def __init__(self, tasks: list[AnatomicalCalculation]):
        self.tasks = tasks

    def run(self, aspect:Aspect):
        """Instantiate tasks and run calculate_and_store on the given aspect."""
        results_log = []
        for task_cls in self.tasks:
            task_instance:AnatomicalCalculation = task_cls()  
            results = task_instance.calculate_and_store(aspect)

            if results:
                results_log.extend(results.messages)

        return results_log



STANDARD_PIPELINE = CalculationPipeline([CenterOfMassCalculation, RigidBonesEnforcement])