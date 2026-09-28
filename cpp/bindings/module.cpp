#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "skellyforge/rigid_fit.h"
#include "skellyforge/rigid_sequence.h"
#include "skellyforge/linked_rigid_fit.h"
#include "skellyforge/linked_sequence.h"
#include "skellyforge/chain_sequence.h"
#include "skellyforge/smoke.h"
namespace py = pybind11;
PYBIND11_MODULE(_native, m) {
  m.attr("DEFAULT_LENGTH_PROPORTION_SCALE") = skellyforge::kDefaultLengthProportionScale;
  py::class_<skellyforge::SharedAxialLength>(m,"SharedAxialLength")
    .def(py::init<>())
    .def_readwrite("minimum_total",&skellyforge::SharedAxialLength::minimum_total)
    .def_readwrite("maximum_total",&skellyforge::SharedAxialLength::maximum_total)
    .def_readwrite("segments",&skellyforge::SharedAxialLength::segments)
    .def_readwrite("ratios",&skellyforge::SharedAxialLength::ratios);
  py::class_<skellyforge::LengthProportionPrior>(m,"LengthProportionPrior")
    .def(py::init<>())
    .def_readwrite("segments",&skellyforge::LengthProportionPrior::segments)
    .def_readwrite("ratios",&skellyforge::LengthProportionPrior::ratios)
    .def_readwrite("scale",&skellyforge::LengthProportionPrior::scale);
  m.attr("DEFAULT_LENGTH_EQUALITY_SCALE") = skellyforge::kDefaultLengthEqualityScale;
  py::class_<skellyforge::LengthEqualityPrior>(m,"LengthEqualityPrior")
    .def(py::init<>())
    .def_readwrite("segment_a",&skellyforge::LengthEqualityPrior::segment_a)
    .def_readwrite("segment_b",&skellyforge::LengthEqualityPrior::segment_b)
    .def_readwrite("scale",&skellyforge::LengthEqualityPrior::scale);
  m.attr("DEFAULT_LINKAGE_SCALE") = skellyforge::kDefaultLinkageScale;
  m.attr("DEFAULT_LINKAGE_ACCELERATION_SCALE") = skellyforge::kDefaultLinkageAccelerationScale;
  m.attr("DEFAULT_LENGTH_PRIOR_FRACTION") = skellyforge::kDefaultLengthPriorFraction;
  m.attr("DEFAULT_LENGTH_ACCELERATION_SCALE") = skellyforge::kDefaultLengthAccelerationScale;
  m.attr("AXIAL_LENGTH_MINIMUM_FRACTION") = skellyforge::kAxialLengthMinimumFraction;
  m.attr("AXIAL_LENGTH_MAXIMUM_FRACTION") = skellyforge::kAxialLengthMaximumFraction;
  py::class_<skellyforge::LandmarkLinePrior>(m,"LandmarkLinePrior")
    .def(py::init<>())
    .def_readwrite("segment",&skellyforge::LandmarkLinePrior::segment)
    .def_readwrite("local_point",&skellyforge::LandmarkLinePrior::local_point)
    .def_readwrite("frames",&skellyforge::LandmarkLinePrior::frames)
    .def_readwrite("distance_scale",&skellyforge::LandmarkLinePrior::distance_scale)
    .def_readwrite("anterior_scale",&skellyforge::LandmarkLinePrior::anterior_scale);
  py::class_<skellyforge::SegmentAxisPrior>(m,"SegmentAxisPrior")
    .def(py::init<>())
    .def_readwrite("segment",&skellyforge::SegmentAxisPrior::segment)
    .def_readwrite("local_lateral",&skellyforge::SegmentAxisPrior::local_lateral)
    .def_readwrite("local_anterior",&skellyforge::SegmentAxisPrior::local_anterior)
    .def_readwrite("frames",&skellyforge::SegmentAxisPrior::frames)
    .def_readwrite("scale",&skellyforge::SegmentAxisPrior::scale);
  py::class_<skellyforge::LandmarkHalfSpacePrior>(m,"LandmarkHalfSpacePrior")
    .def(py::init<>())
    .def_readwrite("segment",&skellyforge::LandmarkHalfSpacePrior::segment)
    .def_readwrite("local_point",&skellyforge::LandmarkHalfSpacePrior::local_point)
    .def_readwrite("frames",&skellyforge::LandmarkHalfSpacePrior::frames)
    .def_readwrite("scale",&skellyforge::LandmarkHalfSpacePrior::scale);
  py::class_<skellyforge::RelativeTwistPrior>(m,"RelativeTwistPrior")
    .def(py::init<>())
    .def_readwrite("parent",&skellyforge::RelativeTwistPrior::parent)
    .def_readwrite("child",&skellyforge::RelativeTwistPrior::child)
    .def_readwrite("reference",&skellyforge::RelativeTwistPrior::reference)
    .def_readwrite("axis",&skellyforge::RelativeTwistPrior::axis)
    .def_readwrite("scale",&skellyforge::RelativeTwistPrior::scale);
  py::class_<skellyforge::LandmarkPositionPrior>(m,"LandmarkPositionPrior")
    .def(py::init<>())
    .def_readwrite("segment",&skellyforge::LandmarkPositionPrior::segment)
    .def_readwrite("local_point",&skellyforge::LandmarkPositionPrior::local_point)
    .def_readwrite("targets",&skellyforge::LandmarkPositionPrior::targets)
    .def_readwrite("scale",&skellyforge::LandmarkPositionPrior::scale);
  py::class_<skellyforge::ParameterInspection>(m,"ParameterInspection")
    .def_readonly("id",&skellyforge::ParameterInspection::id)
    .def_readonly("frame",&skellyforge::ParameterInspection::frame)
    .def_readonly("segment",&skellyforge::ParameterInspection::segment)
    .def_readonly("quantity",&skellyforge::ParameterInspection::quantity)
    .def_readonly("ambient_size",&skellyforge::ParameterInspection::ambient_size)
    .def_readonly("tangent_size",&skellyforge::ParameterInspection::tangent_size)
    .def_readonly("constant",&skellyforge::ParameterInspection::constant)
    .def_readonly("manifold_type",&skellyforge::ParameterInspection::manifold_type)
    .def_readonly("values",&skellyforge::ParameterInspection::values)
    .def_readonly("lower_bounds",&skellyforge::ParameterInspection::lower_bounds)
    .def_readonly("upper_bounds",&skellyforge::ParameterInspection::upper_bounds);
  py::class_<skellyforge::ResidualInspection>(m,"ResidualInspection")
    .def_readonly("id",&skellyforge::ResidualInspection::id)
    .def_readonly("purpose",&skellyforge::ResidualInspection::purpose)
    .def_readonly("cost_type",&skellyforge::ResidualInspection::cost_type)
    .def_readonly("loss_type",&skellyforge::ResidualInspection::loss_type)
    .def_readonly("parameter_ids",&skellyforge::ResidualInspection::parameter_ids)
    .def_readonly("values",&skellyforge::ResidualInspection::values)
    .def_readonly("cost",&skellyforge::ResidualInspection::cost);
  py::class_<skellyforge::ProblemInspection>(m,"ProblemInspection")
    .def_readonly("parameters",&skellyforge::ProblemInspection::parameters)
    .def_readonly("residuals",&skellyforge::ProblemInspection::residuals);
  py::class_<skellyforge::ChainSolveOptions>(m,"ChainSolveOptions")
    .def(py::init<>())
    .def_readwrite("initial_lengths",&skellyforge::ChainSolveOptions::initial_lengths)
    .def_readwrite("inspect_problem",&skellyforge::ChainSolveOptions::inspect_problem)
    .def_readwrite("landmark_position_priors",&skellyforge::ChainSolveOptions::landmark_position_priors)
    .def_readwrite("landmark_huber_scale_mm",&skellyforge::ChainSolveOptions::landmark_huber_scale_mm)
    .def_readwrite("fixed_length_segments",&skellyforge::ChainSolveOptions::fixed_length_segments)
    .def_readwrite("initial_linkage_displacements",&skellyforge::ChainSolveOptions::initial_linkage_displacements)
    .def_readwrite("frame_weights",&skellyforge::ChainSolveOptions::frame_weights)
    .def_readwrite("fixed_prefix_frames",&skellyforge::ChainSolveOptions::fixed_prefix_frames)
    .def_readwrite("function_tolerance",&skellyforge::ChainSolveOptions::function_tolerance)
    .def_readwrite("max_iterations",&skellyforge::ChainSolveOptions::max_iterations)
    .def_readwrite("evaluate_only",&skellyforge::ChainSolveOptions::evaluate_only);
  py::class_<skellyforge::ChainSequenceFit>(m,"ChainSequenceFit")
    .def_readonly("problem_initial",&skellyforge::ChainSequenceFit::problem_initial)
    .def_readonly("problem_final",&skellyforge::ChainSequenceFit::problem_final)
    .def_readonly("usable",&skellyforge::ChainSequenceFit::usable)
    .def_readonly("iterations",&skellyforge::ChainSequenceFit::iterations)
    .def_readonly("full_report",&skellyforge::ChainSequenceFit::full_report)
    .def_readonly("half_space_cost",&skellyforge::ChainSequenceFit::half_space_cost)
    .def_readonly("twist_prior_cost",&skellyforge::ChainSequenceFit::twist_prior_cost)
    .def_readonly("axis_prior_cost",&skellyforge::ChainSequenceFit::axis_prior_cost)
    .def_readonly("length_proportion_cost",&skellyforge::ChainSequenceFit::length_proportion_cost)
    .def_readonly("length_equality_cost",&skellyforge::ChainSequenceFit::length_equality_cost)
    .def_readonly("linkage_displacements",&skellyforge::ChainSequenceFit::linkage_displacements)
    .def_readonly("linkage_prior_cost",&skellyforge::ChainSequenceFit::linkage_prior_cost)
    .def_readonly("linkage_acceleration_cost",&skellyforge::ChainSequenceFit::linkage_acceleration_cost)
    .def_readonly("line_prior_cost",&skellyforge::ChainSequenceFit::line_prior_cost)
    .def_readonly("lengths",&skellyforge::ChainSequenceFit::lengths)
    .def_readonly("length_prior_cost",&skellyforge::ChainSequenceFit::length_prior_cost)
    .def_readonly("length_acceleration_cost",&skellyforge::ChainSequenceFit::length_acceleration_cost)
    .def_readonly("relative_pose_cost",&skellyforge::ChainSequenceFit::relative_pose_cost)
    .def_readonly("displacements",&skellyforge::ChainSequenceFit::displacements)
    .def_readonly("displacement_prior_cost",&skellyforge::ChainSequenceFit::displacement_prior_cost)
    .def_readonly("displacement_acceleration_cost",&skellyforge::ChainSequenceFit::displacement_acceleration_cost)
    .def_readonly("quaternions",&skellyforge::ChainSequenceFit::quaternions)
    .def_readonly("initial_quaternions",&skellyforge::ChainSequenceFit::initial_quaternions)
    .def_readonly("translations",&skellyforge::ChainSequenceFit::translations)
    .def_readonly("initial_translations",&skellyforge::ChainSequenceFit::initial_translations)
    .def_readonly("roots",&skellyforge::ChainSequenceFit::roots)
    .def_readonly("costs",&skellyforge::ChainSequenceFit::costs)
    .def_readonly("angular_acceleration_costs",&skellyforge::ChainSequenceFit::angular_acceleration_costs)
    .def_readonly("landmark_cost",&skellyforge::ChainSequenceFit::landmark_cost)
    .def_readonly("position_prior_cost",&skellyforge::ChainSequenceFit::position_prior_cost)
    .def_readonly("root_acceleration_cost",&skellyforge::ChainSequenceFit::root_acceleration_cost)
    .def_readonly("seconds",&skellyforge::ChainSequenceFit::seconds)
    .def_readonly("converged",&skellyforge::ChainSequenceFit::converged)
    .def_readonly("report",&skellyforge::ChainSequenceFit::report)
    .def_readonly("parameter_blocks",&skellyforge::ChainSequenceFit::parameter_blocks)
    .def_readonly("residual_blocks",&skellyforge::ChainSequenceFit::residual_blocks)
    ;
  m.def("fit_chain_sequence",&skellyforge::fit_chain_sequence,py::kw_only(),
    py::arg("local"),py::arg("observed"),py::arg("parent_attachments"),py::arg("child_attachments"),py::arg("times"),
    py::arg("position_scale"),py::arg("linear_acceleration_scale"),py::arg("angular_acceleration_scale"),py::arg("allow_displacement")=false,py::arg("displacement_scale")=skellyforge::kDefaultDisplacementScale,py::arg("displacement_acceleration_scale")=skellyforge::kDefaultDisplacementAccelerationScale,py::arg("displacement_bound")=skellyforge::kDefaultDisplacementBound,py::arg("parent_indices")=std::vector<int>{0,1},py::arg("initial_quaternions")=std::vector<skellyforge::ChainQuaternions>{},py::arg("initial_roots")=std::vector<skellyforge::Vec3>{},py::arg("rest_relative_quaternions")=skellyforge::ChainQuaternions{},py::arg("rest_pose_scale")=skellyforge::kDefaultRestPoseScale,py::arg("axial_reference_lengths")=std::vector<double>{},py::arg("length_prior_fraction")=skellyforge::kDefaultLengthPriorFraction,py::arg("length_acceleration_scale")=skellyforge::kDefaultLengthAccelerationScale,py::arg("observation_indices")=std::vector<std::vector<std::vector<int>>>{},py::arg("lengthening_prior_fraction")=py::none(),py::arg("free_axial_lengths")=false,py::arg("landmark_line_prior")=py::none(),py::arg("relaxed_linkage_children")=std::vector<int>{},py::arg("linkage_scale")=skellyforge::kDefaultLinkageScale,py::arg("linkage_acceleration_scale")=skellyforge::kDefaultLinkageAccelerationScale,py::arg("length_equality_prior")=py::none(),py::arg("length_proportion_prior")=py::none(),py::arg("solve_options")=skellyforge::ChainSolveOptions{},py::arg("free_length_rest_prior")=false,py::arg("shared_axial_length")=py::none(),py::arg("segment_axis_prior")=py::none(),py::arg("relative_twist_priors")=std::vector<skellyforge::RelativeTwistPrior>{},py::arg("landmark_half_space_prior")=py::none(),py::call_guard<py::gil_scoped_release>());

  py::class_<skellyforge::LinkedSequenceFit>(m,"LinkedSequenceFit")
    .def_readonly("quaternions",&skellyforge::LinkedSequenceFit::quaternions)
    .def_readonly("translations",&skellyforge::LinkedSequenceFit::translations)
    .def_readonly("joints",&skellyforge::LinkedSequenceFit::joints)
    .def_readonly("initial_quaternions",&skellyforge::LinkedSequenceFit::initial_quaternions)
    .def_readonly("initial_translations",&skellyforge::LinkedSequenceFit::initial_translations)
    .def_readonly("initial_joints",&skellyforge::LinkedSequenceFit::initial_joints)
    .def_readonly("costs",&skellyforge::LinkedSequenceFit::costs)
    .def_readonly("landmark_cost",&skellyforge::LinkedSequenceFit::landmark_cost)
    .def_readonly("joint_acceleration_cost",&skellyforge::LinkedSequenceFit::joint_acceleration_cost)
    .def_readonly("parent_acceleration_cost",&skellyforge::LinkedSequenceFit::parent_acceleration_cost)
    .def_readonly("child_acceleration_cost",&skellyforge::LinkedSequenceFit::child_acceleration_cost)
    .def_readonly("converged",&skellyforge::LinkedSequenceFit::converged)
    .def_readonly("report",&skellyforge::LinkedSequenceFit::report)
    .def_readonly("seconds",&skellyforge::LinkedSequenceFit::seconds)
    .def_readonly("parameter_blocks",&skellyforge::LinkedSequenceFit::parameter_blocks)
    .def_readonly("residual_blocks",&skellyforge::LinkedSequenceFit::residual_blocks);
  m.def("fit_linked_sequence",&skellyforge::fit_linked_sequence,py::kw_only(),
    py::arg("local_a"),py::arg("observed_a"),py::arg("attachment_a"),
    py::arg("local_b"),py::arg("observed_b"),py::arg("attachment_b"),py::arg("times"),
    py::arg("position_scale"),py::arg("linear_acceleration_scale"),py::arg("angular_acceleration_scale"),
    py::call_guard<py::gil_scoped_release>());
  py::class_<skellyforge::LinkedRigidFit>(m,"LinkedRigidFit")
    .def_readonly("quaternions",&skellyforge::LinkedRigidFit::quaternions)
    .def_readonly("translations",&skellyforge::LinkedRigidFit::translations)
    .def_readonly("joint",&skellyforge::LinkedRigidFit::joint)
    .def_readonly("costs",&skellyforge::LinkedRigidFit::costs)
    .def_readonly("converged",&skellyforge::LinkedRigidFit::converged)
    .def_readonly("report",&skellyforge::LinkedRigidFit::report)
    .def_readonly("seconds",&skellyforge::LinkedRigidFit::seconds);
  m.def("fit_linked_rigid",&skellyforge::fit_linked_rigid,py::kw_only(),
    py::arg("local_a"),py::arg("observed_a"),py::arg("attachment_a"),
    py::arg("local_b"),py::arg("observed_b"),py::arg("attachment_b"),
    py::call_guard<py::gil_scoped_release>());
  py::class_<skellyforge::RigidSequenceFit>(m,"RigidSequenceFit")
    .def_readonly("quaternions",&skellyforge::RigidSequenceFit::quaternions)
    .def_readonly("translations",&skellyforge::RigidSequenceFit::translations)
    .def_readonly("costs",&skellyforge::RigidSequenceFit::costs)
    .def_readonly("landmark_cost",&skellyforge::RigidSequenceFit::landmark_cost)
    .def_readonly("translation_cost",&skellyforge::RigidSequenceFit::translation_cost)
    .def_readonly("rotation_cost",&skellyforge::RigidSequenceFit::rotation_cost)
    .def_readonly("seconds",&skellyforge::RigidSequenceFit::seconds)
    .def_readonly("converged",&skellyforge::RigidSequenceFit::converged)
    .def_readonly("report",&skellyforge::RigidSequenceFit::report);
  m.def("fit_rigid_sequence",&skellyforge::fit_rigid_sequence,py::kw_only(),
    py::arg("local"),py::arg("observed"),py::arg("times"),py::arg("position_scale"),
    py::arg("linear_motion_scale"),py::arg("angular_motion_scale"),py::arg("temporal_model")="velocity",py::call_guard<py::gil_scoped_release>());
  py::class_<skellyforge::RigidFit>(m,"RigidFit")
    .def_readonly("quaternion",&skellyforge::RigidFit::quaternion)
    .def_readonly("translation",&skellyforge::RigidFit::translation)
    .def_readonly("costs",&skellyforge::RigidFit::costs)
    .def_readonly("converged",&skellyforge::RigidFit::converged)
    .def_readonly("report",&skellyforge::RigidFit::report)
    .def_readonly("seconds",&skellyforge::RigidFit::seconds);
  m.def("fit_rigid",&skellyforge::fit_rigid,py::kw_only(),py::arg("local"),py::arg("observed"),
    py::arg("quaternion"),py::arg("translation"),py::arg("allow_underconstrained")=false,py::call_guard<py::gil_scoped_release>());
  m.doc() = "Ceres infrastructure and rigid-object experiments; not a skeleton solver.";
  m.attr("ceres_version") = skellyforge::ceres_version();
  py::class_<skellyforge::SmokeResult>(m, "SmokeResult")
      .def_readonly("value", &skellyforge::SmokeResult::value)
      .def_readonly("final_cost", &skellyforge::SmokeResult::final_cost)
      .def_readonly("converged", &skellyforge::SmokeResult::converged);
  m.def("solve_scalar", &skellyforge::solve_scalar, py::kw_only(),
        py::arg("initial"), py::arg("target"), py::call_guard<py::gil_scoped_release>());
}
