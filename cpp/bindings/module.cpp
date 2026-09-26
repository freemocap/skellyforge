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
  py::class_<skellyforge::ChainSequenceFit>(m,"ChainSequenceFit")
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
    .def_readonly("root_acceleration_cost",&skellyforge::ChainSequenceFit::root_acceleration_cost)
    .def_readonly("seconds",&skellyforge::ChainSequenceFit::seconds)
    .def_readonly("converged",&skellyforge::ChainSequenceFit::converged)
    .def_readonly("report",&skellyforge::ChainSequenceFit::report)
    .def_readonly("parameter_blocks",&skellyforge::ChainSequenceFit::parameter_blocks)
    .def_readonly("residual_blocks",&skellyforge::ChainSequenceFit::residual_blocks)
    ;
  m.def("fit_chain_sequence",&skellyforge::fit_chain_sequence,py::kw_only(),
    py::arg("local"),py::arg("observed"),py::arg("parent_attachments"),py::arg("child_attachments"),py::arg("times"),
    py::arg("position_scale"),py::arg("linear_acceleration_scale"),py::arg("angular_acceleration_scale"),py::arg("allow_displacement")=false,py::arg("displacement_scale")=skellyforge::kDefaultDisplacementScale,py::arg("displacement_acceleration_scale")=skellyforge::kDefaultDisplacementAccelerationScale,py::arg("displacement_bound")=skellyforge::kDefaultDisplacementBound,py::arg("parent_indices")=std::vector<int>{0,1},py::arg("initial_quaternions")=std::vector<skellyforge::ChainQuaternions>{},py::arg("initial_roots")=std::vector<skellyforge::Vec3>{},py::arg("rest_relative_quaternions")=skellyforge::ChainQuaternions{},py::arg("rest_pose_scale")=skellyforge::kDefaultRestPoseScale,py::arg("axial_reference_lengths")=std::vector<double>{},py::arg("length_prior_fraction")=skellyforge::kDefaultLengthPriorFraction,py::arg("length_acceleration_scale")=skellyforge::kDefaultLengthAccelerationScale,py::arg("observation_indices")=std::vector<std::vector<std::vector<int>>>{},py::arg("lengthening_prior_fraction")=py::none(),py::arg("free_axial_lengths")=false,py::arg("landmark_line_prior")=py::none(),py::arg("relaxed_linkage_children")=std::vector<int>{},py::arg("linkage_scale")=skellyforge::kDefaultLinkageScale,py::arg("linkage_acceleration_scale")=skellyforge::kDefaultLinkageAccelerationScale,py::arg("length_equality_prior")=py::none(),py::call_guard<py::gil_scoped_release>());

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
