#pragma once
#include "skellyforge/rigid_fit.h"
#include "skellyforge/chain_solver_constants.h"
#include <optional>
namespace skellyforge {
using ChainQuaternions = std::vector<std::array<double,4>>;
// One total axial length per frame; each selected segment gets an exact share.
struct SharedAxialLength {
  std::optional<double> minimum_total, maximum_total; // mm; optional experiment bounds
  std::array<int,3> segments{-1,-1,-1};
  std::array<double,3> ratios{1.,1.,1.};
};
struct LengthEqualityPrior {
  int segment_a=-1, segment_b=-1;
  double scale=kDefaultLengthEqualityScale;
};
struct LengthProportionPrior {
  std::array<int,3> segments{-1,-1,-1};
  std::array<double,3> ratios{1.,1.,1.}; // normalized internally; positive relative lengths
  double scale=kDefaultLengthProportionScale; // mm, deviations from shares of total length
};
// Optional per-frame geometric preference, not another keypoint measurement.
// Each active frame stores origin, unit lateral axis, unit anterior axis.
struct LandmarkLinePrior {
  int segment = -1;
  Vec3 local_point{};
  std::vector<std::optional<std::array<Vec3,3>>> frames;
  double distance_scale = 1., anterior_scale = 1.;
};
// World-space lateral axis evidence, compared in the segment transverse plane.
struct SegmentAxisPrior {
  int segment=-1;
  Vec3 local_lateral{1.,0.,0.}, local_anterior{0.,1.,0.};
  std::vector<std::optional<Vec3>> frames;
  double scale=1.; // chord residual scale; not an anatomical limit
};
struct RelativeTwistPrior {
  int parent=-1,child=-1;
  std::array<double,4> reference{1.,0.,0.,0.};
  Vec3 axis{0.,0.,1.}; // twist axis in rest-aligned child coordinates
  double scale=1.;
};
// Prefer a fitted landmark on the positive side of a fixed world-space plane.
struct LandmarkHalfSpacePrior {
  int segment=-1;
  Vec3 local_point{};
  std::vector<std::optional<std::array<Vec3,2>>> frames; // origin, unit normal
  double scale=1.; // mm
};
// Geometric position preferences, separate from independent keypoint evidence.
struct LandmarkPositionPrior {
  int segment=-1;
  Vec3 local_point{};
  std::vector<Vec3> targets;
  double scale=1.; // mm; soft residual, not a hard positional bound
};
struct ChainSolveOptions {
  double landmark_huber_scale_mm=0.; // zero disables robust loss; threshold in physical mm
  std::vector<LandmarkPositionPrior> landmark_position_priors;
  std::vector<int> fixed_length_segments; // independent axial blocks held at reference length
  std::vector<std::vector<double>> initial_lengths;
  std::vector<std::vector<Vec3>> initial_linkage_displacements;
  std::vector<double> frame_weights; // full-recording quadrature weights sliced with a window
  int fixed_prefix_frames=0;
  int max_iterations=kChainMaximumIterations;
  double function_tolerance=kChainFunctionTolerance;
  bool evaluate_only=false; // score supplied state without optimizing
};
struct ChainSequenceFit {
  double position_prior_cost=0.;
  bool usable=false;
  int iterations=0;
  std::string full_report;

  double twist_prior_cost=0.;
  double half_space_cost=0.;
  double axis_prior_cost=0.;
  double length_proportion_cost=0.;
  double length_equality_cost=0.;
  std::vector<std::vector<Vec3>> linkage_displacements;
  double linkage_prior_cost=0., linkage_acceleration_cost=0.;
  double line_prior_cost=0.;
  std::vector<ChainQuaternions> quaternions, initial_quaternions;
  std::vector<std::vector<Vec3>> translations, initial_translations;
  std::vector<Vec3> roots;
  std::vector<double> displacements;
  std::vector<std::vector<double>> lengths;
  double length_prior_cost=0,length_acceleration_cost=0;
  double relative_pose_cost=0;
  double displacement_prior_cost=0, displacement_acceleration_cost=0;
  std::vector<double> costs, angular_acceleration_costs;
  double landmark_cost, root_acceleration_cost, seconds;
  bool converged;
  std::string report;
  int parameter_blocks, residual_blocks;
};
ChainSequenceFit fit_chain_sequence(const std::vector<std::vector<Vec3>>& local,
  const std::vector<std::vector<std::vector<Vec3>>>& observed,
  const std::vector<Vec3>& parent_attachments, const std::vector<Vec3>& child_attachments,
  const std::vector<double>& times, double position_scale,
  double linear_acceleration_scale, double angular_acceleration_scale,
  bool allow_displacement=false, double displacement_scale=kDefaultDisplacementScale,
  double displacement_acceleration_scale=kDefaultDisplacementAccelerationScale, double displacement_bound=kDefaultDisplacementBound,
  const std::vector<int>& parent_indices={0,1},
  const std::vector<ChainQuaternions>& initial_quaternions={},
  const std::vector<Vec3>& initial_roots={},
  const ChainQuaternions& rest_relative_quaternions={}, double rest_pose_scale=kDefaultRestPoseScale, const std::vector<double>& axial_reference_lengths={},
  // The base fraction governs shortening and, unless overridden, lengthening.
  double length_prior_fraction=kDefaultLengthPriorFraction, double length_acceleration_scale=kDefaultLengthAccelerationScale,
  const std::vector<std::vector<std::vector<int>>>& observation_indices={},
  std::optional<double> lengthening_prior_fraction=std::nullopt, bool free_axial_lengths=false, const std::optional<LandmarkLinePrior>& landmark_line_prior=std::nullopt,
  const std::vector<int>& relaxed_linkage_children={}, double linkage_scale=kDefaultLinkageScale, double linkage_acceleration_scale=kDefaultLinkageAccelerationScale,
  const std::optional<LengthEqualityPrior>& length_equality_prior=std::nullopt,
  const std::optional<LengthProportionPrior>& length_proportion_prior=std::nullopt,
  const ChainSolveOptions& solve_options=ChainSolveOptions{}, bool free_length_rest_prior=false,
  const std::optional<SharedAxialLength>& shared_axial_length=std::nullopt,
  const std::optional<SegmentAxisPrior>& segment_axis_prior=std::nullopt,
  const std::vector<RelativeTwistPrior>& relative_twist_priors={},
  const std::optional<LandmarkHalfSpacePrior>& landmark_half_space_prior=std::nullopt);
}
