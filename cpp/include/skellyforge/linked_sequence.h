#pragma once
#include "skellyforge/linked_rigid_fit.h"
namespace skellyforge {
using QuaternionPair = std::array<std::array<double,4>,2>;
using TranslationPair = std::array<Vec3,2>;
struct LinkedSequenceFit {
  std::vector<QuaternionPair> quaternions, initial_quaternions;
  std::vector<TranslationPair> translations, initial_translations;
  std::vector<Vec3> joints, initial_joints;
  std::vector<double> costs;
  double landmark_cost, joint_acceleration_cost, parent_acceleration_cost, child_acceleration_cost;
  double seconds;
  bool converged;
  std::string report;
  int parameter_blocks, residual_blocks;
};
// Parent observations must support a rigid fit at every timestamp. Child frames
// contain all local landmarks or none. Empty child intervals must be bounded by
// observed frames. No reference poses are accepted by this interface.
LinkedSequenceFit fit_linked_sequence(const std::vector<Vec3>& local_a,
  const std::vector<std::vector<Vec3>>& observed_a, const Vec3& attachment_a,
  const std::vector<Vec3>& local_b, const std::vector<std::vector<Vec3>>& observed_b,
  const Vec3& attachment_b, const std::vector<double>& times,
  double position_scale, double linear_acceleration_scale, double angular_acceleration_scale);
}
