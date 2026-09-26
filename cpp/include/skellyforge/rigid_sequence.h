#pragma once
#include "skellyforge/rigid_fit.h"
namespace skellyforge {
struct RigidSequenceFit {
  std::vector<std::array<double,4>> quaternions;
  std::vector<Vec3> translations;
  std::vector<double> costs;
  double landmark_cost, translation_cost, rotation_cost, seconds;
  bool converged;
  std::string report;
};
RigidSequenceFit fit_rigid_sequence(const std::vector<Vec3>& local,
  const std::vector<std::vector<Vec3>>& observed, const std::vector<double>& times,
  double position_scale, double linear_motion_scale, double angular_motion_scale, const std::string& temporal_model);
}
