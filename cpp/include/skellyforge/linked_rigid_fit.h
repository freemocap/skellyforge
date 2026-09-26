#pragma once
#include "skellyforge/rigid_fit.h"
namespace skellyforge {
struct LinkedRigidFit {
  std::array<std::array<double,4>,2> quaternions;
  std::array<Vec3,2> translations;
  Vec3 joint;
  std::vector<double> costs;
  bool converged;
  std::string report;
  double seconds;
};
// Two rigid segments; attachment_a and attachment_b are segment-local points.
// One shared world point enforces coincidence exactly, without a penalty weight.
LinkedRigidFit fit_linked_rigid(const std::vector<Vec3>& local_a,
  const std::vector<Vec3>& observed_a, const Vec3& attachment_a,
  const std::vector<Vec3>& local_b, const std::vector<Vec3>& observed_b,
  const Vec3& attachment_b);
}
