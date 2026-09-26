#pragma once
#include <array>
#include <vector>
#include <string>
namespace skellyforge {
using Vec3 = std::array<double, 3>;
struct RigidFit {
  std::array<double, 4> quaternion;
  Vec3 translation;
  std::vector<double> costs;
  bool converged;
  std::string report;
  double seconds;
};
RigidFit fit_rigid(const std::vector<Vec3>& local, const std::vector<Vec3>& observed,
                  std::array<double,4> quaternion, Vec3 translation,
                  bool allow_underconstrained = false);
}
