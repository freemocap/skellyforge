#include "skellyforge/smoke.h"
#include <ceres/ceres.h>
#include <ceres/version.h>
#include <cmath>
#include <stdexcept>
namespace skellyforge {
struct ScalarResidual {
  double target;
  template <typename T> bool operator()(const T* x, T* residual) const {
    residual[0] = x[0] - T(target);
    return true;
  }
};
SmokeResult solve_scalar(double initial, double target) {
  if (!std::isfinite(initial) || !std::isfinite(target))
    throw std::invalid_argument("initial and target must be finite");
  double value = initial;
  ceres::Problem problem;
  problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ScalarResidual, 1, 1>(
      new ScalarResidual{target}), nullptr, &value);
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  return {value, summary.final_cost, summary.termination_type == ceres::CONVERGENCE};
}
std::string ceres_version() { return CERES_VERSION_STRING; }
}
