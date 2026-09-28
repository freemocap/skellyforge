#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <ceres/ceres.h>
namespace ceres { class Problem; }
namespace skellyforge {
// Read-only copies of Ceres facts. IDs are local to one assembled problem;
// they are not model identities or portable memory addresses.
struct ParameterIdentity {
  int frame=-1, segment=-1;
  std::string quantity;
};
struct ParameterInspection {
  int frame=-1, segment=-1;
  std::string quantity;
  int id, ambient_size, tangent_size;
  bool constant;
  std::string manifold_type;
  std::vector<double> values, lower_bounds, upper_bounds;
};
struct ResidualInspection {
  int id;
  std::string cost_type, loss_type, purpose;
  std::vector<int> parameter_ids;
  std::vector<double> values; // before loss, including weights inside the functor
  double cost; // after loss, Ceres' 1/2 rho(s)
};
struct ProblemInspection {
  std::vector<ParameterInspection> parameters;
  std::vector<ResidualInspection> residuals;
};
ProblemInspection inspect_problem(const ceres::Problem& problem,
 const std::unordered_map<const double*,ParameterIdentity>& identities={},
 const std::unordered_map<ceres::ResidualBlockId,std::string>& purposes={});
}
