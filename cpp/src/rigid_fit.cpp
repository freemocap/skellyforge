#include "skellyforge/rigid_fit.h"
#include "skellyforge/rigid_residuals.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <stdexcept>
namespace skellyforge {
RigidFit fit_rigid(const std::vector<Vec3>& local, const std::vector<Vec3>& observed,
                  std::array<double,4> q, Vec3 t, bool allow_underconstrained) {
  if(local.size()!=observed.size() || local.empty() || (!allow_underconstrained && local.size()<3))
    throw std::invalid_argument("Provide matching landmarks (at least three unless underconstrained fitting is explicit)");
  for(const auto& points : {local, observed}) for(const auto& p:points)
    for(double x:p) if(!std::isfinite(x)) throw std::invalid_argument("Landmarks must be finite");
  double norm=0; for(double x:q) { if(!std::isfinite(x)) throw std::invalid_argument("Quaternion must be finite"); norm+=x*x; }
  if(std::abs(norm-1)>1e-8) throw std::invalid_argument("Quaternion must be unit wxyz");
  for(double x:t) if(!std::isfinite(x)) throw std::invalid_argument("Translation must be finite");
  Eigen::Vector3d center=Eigen::Vector3d::Zero();
  for(auto p:local) center+=Eigen::Vector3d(p[0],p[1],p[2]);
  center/=double(local.size()); Eigen::Matrix3d covariance=Eigen::Matrix3d::Zero();
  for(auto p:local) {Eigen::Vector3d d=Eigen::Vector3d(p[0],p[1],p[2])-center; covariance+=d*d.transpose();}
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(covariance);
  if(!allow_underconstrained && (eig.eigenvalues()[2]<=0 || eig.eigenvalues()[1]<=eig.eigenvalues()[2]*1e-12))
    throw std::invalid_argument("Local landmarks must be non-collinear");
  ceres::Problem problem;
  problem.AddParameterBlock(q.data(),4,new ceres::QuaternionManifold());
  for(size_t i=0;i<local.size();++i)
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<LandmarkResidual,3,4,3>(
      new LandmarkResidual{local[i],observed[i]}),nullptr,q.data(),t.data());
  ceres::Solver::Options options; options.linear_solver_type=ceres::DENSE_QR;
  options.logging_type=ceres::SILENT;
  // Numerical convergence tolerances for this rigid-fit experiment, not noise scales.
  options.function_tolerance=1e-12;
  options.gradient_tolerance=1e-12;
  options.parameter_tolerance=1e-12;
  ceres::Solver::Summary summary; ceres::Solve(options,&problem,&summary);
  std::vector<double> costs; for(const auto& step:summary.iterations) costs.push_back(step.cost);
  return {q,t,costs,summary.termination_type==ceres::CONVERGENCE,summary.BriefReport(),summary.total_time_in_seconds};
}
}
