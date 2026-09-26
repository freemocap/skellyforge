#include "skellyforge/linked_rigid_fit.h"
#include "skellyforge/rigid_residuals.h"
#include <ceres/ceres.h>
#include <cmath>
#include <stdexcept>
namespace skellyforge {
LinkedRigidFit fit_linked_rigid(const std::vector<Vec3>& local_a,
  const std::vector<Vec3>& observed_a, const Vec3& attachment_a,
  const std::vector<Vec3>& local_b, const std::vector<Vec3>& observed_b,
  const Vec3& attachment_b) {
  for(const auto& p:{attachment_a,attachment_b}) for(double x:p)
    if(!std::isfinite(x)) throw std::invalid_argument("Attachments must be finite");
  // Existing rigid fit validates each segment's observations and local geometry.
  const auto a=fit_rigid(local_a,observed_a,{1,0,0,0},{0,0,0});
  // The attachment supplies geometric information absent from a sparse child fit.
  // A one-point child still has unobservable roll; convergence is not uniqueness.
  const auto b=fit_rigid(local_b,observed_b,{1,0,0,0},{0,0,0},true);
  if(!a.converged || !b.converged)
    throw std::runtime_error("Independent segment initialization did not converge");
  LinkedRigidFit result;
  result.quaternions={a.quaternion,b.quaternion};
  const std::array<Vec3,2> attachments={attachment_a,attachment_b};
  Vec3 ja,jb;
  ceres::QuaternionRotatePoint(a.quaternion.data(),attachment_a.data(),ja.data());
  ceres::QuaternionRotatePoint(b.quaternion.data(),attachment_b.data(),jb.data());
  for(int k=0;k<3;++k) result.joint[k]=(ja[k]+a.translation[k]+jb[k]+b.translation[k])/2;
  ceres::Problem problem;
  for(int body=0;body<2;++body){
    auto* q=result.quaternions[body].data();
    problem.AddParameterBlock(q,4,new ceres::QuaternionManifold());
    const auto& local=body==0?local_a:local_b;
    const auto& observed=body==0?observed_a:observed_b;
    for(size_t i=0;i<local.size();++i){
      Vec3 relative;
      for(int k=0;k<3;++k) relative[k]=local[i][k]-attachments[body][k];
      problem.AddResidualBlock(new ceres::AutoDiffCostFunction<LandmarkResidual,3,4,3>(
        new LandmarkResidual{relative,observed[i]}),nullptr,q,result.joint.data());
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type=ceres::DENSE_QR;
  options.logging_type=ceres::SILENT;
  options.max_num_iterations=100;
  options.function_tolerance=1e-12;options.gradient_tolerance=1e-12;options.parameter_tolerance=1e-12;
  ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);
  for(const auto& step:summary.iterations) result.costs.push_back(step.cost);
  for(int body=0;body<2;++body){
    Vec3 offset;
    ceres::QuaternionRotatePoint(result.quaternions[body].data(),attachments[body].data(),offset.data());
    for(int k=0;k<3;++k) result.translations[body][k]=result.joint[k]-offset[k];
  }
  result.converged=summary.termination_type==ceres::CONVERGENCE;
  result.report=summary.BriefReport();result.seconds=summary.total_time_in_seconds;
  return result;
}
}
