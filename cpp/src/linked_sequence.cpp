#include "skellyforge/linked_sequence.h"
#include "skellyforge/rigid_residuals.h"
#include <ceres/ceres.h>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>
namespace skellyforge {
LinkedSequenceFit fit_linked_sequence(const std::vector<Vec3>& local_a,
  const std::vector<std::vector<Vec3>>& observed_a, const Vec3& attachment_a,
  const std::vector<Vec3>& local_b, const std::vector<std::vector<Vec3>>& observed_b,
  const Vec3& attachment_b, const std::vector<double>& times,
  double position_scale, double linear_acceleration_scale, double angular_acceleration_scale) {
  const size_t n=times.size();
  if(n<3 || observed_a.size()!=n || observed_b.size()!=n)
    throw std::invalid_argument("At least three matching frames and timestamps are required");
  for(double scale:{position_scale,linear_acceleration_scale,angular_acceleration_scale})
    if(!std::isfinite(scale)||scale<=0) throw std::invalid_argument("Scales must be finite and positive");
  for(const auto& p:{attachment_a,attachment_b}) for(double x:p)
    if(!std::isfinite(x)) throw std::invalid_argument("Attachments must be finite");
  std::vector<double> weights(n,0);
  for(size_t i=0;i<n;++i){
    if(!std::isfinite(times[i])) throw std::invalid_argument("Timestamps must be finite");
    if(i){const double dt=times[i]-times[i-1];
      if(!std::isfinite(dt)||dt<=0) throw std::invalid_argument("Timestamps must increase strictly");
      weights[i]+=dt/2;weights[i-1]+=dt/2;
    }
    if(observed_a[i].size()!=local_a.size() || (!observed_b[i].empty() && observed_b[i].size()!=local_b.size()))
      throw std::invalid_argument("Parent must be fully observed; child frames must be full or empty");
  }
  if(observed_b.front().empty()||observed_b.back().empty())
    throw std::invalid_argument("Child gaps must be bounded by observed frames");
  LinkedSequenceFit result;
  // Allocate once: Ceres parameter pointers must remain stable.
  result.quaternions.resize(n);result.joints.resize(n);result.translations.resize(n);
  std::vector<size_t> supported;
  for(size_t i=0;i<n;++i){
    if(!observed_b[i].empty()){
      const auto fit=fit_linked_rigid(local_a,observed_a[i],attachment_a,local_b,observed_b[i],attachment_b);
      if(!fit.converged) throw std::runtime_error("Connected frame initialization did not converge");
      result.quaternions[i]=fit.quaternions;result.joints[i]=fit.joint;supported.push_back(i);
    }else{
      const auto fit=fit_rigid(local_a,observed_a[i],{1,0,0,0},{0,0,0});
      if(!fit.converged) throw std::runtime_error("Parent initialization did not converge");
      result.quaternions[i][0]=fit.quaternion;
      Vec3 offset;ceres::QuaternionRotatePoint(fit.quaternion.data(),attachment_a.data(),offset.data());
      for(int k=0;k<3;++k) result.joints[i][k]=offset[k]+fit.translation[k];
    }
  }
  // Slerp is initialization only. Every child quaternion remains a free Ceres
  // parameter block and is subsequently optimized; no interpolation residual.
  for(size_t j=1;j<supported.size();++j){
    const auto lo=supported[j-1],hi=supported[j];
    const auto& qa=result.quaternions[lo][1];const auto& qb=result.quaternions[hi][1];
    const Eigen::Quaterniond a(qa[0],qa[1],qa[2],qa[3]),b(qb[0],qb[1],qb[2],qb[3]);
    for(size_t i=lo+1;i<hi;++i){
      const auto q=a.slerp((times[i]-times[lo])/(times[hi]-times[lo]),b).normalized();
      result.quaternions[i][1]={q.w(),q.x(),q.y(),q.z()};
    }
  }
  const std::array<Vec3,2> attachments={attachment_a,attachment_b};
  auto translations=[&](){
    for(size_t i=0;i<n;++i) for(int b=0;b<2;++b){
      Vec3 offset;ceres::QuaternionRotatePoint(result.quaternions[i][b].data(),attachments[b].data(),offset.data());
      for(int k=0;k<3;++k) result.translations[i][b][k]=result.joints[i][k]-offset[k];
    }
  };
  translations();result.initial_quaternions=result.quaternions;
  result.initial_translations=result.translations;result.initial_joints=result.joints;
  ceres::Problem problem;
  std::vector<ceres::ResidualBlockId> landmarks,joint_motion,parent_motion,child_motion;
  for(size_t i=0;i<n;++i){
    problem.AddParameterBlock(result.joints[i].data(),3);
    for(int b=0;b<2;++b){
      auto* q=result.quaternions[i][b].data();
      problem.AddParameterBlock(q,4,new ceres::QuaternionManifold());
      const auto& local=b==0?local_a:local_b;const auto& observed=b==0?observed_a[i]:observed_b[i];
      for(size_t j=0;j<observed.size();++j){
        Vec3 relative;for(int k=0;k<3;++k) relative[k]=local[j][k]-attachments[b][k];
        landmarks.push_back(problem.AddResidualBlock(new ceres::AutoDiffCostFunction<LandmarkResidual,3,4,3>(
          new LandmarkResidual{relative,observed[j],std::sqrt(weights[i])/position_scale}),nullptr,q,result.joints[i].data()));
      }
    }
  }
  for(size_t i=1;i+1<n;++i){
    const double before=times[i]-times[i-1],after=times[i+1]-times[i];
    const double midpoint_dt=(before+after)/2;
    joint_motion.push_back(problem.AddResidualBlock(new ceres::AutoDiffCostFunction<TranslationAccelerationResidual,3,3,3,3>(
      new TranslationAccelerationResidual{before,after,1/(linear_acceleration_scale*std::sqrt(midpoint_dt))}),nullptr,
      result.joints[i-1].data(),result.joints[i].data(),result.joints[i+1].data()));
    for(int b=0;b<2;++b){
      auto id=problem.AddResidualBlock(new ceres::AutoDiffCostFunction<QuaternionAccelerationResidual,3,4,4,4>(
        new QuaternionAccelerationResidual{before,after,1/(angular_acceleration_scale*std::sqrt(midpoint_dt))}),nullptr,
        result.quaternions[i-1][b].data(),result.quaternions[i][b].data(),result.quaternions[i+1][b].data());
      (b==0?parent_motion:child_motion).push_back(id);
    }
  }
  ceres::Solver::Options options;options.linear_solver_type=ceres::SPARSE_NORMAL_CHOLESKY;
  options.logging_type=ceres::SILENT;options.max_num_iterations=200;
  options.function_tolerance=1e-10;options.gradient_tolerance=1e-10;options.parameter_tolerance=1e-10;
  ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);
  translations();for(const auto& step:summary.iterations) result.costs.push_back(step.cost);
  auto cost=[&](const std::vector<ceres::ResidualBlockId>& blocks){
    ceres::Problem::EvaluateOptions e;e.residual_blocks=blocks;double value;
    if(!problem.Evaluate(e,&value,nullptr,nullptr,nullptr)) throw std::runtime_error("Cost evaluation failed");
    return value;
  };
  result.landmark_cost=cost(landmarks);result.joint_acceleration_cost=cost(joint_motion);
  result.parent_acceleration_cost=cost(parent_motion);result.child_acceleration_cost=cost(child_motion);
  result.parameter_blocks=problem.NumParameterBlocks();result.residual_blocks=problem.NumResidualBlocks();
  result.converged=summary.termination_type==ceres::CONVERGENCE;result.report=summary.BriefReport();result.seconds=summary.total_time_in_seconds;
  return result;
}
}
