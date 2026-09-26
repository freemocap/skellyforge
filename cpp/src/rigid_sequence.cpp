#include "skellyforge/rigid_sequence.h"
#include "skellyforge/rigid_residuals.h"
#include <ceres/ceres.h>
#include <cmath>
#include <stdexcept>
namespace skellyforge {
RigidSequenceFit fit_rigid_sequence(const std::vector<Vec3>& local,
  const std::vector<std::vector<Vec3>>& observed, const std::vector<double>& times,
  double position_scale, double linear_motion_scale, double angular_motion_scale, const std::string& temporal_model) {
  const auto n=times.size();
  const bool acceleration=temporal_model=="acceleration";
  if(!acceleration && temporal_model!="velocity") throw std::invalid_argument("Unknown temporal model");
  if(acceleration && n<3) throw std::invalid_argument("Acceleration requires at least three frames");
  if(n<2 || observed.size()!=n) throw std::invalid_argument("At least two matching frames and timestamps are required");
  for(double s:{position_scale,linear_motion_scale,angular_motion_scale})
    if(!std::isfinite(s)||s<=0) throw std::invalid_argument("Scales must be finite and positive");
  std::vector<double> weights(n,0);
  for(size_t i=0;i<n;++i){
    if(!std::isfinite(times[i])) throw std::invalid_argument("Timestamps must be finite");
    if(i){double dt=times[i]-times[i-1];
      if(!std::isfinite(dt)||dt<=0) throw std::invalid_argument("Timestamps must increase strictly");
      weights[i]+=dt/2;weights[i-1]+=dt/2;}
  }
  RigidSequenceFit result;
  // Reserve before giving Ceres pointers: parameter storage must never relocate.
  result.quaternions.reserve(n);result.translations.reserve(n);
  for(const auto& frame:observed){
    auto fit=fit_rigid(local,frame,{1,0,0,0},{0,0,0});
    if(!fit.converged) throw std::runtime_error("Independent frame initialization did not converge");
    result.quaternions.push_back(fit.quaternion);result.translations.push_back(fit.translation);
  }
  ceres::Problem problem;
  std::vector<ceres::ResidualBlockId> landmarks,translations,rotations;
  for(size_t i=0;i<n;++i){
    auto* q=result.quaternions[i].data();auto* t=result.translations[i].data();
    problem.AddParameterBlock(q,4,new ceres::QuaternionManifold());
    for(size_t j=0;j<local.size();++j)
      landmarks.push_back(problem.AddResidualBlock(new ceres::AutoDiffCostFunction<LandmarkResidual,3,4,3>(
        new LandmarkResidual{local[j],observed[i][j],std::sqrt(weights[i])/position_scale}),nullptr,q,t));
    if(i && !acceleration){double dt=times[i]-times[i-1];
      translations.push_back(problem.AddResidualBlock(new ceres::AutoDiffCostFunction<TranslationMotionResidual,3,3,3>(
        new TranslationMotionResidual{1/(linear_motion_scale*std::sqrt(dt))}),nullptr,result.translations[i-1].data(),t));
      rotations.push_back(problem.AddResidualBlock(new ceres::AutoDiffCostFunction<QuaternionMotionResidual,3,4,4>(
        new QuaternionMotionResidual{1/(angular_motion_scale*std::sqrt(dt))}),nullptr,result.quaternions[i-1].data(),q));
    }
  }
  if(acceleration) for(size_t i=1;i+1<n;++i){
    const double before=times[i]-times[i-1],after=times[i+1]-times[i];
    const double midpoint_dt=(before+after)/2;
    translations.push_back(problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<TranslationAccelerationResidual,3,3,3,3>(
        new TranslationAccelerationResidual{before,after,1/(linear_motion_scale*std::sqrt(midpoint_dt))}),
      nullptr,result.translations[i-1].data(),result.translations[i].data(),result.translations[i+1].data()));
    rotations.push_back(problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<QuaternionAccelerationResidual,3,4,4,4>(
        new QuaternionAccelerationResidual{before,after,1/(angular_motion_scale*std::sqrt(midpoint_dt))}),
      nullptr,result.quaternions[i-1].data(),result.quaternions[i].data(),result.quaternions[i+1].data()));
  }
  ceres::Solver::Options options;options.linear_solver_type=ceres::SPARSE_NORMAL_CHOLESKY;
  options.logging_type=ceres::SILENT;options.max_num_iterations=100;
  options.function_tolerance=1e-10;options.gradient_tolerance=1e-10;options.parameter_tolerance=1e-10;
  ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);
  for(const auto& step:summary.iterations)result.costs.push_back(step.cost);
  auto cost=[&](const std::vector<ceres::ResidualBlockId>& blocks){
    ceres::Problem::EvaluateOptions e;e.residual_blocks=blocks;double value;
    if(!problem.Evaluate(e,&value,nullptr,nullptr,nullptr))throw std::runtime_error("Cost evaluation failed");
    return value;};
  result.landmark_cost=cost(landmarks);result.translation_cost=cost(translations);result.rotation_cost=cost(rotations);
  result.seconds=summary.total_time_in_seconds;result.converged=summary.termination_type==ceres::CONVERGENCE;
  result.report=summary.BriefReport();return result;
}
}
