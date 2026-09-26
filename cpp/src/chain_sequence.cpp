#include "skellyforge/chain_sequence.h"
#include "skellyforge/rigid_residuals.h"
#include <ceres/ceres.h>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>
namespace skellyforge {
namespace {
// Blocks: root XYZ, then segment WORLD quaternions from root through target.
// Translations are derived from attachments, with parent-local XYZ displacement
// only on explicitly selected linkages. No independent child translation blocks.
template <typename T> void chain_translation(T const* const* blocks,size_t target,
  const std::vector<Vec3>& parent,const std::vector<Vec3>& child,T* translation,const T* displacement=nullptr, const std::vector<double>& references={},const std::vector<int>& slots={},const std::vector<int>& linkage_slots={}){
  for(int k=0;k<3;++k) translation[k]=blocks[0][k];
  for(size_t j=0;j<target;++j){
    T a[3],b[3],ra[3],rb[3];
    for(int k=0;k<3;++k){a[k]=T(parent[j][k]);b[k]=T(child[j][k]);}
    if(!references.empty()){
      if(references[j]>0)a[2]*=blocks[slots[j]][0]/T(references[j]);
      if(references[j+1]>0)b[2]*=blocks[slots[j+1]][0]/T(references[j+1]);
    }
    if(!linkage_slots.empty() && linkage_slots[j+1]>=0)
      for(int k=0;k<3;++k)a[k]+=blocks[linkage_slots[j+1]][k];
    if(j==1 && displacement) a[2]+=displacement[0];
    ceres::QuaternionRotatePoint(blocks[j+1],a,ra);
    ceres::QuaternionRotatePoint(blocks[j+2],b,rb);
    for(int k=0;k<3;++k) translation[k]+=ra[k]-rb[k];
  }
}
struct ChainLandmarkResidual {
  size_t target;
  Vec3 local,observed;
  std::vector<Vec3> parent,child;
  double scale;
  bool displacement;
  std::vector<double> references;
  std::vector<int> slots;
  std::vector<int> linkage_slots;
  template <typename T> bool operator()(T const* const* blocks,T* residual)const{
    T translation[3],point[3],rotated[3];
    chain_translation(blocks,target,parent,child,translation,displacement?blocks[target+2]:nullptr,references,slots,linkage_slots);
    for(int k=0;k<3;++k)point[k]=T(local[k]);
    if(!references.empty() && references[target]>0)point[2]*=blocks[slots[target]][0]/T(references[target]);
    ceres::QuaternionRotatePoint(blocks[target+1],point,rotated);
    for(int k=0;k<3;++k)residual[k]=(translation[k]+rotated[k]-T(observed[k]))*T(scale);
    return true;
  }
};
struct ChainLandmarkLineResidual {
  ChainLandmarkResidual point_from_origin;
  Vec3 lateral, anterior;
  double distance_scale, anterior_scale;
  template <typename T> bool operator()(T const* const* blocks,T* residual)const{
    T delta[3];point_from_origin(blocks,delta);
    T side=T(0),front=T(0);
    for(int k=0;k<3;++k){side+=delta[k]*T(lateral[k]);front+=delta[k]*T(anterior[k]);}
    residual[0]=side*T(distance_scale);
    residual[1]=front*T(distance_scale);
    residual[2]=(front>T(0)?front:T(0))*T(anterior_scale);
    return true;
  }
};
}
ChainSequenceFit fit_chain_sequence(const std::vector<std::vector<Vec3>>& local,
  const std::vector<std::vector<std::vector<Vec3>>>& observed,
  const std::vector<Vec3>& parent_attachments,const std::vector<Vec3>& child_attachments,
  const std::vector<double>& times,double position_scale,
  double linear_acceleration_scale,double angular_acceleration_scale,
  bool allow_displacement,double displacement_scale,double displacement_acceleration_scale,double displacement_bound,const std::vector<int>& parent_indices,const std::vector<ChainQuaternions>& initial_quaternions,const std::vector<Vec3>& initial_roots,const ChainQuaternions& rest_relative_quaternions,double rest_pose_scale,const std::vector<double>& axial_reference_lengths,double length_prior_fraction,double length_acceleration_scale,const std::vector<std::vector<std::vector<int>>>& observation_indices,std::optional<double> lengthening_prior_fraction,bool free_axial_lengths,const std::optional<LandmarkLinePrior>& landmark_line_prior,const std::vector<int>& relaxed_linkage_children,double linkage_scale,double linkage_acceleration_scale,const std::optional<LengthEqualityPrior>& length_equality_prior){
  const size_t n=times.size(),bodies=local.size();
  std::vector<bool> relaxed(bodies,false);
  for(int child:relaxed_linkage_children){
    if(child<=0 || static_cast<size_t>(child)>=bodies || relaxed[child])throw std::invalid_argument("Relaxed linkage children must be unique non-root segments");
    relaxed[child]=true;
  }
  if(allow_displacement && !relaxed_linkage_children.empty())throw std::invalid_argument("Legacy axial displacement and general linkage displacement cannot be combined");
  const double extension_fraction=lengthening_prior_fraction.value_or(length_prior_fraction);
  if(bodies<1 || parent_attachments.size()!=bodies-1 || child_attachments.size()!=bodies-1 || parent_indices.size()!=bodies-1)
    throw std::invalid_argument("Each non-root segment requires a parent index and attachment pair");
  std::vector<std::vector<size_t>> paths(bodies);
  paths[0]={0};
  for(size_t b=1;b<bodies;++b){
    const int parent=parent_indices[b-1];
    if(parent<0 || static_cast<size_t>(parent)>=b)
      throw std::invalid_argument("Segments must be ordered with each parent before its child");
    paths[b]=paths[parent];paths[b].push_back(b);
  }
  if(allow_displacement && (bodies!=3 || parent_indices!=std::vector<int>{0,1}))
    throw std::invalid_argument("Displacement is supported only for the three-segment axial chain experiment");
  if(!axial_reference_lengths.empty()){
    if(axial_reference_lengths.size()!=bodies || allow_displacement)throw std::invalid_argument("Axial length definitions must cover segments and cannot combine with linkage displacement");
    for(double reference:axial_reference_lengths)if(!std::isfinite(reference)||reference<0)throw std::invalid_argument("Axial reference lengths must be finite and nonnegative; zero means rigid");
  }
  const std::vector<double> references=axial_reference_lengths.empty()?std::vector<double>(bodies,0.):axial_reference_lengths;
  if(length_equality_prior){
    const auto& prior=*length_equality_prior;
    if(prior.segment_a<0 || prior.segment_b<0 || static_cast<size_t>(prior.segment_a)>=bodies || static_cast<size_t>(prior.segment_b)>=bodies || prior.segment_a==prior.segment_b)
      throw std::invalid_argument("Length equality requires two distinct valid segment indices");
    if(references[prior.segment_a]<=0 || references[prior.segment_b]<=0)
      throw std::invalid_argument("Length equality requires two variable axial lengths");
    if(!std::isfinite(prior.scale) || prior.scale<=0)
      throw std::invalid_argument("Length equality scale must be finite and positive");
  }
  if(n<3||observed.size()!=n)throw std::invalid_argument("At least three matching frames and timestamps are required");
  if(landmark_line_prior){
    const auto& prior=*landmark_line_prior;
    if(prior.segment<0 || static_cast<size_t>(prior.segment)>=bodies || prior.frames.size()!=n)
      throw std::invalid_argument("Line prior must name a segment and match timestamps");
    for(double value:prior.local_point)if(!std::isfinite(value))throw std::invalid_argument("Line prior point must be finite");
    for(double scale:{prior.distance_scale,prior.anterior_scale})if(!std::isfinite(scale)||scale<=0)
      throw std::invalid_argument("Line prior scales must be finite and positive");
    for(const auto& frame:prior.frames)if(frame){
      for(const auto& axis:*frame)for(double value:axis)if(!std::isfinite(value))throw std::invalid_argument("Line prior frame must be finite");
      double a=0,b=0,dot=0;for(int k=0;k<3;++k){a+=(*frame)[1][k]*(*frame)[1][k];b+=(*frame)[2][k]*(*frame)[2][k];dot+=(*frame)[1][k]*(*frame)[2][k];}
      if(std::abs(a-1)>kLineBasisTolerance || std::abs(b-1)>kLineBasisTolerance || std::abs(dot)>kLineBasisTolerance)
        throw std::invalid_argument("Line prior axes must be orthonormal");
    }
  }
  const bool seeded=!initial_quaternions.empty() || !initial_roots.empty();
  const bool indexed=!observation_indices.empty();
  if(indexed && (!seeded || observation_indices.size()!=n))
    throw std::invalid_argument("Indexed observations require explicit initialization and one index frame per timestamp");
  for(const auto& points:local)if(!seeded && points.size()<3)throw std::invalid_argument("Each segment requires at least three local landmarks");
  for(double scale:{linkage_scale,linkage_acceleration_scale,position_scale,linear_acceleration_scale,angular_acceleration_scale,displacement_scale,displacement_acceleration_scale,displacement_bound,rest_pose_scale,length_prior_fraction,extension_fraction,length_acceleration_scale})
    if(!std::isfinite(scale)||scale<=0)throw std::invalid_argument("Scales must be finite and positive");
  for(const auto& offsets:{parent_attachments,child_attachments})for(const auto& p:offsets)for(double x:p)
    if(!std::isfinite(x))throw std::invalid_argument("Attachments must be finite");
  auto valid_quaternion=[](const std::array<double,4>& q){
    double norm=0;for(double value:q){if(!std::isfinite(value))throw std::invalid_argument("Quaternion must be finite");norm+=value*value;}
    if(std::abs(norm-1.)>kQuaternionSquaredNormTolerance)throw std::invalid_argument("Quaternion must be unit length");
  };
  if(seeded){
    if(initial_quaternions.size()!=n || initial_roots.size()!=n)throw std::invalid_argument("Initialization must match timestamps");
    for(const auto& frame:initial_quaternions){if(frame.size()!=bodies)throw std::invalid_argument("Initialization must match segments");for(const auto& q:frame)valid_quaternion(q);}
    for(const auto& root:initial_roots)for(double value:root)if(!std::isfinite(value))throw std::invalid_argument("Root initialization must be finite");
  }
  if(!rest_relative_quaternions.empty()){
    if(rest_relative_quaternions.size()!=bodies-1)throw std::invalid_argument("One rest quaternion per linkage is required");
    for(const auto& q:rest_relative_quaternions)valid_quaternion(q);
  }
  for(const auto& points:local)for(const auto& point:points)for(double value:point)if(!std::isfinite(value))throw std::invalid_argument("Local landmarks must be finite");
  std::vector<double> weights(n,0);
  for(size_t i=0;i<n;++i){
    if(!std::isfinite(times[i]))throw std::invalid_argument("Timestamps must be finite");
    if(i){const double dt=times[i]-times[i-1];
      if(!std::isfinite(dt)||dt<=0)throw std::invalid_argument("Timestamps must increase strictly");
      weights[i]+=dt/2;weights[i-1]+=dt/2;
    }
    if(observed[i].size()!=bodies)throw std::invalid_argument("Each frame must contain one observation list per segment");
    if(indexed && observation_indices[i].size()!=bodies)throw std::invalid_argument("Observation indices must match segments");
    for(size_t b=0;b<bodies;++b){
      for(const auto& point:observed[i][b])for(double value:point)if(!std::isfinite(value))throw std::invalid_argument("Observations must be finite");
      if(indexed){
        const auto& indices=observation_indices[i][b];
        if(indices.size()!=observed[i][b].size())throw std::invalid_argument("Observation indices must match supplied targets");
        std::vector<bool> seen(local[b].size(),false);
        for(int index:indices){
          if(index<0 || static_cast<size_t>(index)>=local[b].size() || seen[index])throw std::invalid_argument("Observation index is invalid or repeated");
          seen[index]=true;
        }
      }else if(observed[i][b].size()!=local[b].size() && !(b>0&&observed[i][b].empty()))
        throw std::invalid_argument("Root must be fully observed; child frames must be full or empty");
    }
  }
  if(!seeded)for(size_t b=1;b<bodies;++b)
    if(observed.front()[b].empty()||observed.back()[b].empty())throw std::invalid_argument("Child gaps must be bounded by observations");
  ChainSequenceFit result;
  result.displacements.resize(n,0.);
  result.linkage_displacements.resize(n,std::vector<Vec3>(bodies,Vec3{0.,0.,0.}));
  result.lengths.resize(n,references);
  result.quaternions.resize(n,ChainQuaternions(bodies));result.roots.resize(n);
  result.translations.resize(n,std::vector<Vec3>(bodies));
  if(seeded){result.quaternions=initial_quaternions;result.roots=initial_roots;}
  else {
  std::vector<std::vector<size_t>> support(bodies);
  for(size_t i=0;i<n;++i)for(size_t b=0;b<bodies;++b){
    if(observed[i][b].empty())continue;
    const auto fit=fit_rigid(local[b],observed[i][b],{1,0,0,0},{0,0,0});
    if(!fit.converged)throw std::runtime_error("Independent initialization did not converge");
    result.quaternions[i][b]=fit.quaternion;
    if(b==0)result.roots[i]=fit.translation;
    support[b].push_back(i);
  }
  for(size_t segment=1;segment<bodies;++segment)for(size_t j=1;j<support[segment].size();++j){
    const auto lo=support[segment][j-1],hi=support[segment][j];
    const auto& qa=result.quaternions[lo][segment];const auto& qb=result.quaternions[hi][segment];
    const Eigen::Quaterniond a(qa[0],qa[1],qa[2],qa[3]),b(qb[0],qb[1],qb[2],qb[3]);
    for(size_t i=lo+1;i<hi;++i){const auto q=a.slerp((times[i]-times[lo])/(times[hi]-times[lo]),b).normalized();result.quaternions[i][segment]={q.w(),q.x(),q.y(),q.z()};}
  }
  }
  auto translations=[&](){for(size_t i=0;i<n;++i){
    for(size_t b=0;b<bodies;++b){
      std::vector<const double*> blocks={result.roots[i].data()};
      std::vector<Vec3> pa,ca;
      std::vector<double> path_references;std::vector<int> slots;
      for(auto node:paths[b]){
        blocks.push_back(result.quaternions[i][node].data());
        if(node){pa.push_back(parent_attachments[node-1]);ca.push_back(child_attachments[node-1]);}
      }
      for(auto node:paths[b]){
        path_references.push_back(references[node]);slots.push_back(-1);
        if(references[node]>0){slots.back()=static_cast<int>(blocks.size());blocks.push_back(&result.lengths[i][node]);}
      }
      std::vector<int> linkage_slots;
      for(auto node:paths[b]){
        linkage_slots.push_back(-1);
        if(relaxed[node]){linkage_slots.back()=static_cast<int>(blocks.size());blocks.push_back(result.linkage_displacements[i][node].data());}
      }
      chain_translation(blocks.data(),paths[b].size()-1,pa,ca,result.translations[i][b].data(),allow_displacement?&result.displacements[i]:nullptr,path_references,slots,linkage_slots);
    }
  }};
  translations();result.initial_quaternions=result.quaternions;result.initial_translations=result.translations;
  ceres::Problem problem;
  std::vector<ceres::ResidualBlockId> landmark_blocks,line_blocks,root_blocks,linkage_priors,linkage_motion,displacement_priors,displacement_motion,relative_pose_blocks,length_priors,length_motion,length_equalities;
  std::vector<std::vector<ceres::ResidualBlockId>> angular_blocks(bodies);
  for(size_t i=0;i<n;++i){
    problem.AddParameterBlock(result.roots[i].data(),3);
    for(size_t b=0;b<bodies;++b)if(references[b]>0){
      problem.AddParameterBlock(&result.lengths[i][b],1);
      // Free-length diagnostic: retain only the nonnegative length domain.
      if(free_axial_lengths)problem.SetParameterLowerBound(&result.lengths[i][b],0,0.);
      else {
        problem.SetParameterLowerBound(&result.lengths[i][b],0,kAxialLengthMinimumFraction*references[b]);
        problem.SetParameterUpperBound(&result.lengths[i][b],0,kAxialLengthMaximumFraction*references[b]);
        length_priors.push_back(problem.AddResidualBlock(new ceres::AutoDiffCostFunction<LengthPriorResidual,1,1>(
          new LengthPriorResidual{references[b],std::sqrt(weights[i])/(length_prior_fraction*references[b]),
            std::sqrt(weights[i])/(extension_fraction*references[b])}),nullptr,&result.lengths[i][b]));
      }
    }
    if(length_equality_prior){
      const auto& prior=*length_equality_prior;
      length_equalities.push_back(problem.AddResidualBlock(new ceres::AutoDiffCostFunction<LengthEqualityResidual,1,1,1>(
        new LengthEqualityResidual{std::sqrt(weights[i])/prior.scale}),nullptr,
        &result.lengths[i][prior.segment_a],&result.lengths[i][prior.segment_b]));
    }
    for(int child:relaxed_linkage_children){
      auto* delta=result.linkage_displacements[i][child].data();
      problem.AddParameterBlock(delta,3);
      linkage_priors.push_back(problem.AddResidualBlock(new ceres::AutoDiffCostFunction<LinkageDisplacementPriorResidual,3,3>(
        new LinkageDisplacementPriorResidual{std::sqrt(weights[i])/linkage_scale}),nullptr,delta));
    }
    if(allow_displacement){
      problem.AddParameterBlock(&result.displacements[i],1);
      problem.SetParameterLowerBound(&result.displacements[i],0,-displacement_bound);
      problem.SetParameterUpperBound(&result.displacements[i],0,displacement_bound);
      displacement_priors.push_back(problem.AddResidualBlock(new ceres::AutoDiffCostFunction<DisplacementPriorResidual,1,1>(
        new DisplacementPriorResidual{std::sqrt(weights[i])/displacement_scale}),nullptr,&result.displacements[i]));
    }
    for(size_t b=0;b<bodies;++b)problem.AddParameterBlock(result.quaternions[i][b].data(),4,new ceres::QuaternionManifold());
    if(!rest_relative_quaternions.empty())for(size_t b=1;b<bodies;++b)
      relative_pose_blocks.push_back(problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<RelativePoseResidual,3,4,4>(new RelativePoseResidual{rest_relative_quaternions[b-1],std::sqrt(weights[i])/rest_pose_scale}),nullptr,
        result.quaternions[i][parent_indices[b-1]].data(),result.quaternions[i][b].data()));
    for(size_t b=0;b<bodies;++b){
      std::vector<double*> blocks={result.roots[i].data()};
      std::vector<Vec3> pa,ca;
      std::vector<double> path_references;std::vector<int> slots;
      for(auto node:paths[b]){
        blocks.push_back(result.quaternions[i][node].data());
        if(node){pa.push_back(parent_attachments[node-1]);ca.push_back(child_attachments[node-1]);}
      }
      if(allow_displacement && b==2)blocks.push_back(&result.displacements[i]);
      for(auto node:paths[b]){
        path_references.push_back(references[node]);slots.push_back(-1);
        if(references[node]>0){slots.back()=static_cast<int>(blocks.size());blocks.push_back(&result.lengths[i][node]);}
      }
      std::vector<int> linkage_slots;
      for(auto node:paths[b]){
        linkage_slots.push_back(-1);
        if(relaxed[node]){linkage_slots.back()=static_cast<int>(blocks.size());blocks.push_back(result.linkage_displacements[i][node].data());}
      }
      for(size_t j=0;j<observed[i][b].size();++j){
        auto* cost=new ceres::DynamicAutoDiffCostFunction<ChainLandmarkResidual>(new ChainLandmarkResidual{
          paths[b].size()-1,local[b][indexed?observation_indices[i][b][j]:j],observed[i][b][j],pa,ca,std::sqrt(weights[i])/position_scale,allow_displacement&&b==2,path_references,slots,linkage_slots});
        cost->AddParameterBlock(3);for(size_t k=0;k<paths[b].size();++k)cost->AddParameterBlock(4);
        if(allow_displacement && b==2)cost->AddParameterBlock(1);
        for(double reference:path_references)if(reference>0)cost->AddParameterBlock(1);
        for(auto node:paths[b])if(relaxed[node])cost->AddParameterBlock(3);
        cost->SetNumResiduals(3);
        landmark_blocks.push_back(problem.AddResidualBlock(cost,nullptr,blocks));
      }
      if(landmark_line_prior && landmark_line_prior->segment==static_cast<int>(b) && landmark_line_prior->frames[i]){
        const auto& prior=*landmark_line_prior;const auto& frame=*prior.frames[i];
        auto* cost=new ceres::DynamicAutoDiffCostFunction<ChainLandmarkLineResidual>(new ChainLandmarkLineResidual{
          {paths[b].size()-1,prior.local_point,frame[0],pa,ca,1.,allow_displacement&&b==2,path_references,slots,linkage_slots},
          frame[1],frame[2],std::sqrt(weights[i])/prior.distance_scale,std::sqrt(weights[i])/prior.anterior_scale});
        cost->AddParameterBlock(3);for(size_t k=0;k<paths[b].size();++k)cost->AddParameterBlock(4);
        if(allow_displacement && b==2)cost->AddParameterBlock(1);
        for(double reference:path_references)if(reference>0)cost->AddParameterBlock(1);
        for(auto node:paths[b])if(relaxed[node])cost->AddParameterBlock(3);
        cost->SetNumResiduals(3);
        line_blocks.push_back(problem.AddResidualBlock(cost,nullptr,blocks));
      }
    }
  }
  for(size_t i=1;i+1<n;++i){
    const double before=times[i]-times[i-1],after=times[i+1]-times[i],midpoint_dt=(before+after)/2;
    for(size_t b=0;b<bodies;++b)if(references[b]>0 && !free_axial_lengths)length_motion.push_back(problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<DisplacementAccelerationResidual,1,1,1,1>(
        new DisplacementAccelerationResidual{before,after,1/(length_acceleration_scale*std::sqrt(midpoint_dt))}),nullptr,
        &result.lengths[i-1][b],&result.lengths[i][b],&result.lengths[i+1][b]));
    for(int child:relaxed_linkage_children)linkage_motion.push_back(problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<TranslationAccelerationResidual,3,3,3,3>(
        new TranslationAccelerationResidual{before,after,1/(linkage_acceleration_scale*std::sqrt(midpoint_dt))}),nullptr,
        result.linkage_displacements[i-1][child].data(),result.linkage_displacements[i][child].data(),result.linkage_displacements[i+1][child].data()));
    if(allow_displacement)displacement_motion.push_back(problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<DisplacementAccelerationResidual,1,1,1,1>(
        new DisplacementAccelerationResidual{before,after,1/(displacement_acceleration_scale*std::sqrt(midpoint_dt))}),nullptr,
        &result.displacements[i-1],&result.displacements[i],&result.displacements[i+1]));
    root_blocks.push_back(problem.AddResidualBlock(new ceres::AutoDiffCostFunction<TranslationAccelerationResidual,3,3,3,3>(
      new TranslationAccelerationResidual{before,after,1/(linear_acceleration_scale*std::sqrt(midpoint_dt))}),nullptr,
      result.roots[i-1].data(),result.roots[i].data(),result.roots[i+1].data()));
    for(size_t b=0;b<bodies;++b)angular_blocks[b].push_back(problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<QuaternionAccelerationResidual,3,4,4,4>(
        new QuaternionAccelerationResidual{before,after,1/(angular_acceleration_scale*std::sqrt(midpoint_dt))}),nullptr,
        result.quaternions[i-1][b].data(),result.quaternions[i][b].data(),result.quaternions[i+1][b].data()));
  }
  ceres::Solver::Options options;options.linear_solver_type=ceres::SPARSE_NORMAL_CHOLESKY;options.logging_type=ceres::SILENT;
  options.max_num_iterations=kChainMaximumIterations;options.function_tolerance=kChainFunctionTolerance;options.gradient_tolerance=kChainGradientTolerance;options.parameter_tolerance=kChainParameterTolerance;
  ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);
  translations();for(const auto& step:summary.iterations)result.costs.push_back(step.cost);
  auto cost=[&](const std::vector<ceres::ResidualBlockId>& blocks){ceres::Problem::EvaluateOptions e;e.residual_blocks=blocks;double value;
    if(blocks.empty())return 0.;
    if(!problem.Evaluate(e,&value,nullptr,nullptr,nullptr))throw std::runtime_error("Cost evaluation failed");return value;};
  if(allow_displacement){result.displacement_prior_cost=cost(displacement_priors);result.displacement_acceleration_cost=cost(displacement_motion);}
  result.linkage_prior_cost=cost(linkage_priors);result.linkage_acceleration_cost=cost(linkage_motion);
  result.length_equality_cost=cost(length_equalities);
  result.line_prior_cost=cost(line_blocks);
  result.length_prior_cost=cost(length_priors);result.length_acceleration_cost=cost(length_motion);
  result.relative_pose_cost=cost(relative_pose_blocks);
  result.landmark_cost=cost(landmark_blocks);result.root_acceleration_cost=cost(root_blocks);
  for(const auto& blocks:angular_blocks)result.angular_acceleration_costs.push_back(cost(blocks));
  result.parameter_blocks=problem.NumParameterBlocks();result.residual_blocks=problem.NumResidualBlocks();
  result.converged=summary.termination_type==ceres::CONVERGENCE;result.report=summary.BriefReport();result.seconds=summary.total_time_in_seconds;
  return result;
}
}
