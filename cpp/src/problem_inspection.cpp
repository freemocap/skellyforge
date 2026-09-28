#include "skellyforge/problem_inspection.h"
#include <ceres/ceres.h>
#include <stdexcept>
#include <typeinfo>
#include <unordered_map>

namespace skellyforge {
ProblemInspection inspect_problem(const ceres::Problem& problem,
 const std::unordered_map<const double*,ParameterIdentity>& identities,
 const std::unordered_map<ceres::ResidualBlockId,std::string>& purposes) {
  ProblemInspection result;
  std::vector<double*> parameters;
  problem.GetParameterBlocks(&parameters);
  std::unordered_map<const double*,int> ids;
  for(const auto* values:parameters){
    ParameterInspection p;
    if(auto it=identities.find(values);it!=identities.end()){
      p.frame=it->second.frame;p.segment=it->second.segment;p.quantity=it->second.quantity;
    }
    p.id=static_cast<int>(result.parameters.size());ids.emplace(values,p.id);
    p.ambient_size=problem.ParameterBlockSize(values);
    p.tangent_size=problem.ParameterBlockTangentSize(values);
    p.constant=problem.IsParameterBlockConstant(values);
    const auto* manifold=problem.GetManifold(values);
    p.manifold_type=manifold?typeid(*manifold).name():"";
    p.values.assign(values,values+p.ambient_size);
    for(int j=0;j<p.ambient_size;++j){
      p.lower_bounds.push_back(problem.GetParameterLowerBound(values,j));
      p.upper_bounds.push_back(problem.GetParameterUpperBound(values,j));
    }
    result.parameters.push_back(std::move(p));
  }
  std::vector<ceres::ResidualBlockId> residuals;
  problem.GetResidualBlocks(&residuals);
  for(const auto block:residuals){
    ResidualInspection r;r.id=static_cast<int>(result.residuals.size());
    if(auto it=purposes.find(block);it!=purposes.end())r.purpose=it->second;
    const auto* cost=problem.GetCostFunctionForResidualBlock(block);
    const auto* loss=problem.GetLossFunctionForResidualBlock(block);
    r.cost_type=typeid(*cost).name();r.loss_type=loss?typeid(*loss).name():"";
    std::vector<double*> connected;
    problem.GetParameterBlocksForResidualBlock(block,&connected);
    for(const auto* p:connected)r.parameter_ids.push_back(ids.at(p));
    r.values.resize(cost->num_residuals());double unmodified_cost;
    if(!problem.EvaluateResidualBlock(block,false,&unmodified_cost,r.values.data(),nullptr) ||
       !problem.EvaluateResidualBlock(block,true,&r.cost,nullptr,nullptr))
      throw std::runtime_error("Ceres residual inspection failed");
    result.residuals.push_back(std::move(r));
  }
  return result;
}
}
