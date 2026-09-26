#include "skellyforge/rigid_residuals.h"
#include <ceres/ceres.h>
#include <cmath>
#include <stdexcept>

int main() {
  constexpr double reference=100.,shortening_fraction=.5,lengthening_fraction=.25;
  ceres::AutoDiffCostFunction<skellyforge::LengthPriorResidual,1,1> cost(
    new skellyforge::LengthPriorResidual{reference,1/(shortening_fraction*reference),1/(lengthening_fraction*reference)});
  auto evaluate=[&](double length){
    const double* blocks[]={&length};double residual,derivative;double* jacobians[]={&derivative};
    if(!cost.Evaluate(blocks,&residual,jacobians))throw std::runtime_error("Residual evaluation failed");
    return std::array<double,2>{residual,derivative};
  };
  const auto compressed=evaluate(80.),extended=evaluate(120.),neutral=evaluate(reference);
  if(std::abs(extended[0]*extended[0]-4*compressed[0]*compressed[0])>1e-12)return 1;
  if(neutral[0]!=0 || neutral[0]*neutral[1]!=0)return 2;
  constexpr double step=1e-5;
  for(double length:{80.,120.}){
    const double numeric=(evaluate(length+step)[0]-evaluate(length-step)[0])/(2*step);
    if(std::abs(numeric-evaluate(length)[1])>1e-9)return 3;
  }
  return 0;
}
