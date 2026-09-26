#include "skellyforge/rigid_residuals.h"
#include <ceres/ceres.h>
#include <cmath>

int main(){
  const std::array<double,3> fractions{18./44.3,20./44.3,6.3/44.3};
  ceres::AutoDiffCostFunction<skellyforge::LengthProportionResidual,3,1,1,1> cost(
    new skellyforge::LengthProportionResidual{fractions,1./50.});
  for(double total:{0.,300.,700.}){
    double lengths[3];for(int k=0;k<3;++k)lengths[k]=total*fractions[k];
    const double* blocks[]{&lengths[0],&lengths[1],&lengths[2]};
    double residual[3],j0[3],j1[3],j2[3];double* jacobians[]{j0,j1,j2};
    if(!cost.Evaluate(blocks,residual,jacobians))return 1;
    for(int row=0;row<3;++row){
      if(std::abs(residual[row])>1e-12)return 2;
      double null_projection=0.;
      for(int column=0;column<3;++column){
        const double expected=((row==column?1.:0.)-fractions[row])/50.;
        if(std::abs(jacobians[column][row]-expected)>1e-12)return 3;
        null_projection+=jacobians[column][row]*fractions[column];
      }
      if(std::abs(null_projection)>1e-12)return 4;
    }
  }
  return 0;
}
