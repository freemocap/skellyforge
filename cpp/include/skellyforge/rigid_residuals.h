#pragma once
#include "skellyforge/rigid_fit.h"
#include <ceres/rotation.h>
namespace skellyforge {
struct LengthEqualityResidual {
  double scale; // sqrt(time_weight) / length_difference_scale_mm
  template <typename T> bool operator()(const T* a,const T* b,T* residual)const{
    residual[0]=(a[0]-b[0])*T(scale);return true;
  }
};
// XYZ displacement in the parent segment's local frame, measured in mm.
struct LinkageDisplacementPriorResidual {
  double scale;
  template <typename T> bool operator()(const T* displacement,T* residual)const{
    for(int k=0;k<3;++k)residual[k]=displacement[k]*T(scale);
    return true;
  }
};
struct LengthPriorResidual {
  // Multipliers: sqrt(time_weight) / (fraction * reference_length).
  double reference,shortening_scale,lengthening_scale;
  template <typename T> bool operator()(const T* length,T* residual)const{
    const T change=length[0]-T(reference);
    const double scale=change<T(0)?shortening_scale:lengthening_scale;
    // The squared objective and its gradient are continuous at the reference.
    residual[0]=change*T(scale);return true;
  }
};
struct DisplacementPriorResidual {
  double scale;
  template <typename T> bool operator()(const T* displacement,T* residual)const{
    residual[0]=displacement[0]*T(scale);return true;
  }
};
struct DisplacementAccelerationResidual {
  double before,after,scale;
  template <typename T> bool operator()(const T* a,const T* b,const T* c,T* residual)const{
    residual[0]=((c[0]-b[0])/T(after)-(b[0]-a[0])/T(before))*T(scale);return true;
  }
};
// Principal relative-quaternion logarithm; poses remain unit wxyz quaternions.
struct RelativePoseResidual {
  std::array<double,4> reference;
  double scale;
  template <typename T> bool operator()(const T* parent,const T* child,T* residual)const{
    T inverse_parent[4]={parent[0],-parent[1],-parent[2],-parent[3]},relative[4],error[4];
    T inverse_reference[4]={T(reference[0]),T(-reference[1]),T(-reference[2]),T(-reference[3])};
    ceres::QuaternionProduct(inverse_parent,child,relative);
    ceres::QuaternionProduct(inverse_reference,relative,error);
    if(error[0]<T(0))for(auto& value:error)value=-value;
    ceres::QuaternionToAngleAxis(error,residual);
    for(int k=0;k<3;++k)residual[k]*=T(scale);
    return true;
  }
};
struct LandmarkResidual {
  Vec3 local, observed;
  double scale = 1.0;
  template <typename T> bool operator()(const T* q, const T* t, T* r) const {
    T point[3] = {T(local[0]),T(local[1]),T(local[2])};
    ceres::QuaternionRotatePoint(q, point, r);
    for(int i=0;i<3;++i) r[i] = (r[i]+t[i]-T(observed[i]))*T(scale);
    return true;
  }
};
struct TranslationMotionResidual {
  double scale;
  template <typename T> bool operator()(const T* a,const T* b,T* r) const {
    for(int i=0;i<3;++i) r[i]=(b[i]-a[i])*T(scale);
    return true;
  }
};
struct QuaternionMotionResidual {
  double scale;
  template <typename T> bool operator()(const T* a,const T* b,T* r) const {
    T inverse[4]={a[0],-a[1],-a[2],-a[3]}, relative[4];
    ceres::QuaternionProduct(inverse,b,relative);
    // Squared norm is sign-invariant: 4 sin²(theta/2), not theta².
    for(int i=0;i<3;++i) r[i]=T(2*scale)*relative[i+1];
    return true;
  }
};
// Central difference of interval velocities; poses remain unit wxyz quaternions.
struct TranslationAccelerationResidual {
  double before, after, scale;
  template <typename T> bool operator()(const T* a,const T* b,const T* c,T* r) const {
    for(int i=0;i<3;++i)
      r[i]=((c[i]-b[i])/T(after)-(b[i]-a[i])/T(before))*T(scale);
    return true;
  }
};
struct QuaternionAccelerationResidual {
  double before, after, scale;
  template <typename T> void world_velocity(const T* a,const T* b,double dt,T* v) const {
    T inverse[4]={a[0],-a[1],-a[2],-a[3]}, relative[4];
    // b * inverse(a) expresses both intervals in the same world basis.
    ceres::QuaternionProduct(b,inverse,relative);
    // Principal logarithm: invariant under either input quaternion changing sign.
    if(relative[0]<T(0)) for(auto& component:relative) component=-component;
    ceres::QuaternionToAngleAxis(relative,v);
    for(int i=0;i<3;++i) v[i]/=T(dt);
  }
  template <typename T> bool operator()(const T* a,const T* b,const T* c,T* r) const {
    T previous[3],next[3];
    world_velocity(a,b,before,previous);world_velocity(b,c,after,next);
    for(int i=0;i<3;++i) r[i]=(next[i]-previous[i])*T(scale);
    return true;
  }
};
}
