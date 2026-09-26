#pragma once

namespace skellyforge {
inline constexpr double kDefaultLengthProportionScale = 50.; // mm; deviation from target share of total axial length
inline constexpr double kDefaultLengthEqualityScale = 50.; // mm; soft length-difference residual scale
inline constexpr double kDefaultLinkageScale = 10.; // mm, experimental residual scale, not an anatomical bound
inline constexpr double kDefaultLinkageAccelerationScale = 300.; // mm/s^2 in parent local frame
// Numerical policy for the experimental connected sequence solver.
inline constexpr double kDefaultLengthPriorFraction = 0.25;
inline constexpr double kDefaultLengthAccelerationScale = 500.; // mm/s^2
inline constexpr double kAxialLengthMinimumFraction = 0.25;
inline constexpr double kAxialLengthMaximumFraction = 2.;
inline constexpr double kDefaultDisplacementScale = 20.; // mm
inline constexpr double kDefaultDisplacementAccelerationScale = 500.; // mm/s^2
inline constexpr double kDefaultDisplacementBound = 40.; // mm
inline constexpr double kDefaultRestPoseScale = 1.; // radians
inline constexpr double kLineBasisTolerance=1e-6;
inline constexpr double kQuaternionSquaredNormTolerance = 1e-6;
inline constexpr int kChainMaximumIterations = 200;
inline constexpr double kChainFunctionTolerance = 1e-10;
inline constexpr double kChainGradientTolerance = 1e-10;
inline constexpr double kChainParameterTolerance = 1e-10;
}
