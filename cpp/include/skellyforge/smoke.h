#pragma once
#include <string>
namespace skellyforge {
struct SmokeResult { double value; double final_cost; bool converged; };
SmokeResult solve_scalar(double initial, double target);
std::string ceres_version();
}
