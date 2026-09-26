#include "skellyforge/smoke.h"
#include <cmath>
int main() {
  const auto result = skellyforge::solve_scalar(-12.0, 7.0);
  return result.converged && std::abs(result.value - 7.0) < 1e-6 ? 0 : 1;
}
