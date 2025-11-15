#pragma once

#include "DBSCAN/DBSCANCommon.h"
#include <cmath>

namespace dbscan
{

class DBSCANDistance
{
  using EPS = decltype(DBSCANParams::eps);

 public:
  DBSCANDistance(const EPS& eps) : mEps(eps) {}

  // Check if two points are neighbors using L-infinity distance
  // Returns true if ALL dimensions are within their respective thresholds
  bool areNeighbors(const float* p1, const float* p2) const
  {
#pragma unroll(NDim)
    for (size_t d{0}; d < NDim; ++d) {
      const float diff = std::abs(p1[d] - p2[d]);
      if (diff > mEps[d]) {
        return false;
      }
    }
    return true;
  }

  // Batch compute
  void computeNeighbors(const float* query, const float* points, const std::vector<size_t>& candidates, std::vector<size_t>& neighbors) const
  {
    neighbors.clear();
    neighbors.reserve(candidates.size());
    for (auto idx : candidates) {
      const float* p = &points[idx * NDim];
      if (areNeighbors(query, p)) {
        neighbors.push_back(idx);
      }
    }
  }

 private:
  EPS mEps;
};

} // namespace dbscan
