#pragma once

#include <vector>
#include <cstdint>
#include <span>
#include <array>

namespace dbscan
{

constexpr int32_t NDim{2};

// Configuration parameters
struct DBSCANParams {
  std::array<float, NDim> eps; // Maximum distance per dimension
  int32_t minPts;              // Minimum points to form a dense region
  int32_t nThreads;            // Number of threads to use
};

// Clustering result
struct DBSCANResult {
  std::vector<int32_t> labels;
  int32_t nClusters = 0;
  int32_t nNoise = 0;
};

// Flat neighbor list
struct FlatNeighborList {
  std::vector<size_t> indices;
  std::vector<size_t> offsets;
  [[nodiscard]] size_t getSize(size_t i) const
  {
    return offsets[i + 1] - offsets[i];
  }
  [[nodiscard]] std::span<const size_t> getNeighbors(size_t i) const
  {
    return {&indices[offsets[i]], getSize(i)};
  }
};

// Point classification
enum DBSCANLabel : int32_t {
  DB_NOISE = -(1 << 0),
  DB_UNVISITED = -(1 << 1),
  DB_BORDER = -(1 << 2),
  DB_CORE = -(1 << 3),
};

} // namespace dbscan
