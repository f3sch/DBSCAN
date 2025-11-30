#pragma once

#include <iomanip>
#include <vector>
#include <cstdint>
#include <span>
#include <array>
#include <iostream>

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

// neighbor list
struct NeighborList {
  [[nodiscard]] int32_t getSize(size_t i) const
  {
    return static_cast<int32_t>(neighbors[i].size());
  }
  [[nodiscard]] const auto& getNeighbors(size_t i) const
  {
    return neighbors[i];
  }
  std::vector<std::vector<size_t>> neighbors;
};

// Point classification
enum DBSCANLabel : int32_t {
  DB_NOISE = -(1 << 0),
  DB_UNVISITED = -(1 << 1),
  DB_BORDER = -(1 << 2),
  DB_CORE = -(1 << 3),
};

#define MEASURE_TIMING
#ifdef MEASURE_TIMING
class ScopedTimer
{
  std::string_view name;
  std::chrono::high_resolution_clock::time_point start;

 public:
  explicit ScopedTimer(std::string_view name)
    : name(name), start(std::chrono::high_resolution_clock::now()) {}

  ~ScopedTimer()
  {
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << name << " : " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n";
  }
};
#define SCOPED_TIMER(name) ScopedTimer _timer##__LINE__(name)
#else
#define SCOPED_TIMER(name) ((void)0)
#endif

} // namespace dbscan
