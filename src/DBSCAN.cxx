#include "DBSCAN/DBSCAN.h"
#include "DBSCAN/DBSCANCommon.h"
#include "DBSCAN/DBSCANGrid.h"
#include <tbb/parallel_for.h>
#include <algorithm>
#include <queue>

namespace dbscan
{

DBSCAN::DBSCAN(const DBSCANParams& p) : mParams(p), mDistance(mParams.eps)
{
  mTaskArena.initialize(mParams.nThreads);
}

DBSCANResult DBSCAN::cluster(const float* points, size_t n)
{
  DBSCANResult result;
  result.labels.resize(n, DB_UNCLASSIFIED);
  if (n == 0) {
    return result;
  }

  // Step 1: Find neighbors for all points using grid
  std::vector<std::vector<size_t>> neighbors(n);
  findNeighbors(points, n, neighbors);
  // Step 2: Classify points and form clusters
  classify(n, neighbors, result.labels);
  // Step 3: Count clusters and noise points
  int32_t max_label = *std::ranges::max_element(result.labels);
  result.nClusters = max_label + 1;
  result.nNoise = static_cast<int32_t>(std::count(result.labels.begin(), result.labels.end(), DB_NOISE));

  return result;
}

void DBSCAN::findNeighbors(const float* points, size_t n, std::vector<std::vector<size_t>>& neighbors)
{
  Grid grid(points, n, mParams.eps);
  // DistanceComputer dist_computer(params_.dimensions, params_.eps);

  // Parallel neighbor finding
  mTaskArena.execute([&] {
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t>& range) {
        std::vector<const GridCell*> neighbor_cells;
        std::vector<size_t> candidates;

        for (size_t i = range.begin(); i < range.end(); ++i) {
          const float* query = &points[i * NDim];
          auto coords = grid.getGridCoords(i);
          grid.getNeighborCells(coords, neighbor_cells);
          candidates.clear();
          for (const GridCell* cell : neighbor_cells) {
            for (auto idx : *cell) {
              if (idx != i) {
                candidates.push_back(idx);
              }
            }
          }
          mDistance.computeNeighbors(query, points, candidates, neighbors[i]);
        }
      });
  });
}

void DBSCAN::classify(size_t n, const std::vector<std::vector<size_t>>& neighbors, std::vector<int32_t>& labels)
{
  std::vector<bool> isCore(n, false);
  std::vector<bool> visited(n, false);
  int32_t nextClsIdx{0};

  mTaskArena.execute([&] {
    // Phase 1: Mark core points
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, n),
      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
          if (static_cast<int32_t>(neighbors[i].size()) >= mParams.minPts) {
            isCore[i] = true;
          }
        }
      });

    // Phase 2: Expand clusters from core points
    for (size_t i = 0; i < n; ++i) {
      if (visited[i] || !isCore[i]) {
        continue;
      }
      // Start a new cluster
      int32_t idx = nextClsIdx++;
      expandCluster(i, idx, neighbors, labels, visited);
    }

    // Phase 3: Mark non-visited points as noise
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, n),
      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
          if (!visited[i]) {
            labels[i] = DB_NOISE;
          }
        }
      });
  });
}

void DBSCAN::expandCluster(size_t i, int32_t idx, const std::vector<std::vector<size_t>>& neighbors, std::vector<int32_t>& labels, std::vector<bool>& visited) const
{
  std::queue<size_t> seeds;
  seeds.push(i);
  visited[i] = true;
  labels[i] = idx;

  while (!seeds.empty()) {
    size_t current = seeds.front();
    seeds.pop();

    // If current is a core point, add its neighbors to the cluster
    if (static_cast<int32_t>(neighbors[current].size()) >= mParams.minPts) {
      for (auto neighbor : neighbors[current]) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          labels[neighbor] = idx;
          seeds.push(neighbor);
        } else if (labels[neighbor] == DB_NOISE) {
          // Border point - add to cluster but don't expand
          labels[neighbor] = idx;
        }
      }
    }
  }
}

} // namespace dbscan
