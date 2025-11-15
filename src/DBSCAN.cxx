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
  result.labels.resize(n, DB_UNVISITED);
  if (n == 0) {
    return result;
  }

  // Step 1: Find neighbors for all points using grid
  FlatNeighborList neighbors;
  findNeighbors(points, n, neighbors);
  // Step 2: Classify points and form clusters
  classify(n, neighbors, result.labels);
  // Step 3: Count clusters and noise points
  int32_t max_label = *std::ranges::max_element(result.labels);
  result.nClusters = max_label + 1;
  result.nNoise = static_cast<int32_t>(std::count(result.labels.begin(), result.labels.end(), DB_NOISE));

  return result;
}

void DBSCAN::findNeighbors(const float* points, size_t n, FlatNeighborList& neighbors)
{
  Grid grid(points, n, mParams.eps);
  std::vector<std::vector<size_t>> temp(n);

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
          mDistance.computeNeighbors(query, points, candidates, temp[i]);
        }
      });
  });
  // flatten
  neighbors.offsets.resize(n + 1);
  neighbors.offsets[0] = 0;
  for (size_t i = 0; i < n; ++i) {
    neighbors.offsets[i + 1] = neighbors.offsets[i] + temp[i].size();
  }
  neighbors.indices.resize(neighbors.offsets[n]);
  for (size_t i = 0; i < n; ++i) {
    std::copy(temp[i].begin(), temp[i].end(), &neighbors.indices[neighbors.offsets[i]]);
  }
}

void DBSCAN::classify(size_t n, const FlatNeighborList& neighbors, std::vector<int32_t>& labels)
{
  std::vector<bool> isCore(n, false);
  int32_t nextClsIdx{0};

  mTaskArena.execute([&] {
    // Phase 1: Mark core points
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, n),
      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
          if (static_cast<int32_t>(neighbors.getSize(i)) >= mParams.minPts) {
            isCore[i] = true;
          }
        }
      });

    // Phase 2: Expand clusters from core points
    for (size_t i = 0; i < n; ++i) {
      if (labels[i] != DB_UNVISITED || !isCore[i]) {
        continue;
      }
      // Start a new cluster
      int32_t idx = nextClsIdx++;
      expandCluster(i, idx, neighbors, labels);
    }

    // Phase 3: Mark non-visited points as noise
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, n),
      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
          if (labels[i] == DB_UNVISITED) {
            labels[i] = DB_NOISE;
          }
        }
      });
  });
}

void DBSCAN::expandCluster(size_t i, int32_t idx, const FlatNeighborList& neighbors, std::vector<int32_t>& labels) const
{
  std::queue<size_t> seeds;
  seeds.push(i);
  labels[i] = idx;

  while (!seeds.empty()) {
    size_t current = seeds.front();
    seeds.pop();

    // If current is a core point, add its neighbors to the cluster
    if (static_cast<int32_t>(neighbors.getSize(current)) >= mParams.minPts) {
      for (auto neighbor : neighbors.getNeighbors(current)) {
        if (labels[neighbor] == DB_UNVISITED) {
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
