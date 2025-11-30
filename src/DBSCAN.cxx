#include "DBSCAN/DBSCAN.h"
#include "DBSCAN/DBSCANCommon.h"
#include "DBSCAN/DBSCANGrid.h"
#include <tbb/parallel_for.h>
#include <algorithm>
#include <queue>
#include <chrono>

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
  NeighborList neighbors;
  {
    SCOPED_TIMER("findNeighbors");
    findNeighbors(points, n, neighbors);
  }
  // Step 2: Classify points and form clusters
  {
    SCOPED_TIMER("Classification");
    classify(n, neighbors, result.labels);
  }
  // Step 3: Count clusters and noise points
  {
    SCOPED_TIMER("Assignment");
    int32_t max_label = *std::ranges::max_element(result.labels);
    result.nClusters = max_label + 1;
    result.nNoise = static_cast<int32_t>(std::count(result.labels.begin(), result.labels.end(), DB_NOISE));
  }

  return result;
}

void DBSCAN::findNeighbors(const float* points, size_t n, NeighborList& neighbors)
{
  Grid grid(points, n, mParams.eps);
  {
    SCOPED_TIMER("\tinit grid");
    grid.initGrid();
  }

  // Parallel neighbor finding
  mTaskArena.execute([&] {
    SCOPED_TIMER("\tneighbor finding");
    neighbors.neighbors.resize(n);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t>& range) {
      std::vector<const GridCell*> neighbor_cells;
      neighbor_cells.reserve(NDim * NDim);

      for (size_t i = range.begin(); i < range.end(); ++i) {
        const float* query = &points[i * NDim];
        auto coords = grid.getGridCoords(i);
        grid.getNeighborCells(coords, neighbor_cells);

        neighbors.neighbors[i].clear();
        for (const GridCell* cell : neighbor_cells) {
          for (auto idx : *cell) {
            if (idx != i && mDistance.areNeighbors(query, &points[idx * NDim])) {
              neighbors.neighbors[i].push_back(idx);
            }
          }
        }
      }
    });
  });
}

namespace
{
inline size_t find(std::vector<std::atomic<size_t>>& parent, size_t x)
{
  while (true) {
    size_t p = parent[x].load(std::memory_order_acquire);
    if (p == x) {
      return x;
    }

    // Path halving
    size_t gp = parent[p].load(std::memory_order_acquire);
    if (p == gp) {
      return p;
    }

    parent[x].compare_exchange_weak(p, gp, std::memory_order_release);
    x = p;
  }
}

inline void unite(std::vector<std::atomic<size_t>>& parent, size_t x, size_t y)
{
  while (true) {
    x = find(parent, x);
    y = find(parent, y);
    if (x == y) {
      return;
    }

    if (x > y) {
      std::swap(x, y); // Smaller root wins
    }

    size_t expected = y;
    if (parent[y].compare_exchange_strong(expected, x, std::memory_order_acq_rel)) {
      return;
    }
  }
}
} // namespace

void DBSCAN::classify(size_t n, const NeighborList& neighbors, std::vector<int32_t>& labels) const
{
  std::vector<std::atomic<size_t>> parent(n);
  std::vector<bool> isCore(n, false);

  // Phase 1: Initialize + mark core points (already parallel)
  {
    SCOPED_TIMER("\tinit core points");
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                      [&](const auto& range) {
                        for (size_t i = range.begin(); i < range.end(); ++i) {
                          parent[i].store(i, std::memory_order_relaxed);
                          isCore[i] = neighbors.getSize(i) >= mParams.minPts;
                        }
                      });
  }

  // Phase 2: Parallel union of core-to-neighbor edges
  {
    SCOPED_TIMER("\tunion");
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                      [&](const auto& range) {
                        for (size_t i = range.begin(); i < range.end(); ++i) {
                          if (!isCore[i]) {
                            continue;
                          }

                          for (size_t neighbor : neighbors.getNeighbors(i)) {
                            // Union core point with all neighbors
                            unite(parent, i, neighbor);
                          }
                        }
                      });
  }

  // Phase 3: Path compression + assign labels
  {
    SCOPED_TIMER("\tpath compression");
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                      [&](const auto& range) {
                        for (size_t i = range.begin(); i < range.end(); ++i) {
                          size_t root = find(parent, i);
                          if (isCore[root]) {
                            labels[i] = static_cast<int32_t>(root); // Use root as cluster ID (remap later)
                          } else {
                            labels[i] = DB_NOISE;
                          }
                        }
                      });
  }
}

} // namespace dbscan
