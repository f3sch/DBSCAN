#pragma once

#include "DBSCANCommon.h"
#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

namespace dbscan
{

// Grid cell for spatial partitioning
using GridCell = std::vector<size_t>;

//  Grid coordinates
using GridCoord = std::array<int32_t, NDim>;

// Supports different cell sizes per dimension
class Grid
{
 public:
  Grid(const float* points, size_t n, const std::array<float, NDim>& cellSizes)
    : mPoints(points), mNPoints(n), mCellSizes(cellSizes) {}

  void initGrid()
  {
    {
      SCOPED_TIMER("\t\tcomputeBounds");
      computeBounds();
    }
    {
      SCOPED_TIMER("\t\tcomputeGridDimensions");
      computeGridDimensions();
    }
    {
      SCOPED_TIMER("\t\tallocateCells");
      allocateCells();
    }
    {
      SCOPED_TIMER("\t\tassignCells");
      assignCells();
    }
  }

  // Get grid coordinates for a point
  [[nodiscard]] GridCoord getGridCoords(size_t idx) const
  {
    GridCoord coords{};
#pragma unroll(NDim)
    for (size_t d = 0; d < NDim; ++d) {
      float val = mPoints[(idx * NDim) + d];
      coords[d] = static_cast<int32_t>((val - mMinBounds[d]) / mCellSizes[d]);
      coords[d] = std::clamp(coords[d], 0, static_cast<int32_t>(mGridDims[d]) - 1);
    }
    return coords;
  }

  // Get flat index from grid coordinates
  [[nodiscard]] size_t getCellIndex(const GridCoord& coords) const
  {
    int32_t index = 0, stride = 1;
#pragma unroll(NDim)
    for (size_t d = 0; d < NDim; ++d) {
      index += coords[d] * stride;
      stride *= mGridDims[d];
    }
    return static_cast<size_t>(index);
  }

  // Get cell at grid coordinates
  [[nodiscard]] const GridCell* getCell(const GridCoord& coords) const
  {
#pragma unroll(NDim)
    for (size_t d = 0; d < NDim; ++d) {
      if (coords[d] < 0 || coords[d] >= static_cast<int32_t>(mGridDims[d])) {
        return nullptr;
      }
    }
    return &mCells[getCellIndex(coords)];
  }

  // Get neighboring cells (including the cell itself)
  void getNeighborCells(const GridCoord& coords, std::vector<const GridCell*>& neighbors) const
  {
    neighbors.clear();
    neighbors.reserve(static_cast<size_t>(std::pow(3, NDim)));
    GridCoord offset{};
    enumerateNeighborOffsets<0>(coords, offset, neighbors);
  }

 private:
  template <int32_t Dim>
  void enumerateNeighborOffsets(const GridCoord& base, GridCoord& offset, std::vector<const GridCell*>& output) const
  {
    if constexpr (Dim == NDim) {
      GridCoord nbr;
#pragma unroll(NDim)
      for (size_t d = 0; d < NDim; ++d) {
        nbr[d] = base[d] + offset[d];
      }
      const GridCell* cell = getCell(nbr);
      if (cell) {
        output.push_back(cell);
      }
      return;
    } else {
      for (int32_t v = -1; v <= 1; ++v) {
        offset[Dim] = v;
        enumerateNeighborOffsets<Dim + 1>(base, offset, output);
      }
    }
  }

  void computeBounds()
  {
    mMinBounds.fill(std::numeric_limits<float>::max());
    mMaxBounds.fill(std::numeric_limits<float>::lowest());
    for (size_t i = 0; i < mNPoints; ++i) {
#pragma unroll(NDim)
      for (size_t d = 0; d < NDim; ++d) {
        float val = mPoints[(i * NDim) + d];
        mMinBounds[d] = std::min(mMinBounds[d], val);
        mMaxBounds[d] = std::max(mMaxBounds[d], val);
      }
    }
  }

  void computeGridDimensions()
  {
    mGridDims.fill(1);
#pragma unroll(NDim)
    for (size_t d = 0; d < NDim; ++d) {
      float range = mMaxBounds[d] - mMinBounds[d];
      mGridDims[d] = std::max(size_t(1), static_cast<size_t>(std::ceil(range / mCellSizes[d])));
    }
  }

  void allocateCells()
  {
    auto total_cells = static_cast<size_t>(mGridDims[0] * mGridDims[1]);
    mCells.resize(total_cells);
  }

  void assignCells()
  {
    for (size_t i = 0; i < mNPoints; ++i) {
      auto coords = getGridCoords(i);
      size_t cell_idx = getCellIndex(coords);
      mCells[cell_idx].push_back(i);
    }
  }

  const float* mPoints;
  size_t mNPoints;
  std::array<float, NDim> mCellSizes;
  std::array<float, NDim> mMinBounds;
  std::array<float, NDim> mMaxBounds;
  std::array<size_t, NDim> mGridDims;
  std::vector<GridCell> mCells;
};

} // namespace dbscan
