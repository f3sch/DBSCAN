#include "DBSCANCommon.h"
#include "DBSCANDistance.h"
#include <tbb/task_arena.h>

namespace dbscan
{

class DBSCAN
{
 public:
  DBSCAN(const DBSCANParams& p);

  DBSCANResult cluster(const float* points, size_t n);

 private:
  void findNeighbors(const float*, size_t n, std::vector<std::vector<size_t>>& neighbors);
  void classify(size_t n, const std::vector<std::vector<size_t>>& neighbors, std::vector<int32_t>& labels);
  void expandCluster(size_t i, int32_t idx, const std::vector<std::vector<size_t>>& neighbors, std::vector<int32_t>& labels, std::vector<bool>& visited) const;

  DBSCANParams mParams;
  DBSCANDistance mDistance;
  tbb::task_arena mTaskArena;
};

} // namespace dbscan
