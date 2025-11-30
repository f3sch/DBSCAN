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
  void findNeighbors(const float*, size_t n, NeighborList& neighbors);
  void classify(size_t n, const NeighborList& neighbors, std::vector<int32_t>& labels) const;

  DBSCANParams mParams;
  DBSCANDistance mDistance;
  tbb::task_arena mTaskArena;
};

} // namespace dbscan
