#include "DBSCAN/DBSCAN.h"
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <iomanip>

using namespace dbscan;

// Export results to CSV for visualization
void export_to_csv(const std::vector<float>& points,
                   const DBSCANResult& result,
                   const std::string& filename)
{
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Write data
  size_t n_points = result.labels.size();
  for (size_t i = 0; i < n_points; ++i) {
    for (size_t d{0}; d < NDim; ++d) {
      file << points[i * NDim + d] << ",";
    }
    file << result.labels[i] << "\n";
  }

  file.close();
  std::cout << "Exported results to: " << filename << std::endl;
}

// Generate synthetic spatiotemporal clustered data with noise
// Dimension 0: space coordinate (meters)
// Dimension 1: time coordinate (seconds)
std::vector<float> generate_test_data(size_t n_points, unsigned int seed = 42)
{
  std::mt19937 gen(seed);
  std::normal_distribution<float> space_dist(0.0f, 5.0f);
  std::normal_distribution<float> time_dist(0.0f, 2.0f);
  std::uniform_real_distribution<float> noise_space(-20.0f, 120.0f);
  std::uniform_real_distribution<float> noise_time(-10.0f, 110.0f);

  std::vector<float> points;
  points.reserve(n_points * NDim);

  // Each cluster represents events at different locations and times
  std::array<std::array<float, 2>, 3> cluster_centers = {{
    {0.0f, 10.0f},  // Cluster 0:
    {50.0f, 50.0f}, // Cluster 1:
    {100.0f, 90.0f} // Cluster 2:
  }};

  // Calculate how many noise points to add (50% of total)
  size_t n_noise = n_points / 2;
  size_t n_cluster_points = n_points - n_noise;

  std::cout << "Generating " << n_cluster_points << " cluster points and "
            << n_noise << " noise points" << std::endl;

  // Generate cluster points
  for (size_t i = 0; i < n_cluster_points; ++i) {
    size_t cluster = i % 3;
    // Space coordinate (dimension 0)
    points.push_back(cluster_centers[cluster][0] + space_dist(gen));
    // Time coordinate (dimension 1)
    points.push_back(cluster_centers[cluster][1] + time_dist(gen));
  }

  // Generate noise points (uniformly distributed in space-time)
  for (size_t i = 0; i < n_noise; ++i) {
    points.push_back(noise_space(gen)); // Random space
    points.push_back(noise_time(gen));  // Random time
  }

  return points;
}

// Print results summary
void print_results(const DBSCANResult& result, double elapsed_ms)
{
  std::cout << "\n=== DBSCAN Results ===" << std::endl;
  std::cout << "Execution time: " << std::fixed << std::setprecision(2)
            << elapsed_ms << " ms" << std::endl;
  std::cout << "Number of clusters: " << result.nClusters << std::endl;
  std::cout << "Noise points: " << result.nNoise << std::endl;

  // Count points per cluster
  if (result.nClusters > 0) {
    std::vector<int> cluster_sizes(size_t(result.nClusters), 0);
    for (int32_t label : result.labels) {
      if (label >= 0) {
        cluster_sizes[size_t(label)]++;
      }
    }

    std::cout << "\nCluster sizes:" << std::endl;
    for (int i = 0; i < result.nClusters; ++i) {
      std::cout << "  Cluster " << i << ": " << cluster_sizes[size_t(i)] << " points" << std::endl;
    }
  }
}

int main()
{
  std::cout << "DBSCAN CPU Implementation Test" << std::endl;
  std::cout << "================================================" << std::endl;

  const size_t n_points = 100'000;
  const float eps_space = 0.6f;
  const float eps_time = 0.6f;
  const int min_pts = 100;

  std::cout << "\nTest configuration:" << std::endl;
  std::cout << "  Points: " << n_points << std::endl;
  std::cout << "  Epsilon (space): " << eps_space << '\n';
  std::cout << "  Epsilon (time): " << eps_time << '\n';
  std::cout << "  Min points: " << min_pts << std::endl;

  // Generate test data
  std::cout << "\nGenerating spatiotemporal test data..." << std::endl;
  auto points = generate_test_data(n_points);

  // Configure DBSCAN with separate epsilons
  DBSCANParams params({eps_space, eps_time}, min_pts);

  // Run clustering
  std::cout << "Running DBSCAN clustering..." << std::endl;
  DBSCAN dbscan(params);

  auto start = std::chrono::high_resolution_clock::now();
  auto result = dbscan.cluster(points.data(), n_points);
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

  // Print results
  print_results(result, elapsed_ms);

  // Export to CSV for visualization
  export_to_csv(points, result, "dbscan_results.csv");

  std::cout << "\nTest completed successfully!" << std::endl;
  std::cout << "\nVisualization:" << std::endl;
  std::cout << "  Run: python ../scripts/plot_dbscan.py" << std::endl;
  return 0;
}
