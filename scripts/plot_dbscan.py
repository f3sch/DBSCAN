#!/usr/bin/env python3
"""
Advanced DBSCAN visualization with multiple plot types.
Usage: python plot_dbscan_advanced.py [csv_file]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle


def plot_comprehensive(csv_file="dbscan_results.csv"):
    """Create comprehensive visualization with multiple subplots."""

    # Read data
    df = pd.read_csv(csv_file, skiprows=1)
    df.columns = ["x", "y", "label"]
    print(f"Loaded {len(df)} points from {csv_file}")

    x_col = xlabel = "x"
    y_col = ylabel = "y"

    # Separate noise and clusters
    noise = df[df["label"] == -1]
    clusters = df[df["label"] >= 0]
    unique_labels = sorted(clusters["label"].unique())
    n_clusters = len(unique_labels)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # Generate colors
    if n_clusters > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_clusters, 10)))
        if n_clusters > 10:
            extra_colors = plt.cm.tab20(np.linspace(0, 1, n_clusters - 10))
            colors = np.vstack([colors, extra_colors])

    # Plot 1: Main clustering result
    ax1 = plt.subplot(2, 3, (1, 4))
    for idx, label in enumerate(unique_labels):
        cluster_points = clusters[clusters["label"] == label]
        ax1.scatter(
            cluster_points[x_col],
            cluster_points[y_col],
            c=[colors[idx]],
            label=f"Cluster {label}",
            s=60,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )
    if len(noise) > 0:
        ax1.scatter(
            noise[x_col],
            noise[y_col],
            c="lightgray",
            label="Noise",
            s=30,
            alpha=0.5,
            marker="x",
        )
    ax1.set_xlabel(xlabel, fontsize=11)
    ax1.set_ylabel(ylabel, fontsize=11)
    title = "DBSCAN Clustering"
    ax1.set_title(f"{title} Results", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", framealpha=0.9, fontsize=9)

    # Plot 2: Cluster sizes (bar chart)
    ax2 = plt.subplot(2, 3, 2)
    cluster_sizes = [
        len(clusters[clusters["label"] == label]) for label in unique_labels
    ]
    bars = ax2.bar(
        range(n_clusters),
        cluster_sizes,
        color=colors[:n_clusters] if n_clusters > 0 else [],
    )
    ax2.set_xlabel("Cluster ID", fontsize=10)
    ax2.set_ylabel("Number of Points", fontsize=10)
    ax2.set_title("Cluster Size Distribution", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(n_clusters))
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(cluster_sizes) * 0.01,
            str(size),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 3: Cluster centroids
    ax3 = plt.subplot(2, 3, 3)
    centroids = []
    for idx, label in enumerate(unique_labels):
        cluster_points = clusters[clusters["label"] == label]
        centroid_x = cluster_points[x_col].mean()
        centroid_y = cluster_points[y_col].mean()
        centroids.append((centroid_x, centroid_y))

        # Plot cluster points faintly
        ax3.scatter(
            cluster_points[x_col],
            cluster_points[y_col],
            c=[colors[idx]],
            s=20,
            alpha=0.2,
        )

        # Plot centroid
        ax3.scatter(
            centroid_x,
            centroid_y,
            c=[colors[idx]],
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=1.5,
            label=f"C{label}",
            zorder=5,
        )

    if len(noise) > 0:
        ax3.scatter(
            noise[x_col], noise[y_col], c="lightgray", s=15, alpha=0.3, marker="x"
        )

    ax3.set_xlabel(xlabel, fontsize=10)
    ax3.set_ylabel(ylabel, fontsize=10)
    ax3.set_title("Cluster Centroids", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", framealpha=0.9, fontsize=8)

    # Plot 4: Density heatmap
    ax4 = plt.subplot(2, 3, 5)
    x_bins = np.linspace(df[x_col].min(), df[x_col].max(), 40)
    y_bins = np.linspace(df[y_col].min(), df[y_col].max(), 40)

    # Create 2D histogram for all points
    H, xedges, yedges = np.histogram2d(df[x_col], df[y_col], bins=[x_bins, y_bins])

    im = ax4.imshow(
        H.T,
        origin="lower",
        cmap="YlOrRd",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        interpolation="bilinear",
    )
    ax4.set_xlabel(xlabel, fontsize=10)
    ax4.set_ylabel(ylabel, fontsize=10)
    ax4.set_title("Point Density Heatmap", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax4, label="Point Count")

    # Plot 5: Statistics text box
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis("off")

    stats_text = f"""
    CLUSTERING STATISTICS
    {'='*30}

    Total Points: {len(df):,}
    Clusters Found: {n_clusters}
    Noise Points: {len(noise):,} ({100*len(noise)/len(df):.1f}%)

    CLUSTER DETAILS
    {'='*30}
    """

    for label in unique_labels[:10]:  # Show first 10 clusters
        count = len(clusters[clusters["label"] == label])
        pct = 100 * count / len(df)
        cluster_data = clusters[clusters["label"] == label]
        spread = np.sqrt(cluster_data[x_col].var() + cluster_data[y_col].var())
        stats_text += f"\n    Cluster {label}:"
        stats_text += f"\n      Points: {count:,} ({pct:.1f}%)"
        stats_text += f"\n      Spread: {spread:.2f}\n"

    if n_clusters > 10:
        stats_text += f"\n    ... and {n_clusters - 10} more clusters"

    ax5.text(
        0.05,
        0.95,
        stats_text,
        transform=ax5.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.suptitle(
        f"DBSCAN Comprehensive Analysis - {csv_file}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    output_file = csv_file.replace(".csv", "_comprehensive.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved comprehensive plot to: {output_file}")

    plt.show()


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "dbscan_results.csv"

    try:
        plot_comprehensive(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        print("Run the C++ test program first to generate the CSV file")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
