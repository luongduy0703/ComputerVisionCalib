"""
pen_geometry.py — AeroScript Phase 1: 3D Geometric Modeling of the Paint Pen

Defines the physical 3D object points of the paint pen for use with
cv2.solvePnP in the markerless tracking pipeline. Includes a matplotlib
3D visualization for verifying caliper measurements.

Coordinate Convention:
    - Origin: Pen tip  [0, 0, 0]
    - Y-axis:  Along the pen body (tip → tail is +Y)
    - X-axis:  Lateral (left/right cap edge spread)
    - Z-axis:  Completes a right-handed coordinate system (perpendicular to pen plane)

Point Order (strict, matches YOLOv8-Pose keypoint indices):
    0: Tip         — The writing/painting tip of the pen
    1: Tail        — The rear end of the pen
    2: Left_Edge   — Left lateral edge of cap/ridge (−X direction)
    3: Right_Edge  — Right lateral edge of cap/ridge (+X direction)

Usage:
    python pen_geometry.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURABLE CONSTANTS — Update these after measuring the real pen (mm)
# ============================================================================

PEN_LENGTH: float = 64.0
"""Total length from the tip to the tail of the pen (mm)."""

CAP_DISTANCE_FROM_TIP: float = 44.0
"""Distance along the pen body from the tip to the cap/ridge cross-section (mm)."""

CAP_WIDTH: float = 23.0
"""Total width across the cap/ridge (mm). Each side offsets by CAP_WIDTH / 2."""


# ============================================================================
# 3D Object Points
# ============================================================================

def get_pen_object_points() -> np.ndarray:
    """Return the 4 pen keypoints as a (4, 3) float32 array in millimeters.

    Coordinate system:
        Tip is at the origin.  Y runs along the pen body toward the tail.
        X spans the lateral cap width.  Z is zero (all points coplanar).

    Returns:
        np.ndarray: shape (4, 3), dtype np.float32
            Row 0 — Tip        : [0, 0, 0]
            Row 1 — Tail       : [0, PEN_LENGTH, 0]
            Row 2 — Left_Edge  : [-CAP_WIDTH/2, CAP_DISTANCE_FROM_TIP, 0]
            Row 3 — Right_Edge : [+CAP_WIDTH/2, CAP_DISTANCE_FROM_TIP, 0]
    """
    half_cap = CAP_WIDTH / 2.0

    object_points = np.array([
        [0.0,      0.0,                    0.0],   # 0: Tip
        [0.0,      PEN_LENGTH,             0.0],   # 1: Tail
        [-half_cap, CAP_DISTANCE_FROM_TIP, 0.0],   # 2: Left_Edge
        [half_cap,  CAP_DISTANCE_FROM_TIP, 0.0],   # 3: Right_Edge
    ], dtype=np.float32)

    return object_points


# ============================================================================
# Visualization
# ============================================================================

def plot_pen_geometry() -> None:
    """Render an interactive 3D scatter + wireframe plot of the pen geometry.

    If 3D projection is not available (e.g. system matplotlib installation issue),
    falls back gracefully to a 2D X-Y projection.
    """
    pts = get_pen_object_points()

    labels = ["Tip", "Tail", "Left_Edge", "Right_Edge"]
    colors = ["#FF4136", "#0074D9", "#2ECC40", "#FF851B"]  # red, blue, green, orange

    # --- Wireframe edges: (start_index, end_index) ---
    edges = [
        (0, 1),  # Tip → Tail  (pen body axis)
        (0, 2),  # Tip → Left_Edge
        (0, 3),  # Tip → Right_Edge
        (2, 3),  # Left_Edge → Right_Edge  (cap cross-section)
        (1, 2),  # Tail → Left_Edge  (outline)
        (1, 3),  # Tail → Right_Edge (outline)
    ]

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")  # dark background

    # Try setting up a 3D plot
    try:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#16213e")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Draw wireframe edges
        for i, j in edges:
            ax.plot(
                [pts[i, 0], pts[j, 0]],
                [pts[i, 1], pts[j, 1]],
                [pts[i, 2], pts[j, 2]],
                color="#a0c4ff", linewidth=1.5, alpha=0.7,
            )

        # Draw keypoints
        for idx, (label, color) in enumerate(zip(labels, colors)):
            x, y, z = pts[idx]
            ax.scatter(x, y, z, color=color, s=120, edgecolors="white",
                       linewidths=0.8, zorder=5, depthshade=False)
            ax.text(
                x, y, z,
                f"  {label}\n  ({x:.1f}, {y:.1f}, {z:.1f})",
                fontsize=9, color=color, fontweight="bold",
                ha="left", va="bottom",
            )

        # Axis labels & title
        ax.set_xlabel("X  (mm)", color="white", fontsize=10, labelpad=10)
        ax.set_ylabel("Y  (mm)", color="white", fontsize=10, labelpad=10)
        ax.set_zlabel("Z  (mm)", color="white", fontsize=10, labelpad=10)
        ax.set_title(
            "AeroScript — Pen 3D Geometry (Phase 1)\n"
            f"Length={PEN_LENGTH} mm | Cap@{CAP_DISTANCE_FROM_TIP} mm | Width={CAP_WIDTH} mm",
            color="white", fontsize=12, fontweight="bold", pad=18,
        )

        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.tick_params(axis="z", colors="white")

        # Set equal aspect ratio so geometry isn't distorted
        _set_equal_aspect_3d(ax, pts)

        # Legend (manual)
        for label, color in zip(labels, colors):
            ax.plot([], [], "o", color=color, label=label, markersize=8)
        ax.legend(
            loc="upper left", fontsize=9, framealpha=0.6,
            facecolor="#16213e", edgecolor="white", labelcolor="white",
        )

    except (ValueError, KeyError, TypeError, Exception) as e:
        print(f"\n[WARNING] 3D plotting failed (Matplotlib issue: {e}).")
        print("Falling back to 2D X-Y projection (since Z is 0 for all points).")
        
        ax = fig.add_subplot(111)
        ax.set_facecolor("#16213e")
        ax.grid(True, color="#2c3e50", linestyle="--", linewidth=0.5)

        # Draw wireframe edges in 2D
        for i, j in edges:
            ax.plot(
                [pts[i, 0], pts[j, 0]],
                [pts[i, 1], pts[j, 1]],
                color="#a0c4ff", linewidth=1.5, alpha=0.7,
            )

        # Draw keypoints in 2D
        for idx, (label, color) in enumerate(zip(labels, colors)):
            x, y, _ = pts[idx]
            ax.scatter(x, y, color=color, s=120, edgecolors="white", linewidths=0.8, zorder=5)
            ax.text(
                x, y,
                f"  {label} ({x:.1f}, {y:.1f})",
                fontsize=9, color=color, fontweight="bold",
                ha="left", va="bottom",
            )

        # Axis labels & title
        ax.set_xlabel("X  (mm)", color="white", fontsize=10)
        ax.set_ylabel("Y  (mm)", color="white", fontsize=10)
        ax.set_title(
            "AeroScript — Pen 2D Geometry Fallback (Phase 1)\n"
            f"Length={PEN_LENGTH} mm | Cap@{CAP_DISTANCE_FROM_TIP} mm | Width={CAP_WIDTH} mm",
            color="white", fontsize=12, fontweight="bold", pad=18,
        )

        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        
        # Equal aspect
        ax.set_aspect("equal", adjustable="box")
        
        # Adjust margins/limits to show all points nicely
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        padding_x = max((x_max - x_min) * 0.2, 5.0)
        padding_y = max((y_max - y_min) * 0.1, 5.0)
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)

        # Legend
        for label, color in zip(labels, colors):
            ax.plot([], [], "o", color=color, label=label, markersize=8)
        ax.legend(
            loc="upper left", fontsize=9, framealpha=0.6,
            facecolor="#16213e", edgecolor="white", labelcolor="white",
        )

    plt.tight_layout()
    plt.show()


def _set_equal_aspect_3d(ax, points: np.ndarray) -> None:
    """Force equal scaling on all three axes so the pen shape isn't skewed."""
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  AeroScript — Pen 3D Geometry (Phase 1)")
    print("=" * 60)

    obj_pts = get_pen_object_points()

    print(f"\n  Pen Length           : {PEN_LENGTH} mm")
    print(f"  Cap Distance from Tip: {CAP_DISTANCE_FROM_TIP} mm")
    print(f"  Cap Width            : {CAP_WIDTH} mm")
    print(f"\n  Object Points (mm):")
    print(f"  {'Index':<6} {'Label':<12} {'X':>8} {'Y':>8} {'Z':>8}")
    print(f"  {'-'*42}")

    point_labels = ["Tip", "Tail", "Left_Edge", "Right_Edge"]
    for i, label in enumerate(point_labels):
        x, y, z = obj_pts[i]
        print(f"  {i:<6} {label:<12} {x:>8.1f} {y:>8.1f} {z:>8.1f}")

    print(f"\n  dtype : {obj_pts.dtype}")
    print(f"  shape : {obj_pts.shape}")
    print("=" * 60)

    # Launch interactive 3D plot
    plot_pen_geometry()
