#!/usr/bin/env python3
"""
Drawing Training Configuration

Central configuration file for drawing training parameters.
Change POINTS_PER_EDGE to scale waypoint density.
"""

# =============================================================================
# WAYPOINT CONFIGURATION
# =============================================================================

# Number of waypoints per edge of the triangle
# Total waypoints = POINTS_PER_EDGE * 3 + 1 (for return to start)
# Examples:
#   POINTS_PER_EDGE = 1  → 4 waypoints (3 corners + 1 return)
#   POINTS_PER_EDGE = 3  → 10 waypoints (9 + 1 return)
POINTS_PER_EDGE = 7  # 10 waypoints total

# Shape type
SHAPE_TYPE = 'triangle'

# Computed total waypoints
if SHAPE_TYPE == 'square':
    # 4 edges * points + 1 return
    TOTAL_WAYPOINTS = POINTS_PER_EDGE * 4 + 1  
else:
    # 3 edges * points + 1 return (triangle)
    TOTAL_WAYPOINTS = POINTS_PER_EDGE * 3 + 1

# =============================================================================
# SHAPE PARAMETERS (Dynamic Workspace)
# =============================================================================

# Square size (side length in meters)
SHAPE_SIZE = 0.10  # 10cm sides requested by user

# X-plane (drawing surface - set dynamically by ArUco detection)
X_PLANE = 0.50  # Default ~0.50m (forward from UAV base)
USE_DYNAMIC_WORKSPACE = True  # Enable dynamic centering on detected board

# Shape center position - set dynamically from board detection
# Default fallback if no board detected
# X=0.5 (forward), Y=0.0m (center), Z=0.35m (height of the vertical board)
TRIANGLE_CENTER = (0.50, 0.0, 0.35)  

# Workspace radius (safe drawing area from center)
WORKSPACE_RADIUS = 0.07  # 7cm radius (14cm diameter) to fit 10cm shape

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Waypoint tolerance (distance threshold to consider waypoint reached)
WAYPOINT_TOLERANCE = 0.01  # 1cm tolerance

# Max steps per episode
DEFAULT_MAX_STEPS = 100
MIN_MAX_STEPS = 5  # Minimum for any configuration

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_waypoint_info():
    """Get human-readable waypoint configuration info."""
    return f"{TOTAL_WAYPOINTS} waypoints ({POINTS_PER_EDGE} per edge)"

def validate_config():
    """Validate configuration parameters."""
    assert POINTS_PER_EDGE >= 1, "POINTS_PER_EDGE must be >= 1"
    assert SHAPE_SIZE > 0, "SHAPE_SIZE must be positive"
    assert WAYPOINT_TOLERANCE > 0, "WAYPOINT_TOLERANCE must be positive"
    assert WORKSPACE_RADIUS > 0, "WORKSPACE_RADIUS must be positive"
    print(f"✅ Drawing config validated: {get_waypoint_info()}")
    if USE_DYNAMIC_WORKSPACE:
        print(f"   Dynamic workspace enabled (Y_PLANE from ArUco detection)")

# Auto-validate on import
if __name__ != "__main__":
    validate_config()

