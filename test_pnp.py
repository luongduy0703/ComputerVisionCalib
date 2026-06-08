import cv2
import numpy as np

# 3D points
PEN_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],         # Point 0: Tip
    [0.0, 145.0, 0.0],       # Point 1: Tail
    [-8.0, 100.0, 0.0],      # Point 2: Left Cap
    [8.0, 100.0, 0.0]        # Point 3: Right Cap
], dtype=np.float32)

# Camera matrix
fx, fy = 600.0, 600.0
cx, cy = 320.0, 240.0
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# Let's simulate some 2D keypoints
# Let's say the pen is at Z = 300mm, horizontal, centered at (320, 240)
# Projection of (0, 0, 300) -> (320, 240)
# Projection of (0, 145, 300):
# X_px = 320 + 0 * 600 / 300 = 320
# Y_px = 240 + 145 * 600 / 300 = 240 + 290 = 530
# Let's check with standard projectPoints:
rvec_sim = np.array([0, 0, 0], dtype=np.float32) # no rotation
tvec_sim = np.array([0, 0, 300], dtype=np.float32) # Z = 300mm

projected, _ = cv2.projectPoints(PEN_3D_POINTS, rvec_sim, tvec_sim, camera_matrix, dist_coeffs)
print("Simulated 2D points (no rotation, at Z=300):")
print(projected.squeeze())

# Let's run solvePnP on these simulated points
success, rvec, tvec = cv2.solvePnP(PEN_3D_POINTS, projected, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
print(f"\nSolvePnP success: {success}")
print(f"Calculated tvec: {tvec.squeeze()}")

# Now, let's simulate the case where the points are misordered!
# What if the 2D points are ordered differently?
# For example, if we swap Point 2 and Point 3, or if the order is completely wrong.
# Let's see what happens.
