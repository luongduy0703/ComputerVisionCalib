#!/usr/bin/env python3
"""
Board Transform Utility for Visual Servoing

Builds the transform pipeline from board-local coordinates to base_link:
  board-local (x, y, 0, 1) → T_vision (board→camera) → TF2 (camera→base_link)

The T_vision matrix comes from ArUco solvePnP (rotation + translation).
The TF2 transform comes from the robot URDF (camera_link → base_link).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
import tf2_ros
import rclpy
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped


class BoardTransform:
    """Transforms points from board-local 2D to base_link 3D frame."""
    
    def __init__(self, tf_buffer: tf2_ros.Buffer):
        self.tf_buffer = tf_buffer
        
        # T_vision: 4×4 board→camera (from ArUco PoseStamped)
        self.T_vision = None
        # T_tf2: 4×4 camera_link→base_link (from TF2)
        self.T_tf2 = None
        # Combined: T_combined = T_tf2 @ T_vision
        self.T_combined = None
        
        self.locked = False
    
    def update_from_pose(self, pose_msg: PoseStamped) -> bool:
        """
        Build transform from ArUco PoseStamped position + TF2.
        
        Uses ArUco only for POSITION (where the board center is).
        Uses a CLEAN rotation that maps board-local axes correctly:
          - Board X (left/right) → base_link X
          - Board Y (up/down)    → base_link -Z  (robot is flipped 180° on X)
          - Board Z (depth)      → base_link +Y  (toward camera in flipped frame)
        
        This avoids the tilt errors from ArUco solvePnP rotation noise.
        """
        if self.locked:
            return True
        
        # Build T_vision (4×4 board→camera) from pose quaternion + translation
        p = pose_msg.pose.position
        q = pose_msg.pose.orientation
        
        # Use scipy for robust quaternion to matrix conversion
        q_list = [q.x, q.y, q.z, q.w]
        R = R_scipy.from_quat(q_list).as_matrix()
        
        # Build T_vision (4×4 board -> camera_optical_link)
        self.T_vision = np.eye(4)
        self.T_vision[:3, :3] = R
        self.T_vision[:3, 3] = [p.x, p.y, p.z]
        
        # Build T_tf2 (4×4 camera_optical_link→base_link) from TF2
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', pose_msg.header.frame_id,
                rclpy.time.Time(seconds=0), timeout=Duration(seconds=0.2)
            )
            
            t = tf.transform.translation
            r = tf.transform.rotation
            q2_list = [r.x, r.y, r.z, r.w]
            R2 = R_scipy.from_quat(q2_list).as_matrix()
            
            self.T_tf2 = np.eye(4)
            self.T_tf2[:3, :3] = R2
            self.T_tf2[:3, 3] = [t.x, t.y, t.z]
            
        except Exception:
            return False
        
        # Get board center in base_link (using full ArUco transform for position)
        T_full = self.T_tf2 @ self.T_vision
        board_center_base = T_full[:3, 3]  # Translation = board origin in base_link
        
        # Build CLEAN T_combined using only the detected position
        # but with an IDEAL rotation for a VERTICAL board placed in front of the drone:
        #   Board X (right/left on face) → base_link -Y (since +Y is left)
        #   Board Y (up/down on face) → base_link +Z (up)
        #   Board Z (out of board) → base_link -X (towards drone)
        R_ideal = np.array([
            [ 0,  0, -1],   # board Z → base -X 
            [-1,  0,  0],   # board X → base -Y 
            [ 0,  1,  0],   # board Y → base +Z 
        ], dtype=np.float64)
        
        self.T_combined = np.eye(4)
        self.T_combined[:3, :3] = R_ideal
        self.T_combined[:3, 3] = board_center_base
        
        self.locked = True
        return True
    
    def board_to_base(self, points_board: np.ndarray) -> np.ndarray:
        """
        Transform points from board-local to base_link frame.
        
        Args:
            points_board: (N, 4) array of [x, y, 0, 1] in board-local coords
                          OR (N, 3) array of [x, y, z] (will add homogeneous coord)
        
        Returns:
            (N, 3) array of [x, y, z] in base_link frame
        """
        if self.T_combined is None:
            raise RuntimeError("Board transform not initialized — wait for ArUco detection")
        
        pts = np.atleast_2d(points_board)
        
        # Add homogeneous coordinate if needed
        if pts.shape[1] == 3:
            pts = np.hstack([pts, np.ones((pts.shape[0], 1))])
        
        # Transform: (4×4) @ (4×N).T → (4×N).T → take [:, :3]
        transformed = (self.T_combined @ pts.T).T
        return transformed[:, :3]
    
    def board_to_camera(self, points_board: np.ndarray) -> np.ndarray:
        """Transform points from board-local to camera_link frame."""
        if self.T_vision is None:
            raise RuntimeError("T_vision not initialized")
        
        pts = np.atleast_2d(points_board)
        if pts.shape[1] == 3:
            pts = np.hstack([pts, np.ones((pts.shape[0], 1))])
        
        transformed = (self.T_vision @ pts.T).T
        return transformed[:, :3]
    
    def get_board_center_base(self) -> np.ndarray:
        """Get board center position in base_link frame."""
        origin = np.array([[0, 0, 0, 1]])
        return self.board_to_base(origin)[0]
    
    def reset(self):
        """Reset transform (unlock for re-detection)."""
        self.T_vision = None
        self.T_tf2 = None
        self.T_combined = None
        self.locked = False
