#!/usr/bin/env python3
"""
Camera Viewer with RL Training Overlay

Visualizes:
- ArUco board detection with PBVS monitor
- RL target position (green circle)
- Drawing pen trajectory (purple line)
"""

import os

# Suppress C++ TF_OLD_DATA warnings from buffer_core.cpp
# These are caused by Gazebo sim-time clock mismatches and are harmless
os.environ['TF2_CPP_LOGGING_LEVEL'] = 'ERROR'

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time
import logging
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from std_msgs.msg import Bool, Float32MultiArray
from cv_bridge import CvBridge
import tf2_ros
from rclpy.duration import Duration


class CameraViewer(Node):
    """Camera viewer with RL training overlay visualization."""
    
    def __init__(self):
        super().__init__('camera_viewer',
                         parameter_overrides=[
                             rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
                         ])
        
        self.bridge = CvBridge()
        
        # TF2 for coordinate transform
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Suppress TF2 C++ warnings (TF_OLD_DATA from sim-time clock mismatch)
        tf2_logger = logging.getLogger('tf2_ros')
        tf2_logger.setLevel(logging.ERROR)
        
        # Camera intrinsics
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # ArUco board state
        self.board_pose = None
        self.board_detected = False
        
        # RL Training state
        self.current_target = None
        self.pen_trajectory = []
        self.max_trajectory_points = 200  # Last 200 points
        self.shape_waypoints = None  # Full shape outline
        
        # Camera extrinsics (base_link → camera_link)
        # Will be computed from board pose (rvec/tvec from ArUco)
        self.cam_rvec = None
        self.cam_tvec = None
        
        # FPS Calculation
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0.0
        
        # Subscribers - Board detection
        self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(
            CameraInfo, '/camera/camera_info', self.info_callback, 10)
        self.create_subscription(
            PoseStamped, '/vision/board_pose', self.board_pose_callback, 10)
        self.create_subscription(
            Bool, '/vision/board_detected', self.board_detected_callback, 10)
        
        # Subscribers - RL Training
        self.create_subscription(
            Point, '/rl/current_target', self.target_callback, 10)
        self.create_subscription(
            PointStamped, '/rl/pen_position', self.pen_callback, 10)
        self.create_subscription(
            Float32MultiArray, '/rl/shape_waypoints', self.shape_callback, 10)
        self.create_subscription(
            Bool, '/rl/reset_trajectory', self.reset_trajectory_callback, 10)
        
        self.get_logger().info("Camera Viewer with RL Overlay started")
    
    def info_callback(self, msg):
        """Get camera intrinsics for drawing axes."""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Camera intrinsics received")
    
    def board_pose_callback(self, msg):
        self.board_pose = msg
    
    def board_detected_callback(self, msg):
        self.board_detected = msg.data
    
    def target_callback(self, msg):
        """Update current RL target position."""
        self.current_target = msg
    
    def pen_callback(self, msg):
        """Add pen position to trajectory."""
        self.pen_trajectory.append(msg.point)
        if len(self.pen_trajectory) > self.max_trajectory_points:
            self.pen_trajectory.pop(0)
    
    def shape_callback(self, msg):
        """Receive full shape waypoints [x0,y0,z0, x1,y1,z1, ...]."""
        data = msg.data
        if len(data) >= 6:  # At least 2 points
            self.shape_waypoints = np.array(data).reshape(-1, 3)
            self.get_logger().info(f"Shape received: {len(self.shape_waypoints)} waypoints")
    
    def reset_trajectory_callback(self, msg):
        """Clear pen trajectory on episode reset."""
        self.pen_trajectory.clear()
        self.get_logger().info("Trajectory cleared (episode reset)")
    
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        
        # FPS Calculation
        self.frame_count += 1
        now = time.time()
        if now - self.fps_start_time >= 1.0:
            self.fps = self.frame_count / (now - self.fps_start_time)
            self.frame_count = 0
            self.fps_start_time = now
        
        # Latency Calculation: Use wall-clock time for actual processing delay
        # This measures the time from when frame arrived to when it's displayed
        # Store arrival time on first access for this message
        frame_arrival_time = time.time()
        
        # Calculate processing latency (time spent in this callback)
        # For display, we just show how long since last frame was processed
        if not hasattr(self, '_last_frame_time'):
            self._last_frame_time = frame_arrival_time
        
        frame_delta_ms = (frame_arrival_time - self._last_frame_time) * 1000
        self._last_frame_time = frame_arrival_time
        
        # Show frame interval as "latency" (time between frames)
        latency_ms = frame_delta_ms if frame_delta_ms < 1000 else 0.0
        
        # Initialize status (default: searching)
        status_text = f"Robust: SEARCHING... FPS:{self.fps:.1f}"
        color = (0, 0, 255)  # Red
        
        # Draw Overlay (Matching original style)
        if self.board_detected and self.board_pose:
            # "Robust: LOCKED"
            status_text = f"Robust: LOCKED FPS:{self.fps:.1f}"
            color = (0, 255, 0)  # Green
            
            # Extract position (needed for display)
            pos = self.board_pose.pose.position
            
            # Draw Axes if intrinsics available
            if self.camera_matrix is not None:
                ori = self.board_pose.pose.orientation
                
                tvec = np.array([[pos.x], [pos.y], [pos.z]])
                rmat = self.quaternion_to_rotation_matrix(ori)
                rvec, _ = cv2.Rodrigues(rmat)
                
                try:
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, 
                                     rvec, tvec, 0.05)  # 5cm axes
                except Exception:
                    pass
            
            # Additional Monitor Info
            y0 = 55
            dy = 22
            cv2.putText(cv_image, "PBVS MONITOR", (10, y0), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            y0 += dy
            cv2.putText(cv_image, f"cam=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f})", 
                       (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            
            # Transform to base_link and show
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link', self.board_pose.header.frame_id,
                    rclpy.time.Time(seconds=0), timeout=Duration(seconds=0, nanoseconds=50000000))
                t = transform.transform.translation
                r = transform.transform.rotation
                qx, qy, qz, qw = r.x, r.y, r.z, r.w
                rot = np.array([
                    [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                    [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                    [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
                ])
                pt = rot @ np.array([pos.x, pos.y, pos.z]) + np.array([t.x, t.y, t.z])
                y0 += dy
                cv2.putText(cv_image, f"base=({pt[0]:.3f},{pt[1]:.3f},{pt[2]:.3f})", 
                           (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            except Exception:
                y0 += dy
                cv2.putText(cv_image, "base=(TF2 pending...)", 
                           (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        
        # NOTE: Shape/target/pen visualization handled by gazebo_visualizer.py
        # (spawns 3D entities in Gazebo that camera sees naturally)
        
        # Status Text (Top Left)
        cv2.putText(cv_image, f"{status_text} | Latency: {latency_ms:.1f}ms", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Camera View', cv_image)
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            rclpy.shutdown()
        elif key == ord('s'):
            filename = f"/tmp/camera_snapshot_{time.time()}.png"
            cv2.imwrite(filename, cv_image)
            self.get_logger().info(f"Saved: {filename}")
        elif key == ord('c'):  # Clear trajectory
            self.pen_trajectory.clear()
            self.get_logger().info("Trajectory cleared")
    

            
def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
