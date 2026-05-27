#!/usr/bin/env python3
"""
Vision ArUco Detector for Visual Servoing
Adapted from Visual_Servoing/nodes/vision_node_ros2.py

Detects ArUco markers (IDs 0-3) and publishes board pose.
Works with Gazebo camera simulation.
"""

import rclpy
from rclpy.node import Node
import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
import os

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import time


# ArUco Board Configuration (matching Visual_Servoing)
BOARD_SIZE_M = 0.120       # 120mm board
OFFSET = 0.048             # Marker offset from center
MARKER_SIZE = 0.020        # 20mm markers
HALF_MARKER = MARKER_SIZE / 2


def get_marker_corners_3d(center_x, center_y, half_size):
    """Get 3D coordinates of marker corners in board frame."""
    return np.array([
        [center_x - half_size, center_y + half_size, 0],
        [center_x + half_size, center_y + half_size, 0],
        [center_x + half_size, center_y - half_size, 0],
        [center_x - half_size, center_y - half_size, 0]
    ], dtype=np.float32)


# Board marker positions (IDs 0-3 at corners)
# Standard ArUco Board Layout:
# 0: Top-Left, 1: Top-Right, 2: Bottom-Right, 3: Bottom-Left
# Using standard 2D coords: X-right, Y-up (relative to board center)
BOARD_CONFIG_3D = {
    0: get_marker_corners_3d(-OFFSET,  OFFSET, HALF_MARKER),  # Top-left
    1: get_marker_corners_3d( OFFSET,  OFFSET, HALF_MARKER),  # Top-right
    2: get_marker_corners_3d( OFFSET, -OFFSET, HALF_MARKER),  # Bottom-right
    3: get_marker_corners_3d(-OFFSET, -OFFSET, HALF_MARKER)   # Bottom-left
}


class VisionArucoDetector(Node):
    """ROS2 node for ArUco marker detection and board pose estimation."""
    
    def __init__(self):
        super().__init__('vision_aruco_detector')
        
        # Declare parameters
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('show_gui', False)
        
        image_topic = self.get_parameter('image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.show_gui = self.get_parameter('show_gui').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # ArUco detector setup - handle different OpenCV versions
        # Using DICT_4X4_1000 to match user's marker files (4x4_1000-*.svg)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        # OpenCV 4.7+ uses DetectorParameters(), older uses DetectorParameters_create()
        try:
            self.aruco_params = aruco.DetectorParameters()
            self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            self.aruco_params = aruco.DetectorParameters_create()
            self.use_new_api = False
        
        # Camera intrinsics (will be updated from camera_info)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Default camera matrix for Gazebo (640x480, 80° FOV)
        fx = 554.254  # Approximate for 80° FOV
        fy = 554.254
        cx = 320.0
        cy = 240.0
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(5)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        
        # Publishers
        self.board_pose_pub = self.create_publisher(
            PoseStamped, '/vision/board_pose', 10)
        self.board_detected_pub = self.create_publisher(
            Bool, '/vision/board_detected', 10)
        
        # State
        self.last_board_pose = None
        self.last_detection_time = 0.0
        self.detection_count = 0
        self.board_locked = False
        self.locked_pose = None
        self.pose_buffer = []  # To average multiple detections for stable lock
        self.buffer_size = 30  # Average 30 frames for a solid lock
        self.cache_timeout = 1.0  # Use cached pose for 1 second after losing detection
        
        self.get_logger().info(f"[VisionAruco] Started, listening on {image_topic}")
        self.get_logger().info(f"[VisionAruco] Board size: {BOARD_SIZE_M*100:.0f}cm, Pose cache: {self.cache_timeout}s")
    
    def camera_info_callback(self, msg: CameraInfo):
        """Update camera intrinsics from camera_info topic (only once)."""
        if msg.k[0] > 0:
            new_fx = msg.k[0]
            # Only update and log if fx changed significantly (first real update)
            if self.camera_matrix is None or abs(self.camera_matrix[0, 0] - new_fx) > 1.0:
                self.camera_matrix = np.array(msg.k).reshape(3, 3)
                self.dist_coeffs = np.array(msg.d) if len(msg.d) > 0 else np.zeros(5)
                self.get_logger().info(f"[VisionAruco] Camera matrix updated: fx={new_fx:.1f}")
    
    def image_callback(self, msg: Image):
        """Process camera image for ArUco detection."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        
        # Detect ArUco markers - use correct API based on version
        if self.use_new_api:
            corners, ids, rejected = self.aruco_detector.detectMarkers(cv_image)
        else:
            corners, ids, rejected = aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)
        
        detected = Bool()
        detected.data = False
        
        if ids is not None and len(ids) >= 2:
            # Collect detected markers that match our board
            object_points = []
            image_points = []
            
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in BOARD_CONFIG_3D:
                    obj_pts = BOARD_CONFIG_3D[marker_id]
                    img_pts = corners[i][0]
                    object_points.append(obj_pts)
                    image_points.append(img_pts)
            
            if len(object_points) >= 2:
                # Stack points for solvePnP
                object_points = np.vstack(object_points).astype(np.float32)
                image_points = np.vstack(image_points).astype(np.float32)
                
                # Solve PnP
                success, rvec, tvec = cv2.solvePnP(
                    object_points, image_points,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Convert rotation vector to matrix then to quaternion using scipy
                    R_mat, _ = cv2.Rodrigues(rvec)
                    quat = R_scipy.from_matrix(R_mat).as_quat() # [x, y, z, w]
                    
                    # Store in temporary pose
                    current_pose = PoseStamped()
                    current_pose.header.stamp = self.get_clock().now().to_msg()
                    current_pose.header.frame_id = 'camera_optical_link'
                    current_pose.pose.position.x = float(tvec[0])
                    current_pose.pose.position.y = float(tvec[1])
                    current_pose.pose.position.z = float(tvec[2])
                    current_pose.pose.orientation.x = float(quat[0])
                    current_pose.pose.orientation.y = float(quat[1])
                    current_pose.pose.orientation.z = float(quat[2])
                    current_pose.pose.orientation.w = float(quat[3])
                    
                    if not self.board_locked:
                        # Collect samples for averaging
                        self.pose_buffer.append(current_pose)
                        if len(self.pose_buffer) >= self.buffer_size:
                            # Average position
                            avg_pos = np.mean([
                                [p.pose.position.x, p.pose.position.y, p.pose.position.z] 
                                for p in self.pose_buffer
                            ], axis=0)
                            
                            # Average quaternion (simple mean + normalize is robust for close rotations)
                            avg_q = np.mean([
                                [p.pose.orientation.x, p.pose.orientation.y, 
                                 p.pose.orientation.z, p.pose.orientation.w]
                                for p in self.pose_buffer
                            ], axis=0)
                            avg_q /= np.linalg.norm(avg_q)
                            
                            self.locked_pose = PoseStamped()
                            self.locked_pose.header.frame_id = 'camera_optical_link'
                            self.locked_pose.pose.position.x = avg_pos[0]
                            self.locked_pose.pose.position.y = avg_pos[1]
                            self.locked_pose.pose.position.z = avg_pos[2]
                            self.locked_pose.pose.orientation.x = avg_q[0]
                            self.locked_pose.pose.orientation.y = avg_q[1]
                            self.locked_pose.pose.orientation.z = avg_q[2]
                            self.locked_pose.pose.orientation.w = avg_q[3]
                            
                            self.board_locked = True
                            self.get_logger().info(f"✨ Board pose LOCKED after {self.buffer_size} samples")
                    
                    # Always publish the locked pose once we have it
                    if self.board_locked:
                        self.locked_pose.header.stamp = self.get_clock().now().to_msg()
                        self.board_pose_pub.publish(self.locked_pose)
                        self.last_board_pose = self.locked_pose
                        self.last_detection_time = time.time()
                        detected.data = True
                        
                        if self.detection_count == 0 or self.detection_count % 3000 == 0:
                            self.get_logger().info(
                                f"[VisionAruco] Board locked at "
                                f"({self.locked_pose.pose.position.x:.3f}, "
                                f"{self.locked_pose.pose.position.y:.3f}, "
                                f"{self.locked_pose.pose.position.z:.3f})m "
                                f"[frame #{self.detection_count}]"
                            )
                    
                    self.detection_count += 1
        
        # Use cached pose when detection fails (handles robot occlusion)
        if not detected.data and self.last_board_pose is not None:
            elapsed = time.time() - self.last_detection_time
            if elapsed < self.cache_timeout:
                # Publish cached pose with updated timestamp
                cached_msg = PoseStamped()
                cached_msg.header.stamp = self.get_clock().now().to_msg()
                cached_msg.header.frame_id = 'camera_optical_link'
                cached_msg.pose = self.last_board_pose.pose
                self.board_pose_pub.publish(cached_msg)
                detected.data = True  # Still "detected" via cache
        
        self.board_detected_pub.publish(detected)
        
        # GUI visualization
        if self.show_gui:
            vis_image = cv_image.copy()
            if ids is not None:
                aruco.drawDetectedMarkers(vis_image, corners, ids)
            cv2.imshow('ArUco Detection', vis_image)
            cv2.waitKey(1)
    
    def rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w])


def main(args=None):
    rclpy.init(args=args)
    node = VisionArucoDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_gui:
            cv2.destroyAllWindows()
        node.destroy_node()
        # Only shutdown if not already shut down
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

