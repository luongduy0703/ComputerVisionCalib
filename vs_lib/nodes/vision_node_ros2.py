#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ROS 2 Humble Version

import rclpy
from rclpy.node import Node
import cv2
import cv2.aruco as aruco
import numpy as np
import csv
import os
import math
import sys
from datetime import datetime
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import Vector3, PoseStamped

# Import Profiler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from core.profiler import SystemProfiler
except ImportError:
    # Dummy class nếu không tìm thấy để tránh crash
    class SystemProfiler:
        def __init__(self, filename): pass
        def start_timer(self, k): pass
        def stop_timer(self, k): return 0.0
        def log_data(self, **k): pass
        def print_summary(self): pass

# ==========================================
# CẤU HÌNH NGƯỜI DÙNG (USER CONFIG)
# ==========================================
FORCE_USER_CALIB = True 

# Thông số Camera
K_user = np.array([
    [839.3449,    0.    , 288.5372],
    [   0.    , 839.4361, 236.4550],
    [   0.    ,    0.    ,    1.    ]
], dtype=np.float32)

D_user = np.array([[-0.22608, -0.02472, -0.00085, 0.00054, -0.31300]], dtype=np.float32)

# Cấu hình Marker
BOARD_SIZE_M = 0.120       
OFFSET = 0.048             
SAFE_ZONE = 0.070          
MARKER_SIZE = 0.020        
HALF_MARKER = MARKER_SIZE / 2
NUM_POINTS = 20

def get_marker_corners_3d(center_x, center_y, half_size):
    return np.array([
        [center_x - half_size, center_y + half_size, 0], 
        [center_x + half_size, center_y + half_size, 0], 
        [center_x + half_size, center_y - half_size, 0], 
        [center_x - half_size, center_y - half_size, 0]  
    ], dtype=np.float32)

BOARD_CONFIG_3D = {
    0: get_marker_corners_3d(-OFFSET,  OFFSET, HALF_MARKER), 
    1: get_marker_corners_3d( OFFSET,  OFFSET, HALF_MARKER), 
    2: get_marker_corners_3d( OFFSET, -OFFSET, HALF_MARKER), 
    3: get_marker_corners_3d(-OFFSET, -OFFSET, HALF_MARKER)  
}

class RobustPBVSNode(Node):
    def __init__(self):
        super().__init__('robust_pbvs_node')
        
        # Declare parameters (ROS 2 style)
        self.declare_parameter('image_topic', '/camera/image_raw')  # Fixed: was '/image_raw'
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('show_gui', False)
        # Web-video-server friendly monitor stream
        self.declare_parameter('monitor_image_topic', '/pbvs/monitor_image')
        self.declare_parameter('monitor_metrics_topic', '/pbvs/monitor_metrics')
        
        # Get parameters
        self.image_topic = self.get_parameter('image_topic').value
        self.info_topic = self.get_parameter('camera_info_topic').value
        self.show_gui = self.get_parameter('show_gui').value
        self.monitor_image_topic = self.get_parameter('monitor_image_topic').value
        self.monitor_metrics_topic = self.get_parameter('monitor_metrics_topic').value

        # Publishers (ROS 2 style)
        self.pub_board_pose = self.create_publisher(PoseStamped, '/target_pose', 1)
        self.pub_euler = self.create_publisher(Vector3, 'board_euler', 1)
        self.pub_position = self.create_publisher(Vector3, 'camera_position', 1)
        self.debug_pub = self.create_publisher(Image, '/front_cam/usb_cam/robust_debug', 1)
        self.monitor_pub = self.create_publisher(Image, self.monitor_image_topic, 1)

        # Subscribers
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            1  # QoS depth (equivalent to queue_size in ROS 1)
        )

        # Subscribe control-side metrics to overlay on the monitor stream
        self.latest_metrics = None
        self.metrics_sub = self.create_subscription(
            Float32MultiArray,
            self.monitor_metrics_topic,
            self._metrics_cb,
            1
        )
        
        # ArUco Init - Fixed to match marker files (4x4_1000-*.svg)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        self.params = aruco.DetectorParameters_create()
        
        # [CẢI TIẾN] Profiler riêng cho Vision Node (ghi vào vision_metrics.csv)
        self.profiler = SystemProfiler("vision_metrics.csv")
        
        self.K = K_user if FORCE_USER_CALIB else None
        self.D = D_user if FORCE_USER_CALIB else None
        
        # FPS Counter
        self.frame_count = 0
        self.fps_start_time = self.get_clock().now().nanoseconds / 1e9
        self.fps = 0.0
        
        self.get_logger().info("[RobustPBVS] Started with Benchmark enabled.")

    def _metrics_cb(self, msg: Float32MultiArray):
        # Expected layout (from center_tracking_executor_ros2.py and similar nodes):
        # [delay_ms, err_cm, vec_cm(x,y,z), target_base_cm(x,y,z), cmd_base_cm(x,y,z), roll_deg, pitch_deg, yaw_deg]
        self.latest_metrics = list(msg.data)

    def rotation_matrix_to_quaternion(self, R):
        tr = R[0,0] + R[1,1] + R[2,2]
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
        return [qx, qy, qz, qw]

    def rotation_matrix_to_euler(self, R):
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])

    def image_cb(self, msg):
        self.profiler.start_timer("Vision_Total_ms")
        
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # --- 1. Đo ArUco Detect ---
        self.profiler.start_timer("Vision_Detect_ms")
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
        t_detect = self.profiler.stop_timer("Vision_Detect_ms")

        found_board = False
        t_solve = 0.0
        
        # FPS Calculation
        self.frame_count += 1
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if now_sec - self.fps_start_time >= 1.0:
            self.fps = self.frame_count / (now_sec - self.fps_start_time)
            self.frame_count = 0
            self.fps_start_time = now_sec

        image_points_collected = []
        object_points_collected = []

        if ids is not None and len(ids) > 0:
            for i in range(len(ids)):
                curr_id = ids[i][0]
                if curr_id in BOARD_CONFIG_3D:
                    curr_corners_2d = corners[i][0]
                    curr_corners_3d = BOARD_CONFIG_3D[curr_id]
                    for pt in curr_corners_2d: image_points_collected.append(pt)
                    for pt in curr_corners_3d: object_points_collected.append(pt)

            if len(image_points_collected) >= 4:
                img_pts = np.array(image_points_collected, dtype=np.float32)
                obj_pts = np.array(object_points_collected, dtype=np.float32)
                
                # --- 2. Đo SolvePnP ---
                self.profiler.start_timer("Vision_Solve_ms")
                success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.D)
                t_solve = self.profiler.stop_timer("Vision_Solve_ms")
                
                if success:
                    found_board = True
                    rmat, _ = cv2.Rodrigues(rvec)
                    quat = self.rotation_matrix_to_quaternion(rmat)
                    
                    pose_msg = PoseStamped()
                    # [QUAN TRỌNG] Gán timestamp của ảnh gốc để tính Phase Delay chính xác bên Executor
                    pose_msg.header.stamp = msg.header.stamp
                    pose_msg.header.frame_id = "camera_link"
                    pose_msg.pose.position.x = float(tvec[0][0])
                    pose_msg.pose.position.y = float(tvec[1][0])
                    pose_msg.pose.position.z = float(tvec[2][0])
                    pose_msg.pose.orientation.x = float(quat[0])
                    pose_msg.pose.orientation.y = float(quat[1])
                    pose_msg.pose.orientation.z = float(quat[2])
                    pose_msg.pose.orientation.w = float(quat[3])
                    
                    self.pub_board_pose.publish(pose_msg)
                    
                    # Debug output
                    euler = self.rotation_matrix_to_euler(rmat)
                    euler_msg = Vector3()
                    euler_msg.x = float(euler[0])
                    euler_msg.y = float(euler[1])
                    euler_msg.z = float(euler[2])
                    self.pub_euler.publish(euler_msg)
                    
                    # Vẽ trục
                    cv2.drawFrameAxes(cv_img, self.K, self.D, rvec, tvec, 0.05)

        # Tính latency nội bộ (từ lúc chụp đến lúc xử lý xong)
        current_time = self.get_clock().now()
        latency_ns = (current_time.nanoseconds - msg.header.stamp.sec * 1e9 - msg.header.stamp.nanosec)
        latency_ms = latency_ns / 1e6
        t_total = self.profiler.stop_timer("Vision_Total_ms")
        
        # Log vào CSV riêng của Vision (chỉ log khi tìm thấy bảng để tránh rác)
        if found_board:
            stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            self.profiler.log_data(
                Timestamp=stamp_sec,
                Vision_Detect_ms=t_detect,
                Vision_Solve_ms=t_solve,
                Vision_Total_ms=t_total,
                Vision_Latency_ms=latency_ms
            )
        
        # Debug Text
        status_text = f"Robust: LOCKED FPS:{self.fps:.1f}" if found_board else f"Robust: SEARCHING... FPS:{self.fps:.1f}"
        color = (0, 255, 0) if found_board else (0, 0, 255)
        cv2.putText(cv_img, f"{status_text} | Latency: {latency_ms:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ----- Monitor overlay (for web_video_server) -----
        # Show: delay, error, vector-to-target, target pose direction hints.
        y0 = 55
        dy = 22
        cv2.putText(cv_img, "PBVS MONITOR", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y0 += dy

        if self.latest_metrics is not None and len(self.latest_metrics) >= 14:
            delay_ms = float(self.latest_metrics[0])
            err_cm = float(self.latest_metrics[1])
            vx, vy, vz = float(self.latest_metrics[2]), float(self.latest_metrics[3]), float(self.latest_metrics[4])
            txb, tyb, tzb = float(self.latest_metrics[5]), float(self.latest_metrics[6]), float(self.latest_metrics[7])
            cxb, cyb, czb = float(self.latest_metrics[8]), float(self.latest_metrics[9]), float(self.latest_metrics[10])
            roll_d = float(self.latest_metrics[11])
            pitch_d = float(self.latest_metrics[12])
            yaw_d = float(self.latest_metrics[13])

            cv2.putText(cv_img, f"CTRL delay={delay_ms:.0f}ms  err={err_cm:.2f}cm", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y0 += dy
            cv2.putText(cv_img, f"vec_cm=({vx:+.2f},{vy:+.2f},{vz:+.2f})", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y0 += dy
            cv2.putText(cv_img, f"target_base_cm=({txb:+.1f},{tyb:+.1f},{tzb:+.1f})", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            y0 += dy
            cv2.putText(cv_img, f"cmd_base_cm=({cxb:+.1f},{cyb:+.1f},{czb:+.1f})", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            y0 += dy
            cv2.putText(cv_img, f"roll/pitch/yaw=({roll_d:+.1f},{pitch_d:+.1f},{yaw_d:+.1f})", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            y0 += dy
        else:
            cv2.putText(cv_img, "CTRL metrics: (waiting...)", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        try:
            out_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            self.debug_pub.publish(out_msg)
            # Dedicated topic intended for web_video_server streaming
            self.monitor_pub.publish(out_msg)
        except: pass

def main(args=None):
    rclpy.init(args=args)
    node = RobustPBVSNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # In tổng kết khi tắt node
        print("\n" + "="*40)
        print("🛑 Vision Node Stopped. Generating Report...")
        node.profiler.print_summary()
        node.destroy_node()
        # Only shutdown if not already shut down
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()

