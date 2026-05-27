#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ROS 2 Humble Version

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import math
import time
import sys
import os
import numpy as np
import json
import argparse
import yaml
from geometry_msgs.msg import PoseStamped

# --- Import Core Modules ---
try:
    from shape_generator import ShapeGenerator
except ImportError:
    print("❌ Lỗi: Không tìm thấy file 'shape_generator.py'.")
    sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import board
import busio
from adafruit_tca9548a import TCA9548A

from core.kinematics import KinematicsSolver
from core.filters import EMASmoother, KalmanFilter1D, OutlierRejector, OneEuroFilter
from drivers.i2c_manager import ServoController

# [BENCHMARK] Import Profiler
try:
    from core.profiler import SystemProfiler
except ImportError:
    print("⚠️ Không tìm thấy profiler.py, chạy chế độ dummy.")
    class SystemProfiler:
        def __init__(self, f, output_dir=None): pass
        def start_timer(self, k): pass
        def stop_timer(self, k): return 0.0
        def log_data(self, **k): pass
        def print_summary(self): pass

import threading
from queue import Queue, Full
from collections import deque

class PBVSArtist(Node):
    def __init__(self, input_data, is_file=True):
        super().__init__('pbvs_artist_node')

        # --- 1. LOAD CONFIG ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config', 'robot_config.yaml')
        
        try:
            with open(config_path, 'r') as f:
                self.full_config = yaml.safe_load(f)
            print(f"✅ Loaded Control Config: {config_path}")
        except FileNotFoundError:
            print(f"❌ Error: Config file not found at {config_path}")
            sys.exit(1)

        ctrl_cfg = self.full_config.get('control', {})
        self.DRAW_SPEED_CM_S = ctrl_cfg.get('speed', {}).get('draw_cm_s', 1.0)
        self.AIR_SPEED_CM_S  = ctrl_cfg.get('speed', {}).get('air_cm_s', 1.5)
        self.LIFT_HEIGHT_CM  = ctrl_cfg.get('geometry', {}).get('lift_height_cm', -2.0)
        self.FIXED_TILT      = ctrl_cfg.get('geometry', {}).get('fixed_tilt', -35.0)
        self.DRAWING_THRESHOLD_CM = ctrl_cfg.get('geometry', {}).get('drawing_threshold_cm', 0.5)
        self.STROKE_INPUT_METERS = bool(ctrl_cfg.get('geometry', {}).get('stroke_input_meters', True))
        self.MIN_SAFETY_DIST_CM = ctrl_cfg.get('safety', {}).get('min_dist_cm', 4.0)
        
        # [DEBUG] Waypoint logs (inspect EE waypoint input quality)
        dbg_cfg = ctrl_cfg.get('debug', {})
        self.DEBUG_LOG_WAYPOINTS = bool(dbg_cfg.get('log_waypoints', False))
        self.DEBUG_LOG_EVERY_N_STEPS = int(max(1, dbg_cfg.get('log_every_n_steps', 5)))
        self.DEBUG_LOG_CSV_EXTRA = bool(dbg_cfg.get('log_csv_extra_columns', True))

        # [6-DOF compensation] Defaults are safe no-ops unless user config enables them
        balance_cfg = ctrl_cfg.get('autobalancing', {})
        self.AUTOBALANCING_ENABLED = bool(balance_cfg.get('enabled', False))
        self.COMPENSATION_GAIN = float(balance_cfg.get('compensation_gain', 1.0))
        self.ORIENTATION_COMPENSATION_GAIN = float(balance_cfg.get('orientation_compensation_gain', 1.0))
        self.MAX_COMPENSATION_CM = float(balance_cfg.get('max_compensation_cm', 5.0))
        self.MAX_COMPENSATION_DEG = float(balance_cfg.get('max_compensation_deg', 15.0))
        self.ROLL_COMPENSATION = bool(balance_cfg.get('roll_compensation', True))
        self.PITCH_COMPENSATION = bool(balance_cfg.get('pitch_compensation', True))
        self.YAW_COMPENSATION = bool(balance_cfg.get('yaw_compensation', False))

        # Reference pose & attitude (used by calculate_6dof_compensation)
        self.reference_board_pose = None
        self.drone_attitude = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}  # degrees
        
        # [NEW] Prediction settings
        pred_cfg = ctrl_cfg.get('prediction', {})
        self.PREDICTION_ENABLED = pred_cfg.get('enabled', True)
        self.PREDICTION_HISTORY_SIZE = pred_cfg.get('history_size', 10)
        self.MAX_PREDICTION_TIME_MS = pred_cfg.get('max_prediction_time_ms', 50.0)
        self.EXTRAPOLATION_TIMEOUT_MS = pred_cfg.get('extrapolation_timeout_ms', 200.0)
        self.VELOCITY_SMOOTHING_ALPHA = pred_cfg.get('velocity_smoothing_alpha', 0.3)
        self.HOME_POSE = [90, 60, 60, 90] 

        # --- Servo calibration (sign & offset around home) ---
        # Mirrors the runtime fixes you used in the standalone script:
        #   - sign_shoulder:=-1.0, sign_elbow:=-1.0
        #   - offset_shoulder_deg:=30.0, offset_elbow_deg:=-30.0
        robot_cfg = self.full_config.get('robot', {})
        servo_cal = robot_cfg.get('servo_calibration', {})
        self.HOME_DEG = float(servo_cal.get('home_deg', 90.0))
        self.SIGN_BASE = float(servo_cal.get('sign_base', 1.0))
        self.SIGN_SHOULDER = float(servo_cal.get('sign_shoulder', -1.0))
        self.SIGN_ELBOW = float(servo_cal.get('sign_elbow', -1.0))
        self.SIGN_WRIST = float(servo_cal.get('sign_wrist', 1.0))
        self.OFFSET_BASE_DEG = float(servo_cal.get('offset_base_deg', 0.0))
        self.OFFSET_SHOULDER_DEG = float(servo_cal.get('offset_shoulder_deg', 30.0))
        self.OFFSET_ELBOW_DEG = float(servo_cal.get('offset_elbow_deg', -30.0))
        self.OFFSET_WRIST_DEG = float(servo_cal.get('offset_wrist_deg', 0.0))

        # --- 2. INIT FILTERS ---
        # [NOTE: All filters operate in CM for consistency]
        # core.filters.OutlierRejector signature: OutlierRejector(max_jump=...)
        self.outlier_x = OutlierRejector(max_jump=5.0)
        self.outlier_y = OutlierRejector(max_jump=5.0)
        self.outlier_z = OutlierRejector(max_jump=5.0)
        
        # core.filters.KalmanFilter1D signature: KalmanFilter1D(R=..., Q=...)
        # R: measurement noise, Q: process noise
        self.kalman_x = KalmanFilter1D(R=0.5, Q=0.01)
        self.kalman_y = KalmanFilter1D(R=0.5, Q=0.01)
        self.kalman_z = KalmanFilter1D(R=0.5, Q=0.01)
        
        self.one_euro_x = OneEuroFilter(min_cutoff=0.01, beta=0.01)
        self.one_euro_y = OneEuroFilter(min_cutoff=0.01, beta=0.01)
        self.one_euro_z = OneEuroFilter(min_cutoff=0.01, beta=0.01)

        # --- 3. INIT SERVOS ---
        # Use wiring + calibration consistent with wicom_roboarm_4dof_standalone.py
        serv_cfg = robot_cfg.get('servos', {})
        fixed_channels = serv_cfg.get('fixed_channels', [])
        fixed_degs = serv_cfg.get('fixed_degs', [])
        off_channels = serv_cfg.get('off_channels', [])
        shoulder_mirror_enabled = bool(serv_cfg.get('shoulder_mirror_enabled', False))
        shoulder_mirror_channel = serv_cfg.get('shoulder_mirror_channel', None)
        shoulder_mirror_angle_max = float(serv_cfg.get('shoulder_mirror_angle_max', 180.0))

        self.servos = ServoController(
            fixed_channels=fixed_channels,
            fixed_degs=fixed_degs,
            off_channels=off_channels,
            shoulder_mirror_enabled=shoulder_mirror_enabled,
            shoulder_mirror_channel=shoulder_mirror_channel,
            shoulder_mirror_angle_max=shoulder_mirror_angle_max,
        )

        # --- 4. LOAD CALIBRATION MATRIX ---
        calib_path = os.path.join(current_dir, '..', 'config', 'T_cam_to_base_THEORETICAL.npy')
        try:
            self.T_calib = np.load(calib_path)
            self.T_calib_inv = np.linalg.inv(self.T_calib)  # base → camera for prediction
            print(f"✅ Calibration Matrix Loaded: {calib_path}")
        except FileNotFoundError:
            print(f"❌ Error: Calibration matrix not found at {calib_path}")
            sys.exit(1)

        # --- 5. INIT PROFILER ---
        self.profiler = SystemProfiler("pbvs_metrics.csv", output_dir=current_dir)

        # --- 6. VISION QUEUE & THREADING ---
        self.vision_queue = Queue(maxsize=2)
        self.latest_board_pose = None
        self.last_vision_time = 0.0
        self.vision_timeout = 2.0  # seconds

        # --- 7. PREDICTION STATE (velocity in base frame to avoid yaw "tail" error) ---
        self.pose_history = deque(maxlen=self.PREDICTION_HISTORY_SIZE)
        self.velocity_linear = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # cm/s in base frame
        self.velocity_angular = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # rad/s (approximate)
        self.last_predicted_pose = None

        # --- 8. CONTROL LOOP TIMING ---
        self.dt_period = 0.02  # 50Hz
        self.next_wake_time = 0.0

        # --- 9. LOAD DRAWING DATA ---
        if is_file:
            with open(input_data, 'r') as f:
                self.strokes = json.load(f)
        else:
            self.strokes = input_data

        # --- 10. INIT VISION SUBSCRIBER (ROS 2) ---
        self.vision_sub = self.create_subscription(
            PoseStamped,
            '/target_pose',
            self.vision_callback,
            1  # QoS depth
        )

        # --- 11. INIT JOINT SMOOTHERS ---
        self.SERVO_SMOOTH_ALPHA = 0.2
        self.joint_smoothers = [EMASmoother(self.SERVO_SMOOTH_ALPHA) for _ in range(4)]
        for i, val in enumerate(self.HOME_POSE):
            self.joint_smoothers[i].value = val

    def vision_callback(self, msg):
        """ROS 2 callback for vision pose messages"""
        # [BENCHMARK] Lấy timestamp gốc từ ảnh (quan trọng để tính Phase Delay)
        # ROS 2: stamp is a Time object with sec and nanosec
        img_ts = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

        tx = msg.pose.position.x
        ty = msg.pose.position.y
        tz = msg.pose.position.z
        qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w

        # Convert to CM and filter (optimized: keep in CM to avoid unnecessary conversions)
        tx_cm, ty_cm, tz_cm = tx*100.0, ty*100.0, tz*100.0
        
        # Outlier Rejection (operates in CM)
        tx_cm = self.outlier_x.check(tx_cm)
        ty_cm = self.outlier_y.check(ty_cm)
        tz_cm = self.outlier_z.check(tz_cm)
        
        # Kalman Filter (operates in CM)
        tx_cm = self.kalman_x.update(tx_cm)
        ty_cm = self.kalman_y.update(ty_cm)
        tz_cm = self.kalman_z.update(tz_cm)

        # Store filtered pose in CM (optimized: avoid METERS→CM→METERS conversion)
        # Convert to meters only when needed for coordinate transformation
        filtered_pose = {
            'tx_cm': tx_cm, 'ty_cm': ty_cm, 'tz_cm': tz_cm,  # Store in CM for direct use
            'tx': tx_cm/100.0, 'ty': ty_cm/100.0, 'tz': tz_cm/100.0,  # Also provide in meters for transform
            'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw,
            'stamp': img_ts  # [BENCHMARK] Gửi kèm timestamp vào queue
        }

        try:
            self.vision_queue.put_nowait(filtered_pose)
        except Full:
            self.vision_queue.get_nowait()
            self.vision_queue.put_nowait(filtered_pose)

    # [NOTE: All the quaternion/matrix/euler methods remain the same - no ROS dependency]
    # Copying from original file would be too long, but they're identical
    # The key changes are only in ROS-specific parts

    def quaternion_to_matrix(self, q):
        x, y, z, w = q
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,      2*x*z + 2*w*y],
            [2*x*y + 2*w*z,         1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x],
            [2*x*z - 2*w*y,         2*y*z + 2*w*x,      1 - 2*x*x - 2*y*y]
        ])

    # ----- Low-level servo calibration helpers (sign + offset around home) -----
    def _apply_sign_around_home(self, deg_calc: float, sign: float) -> float:
        """Invert or keep angle around HOME_DEG based on sign (+1 or -1)."""
        return self.HOME_DEG + float(sign) * (float(deg_calc) - self.HOME_DEG)

    def _apply_output_adjust(self, deg_calc: float, sign: float, offset_deg: float) -> float:
        """Apply sign around home, then add extra offset in degrees."""
        d = self._apply_sign_around_home(deg_calc, sign)
        return d + float(offset_deg)
    
    def matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion [x, y, z, w]"""
        tr = R[0,0] + R[1,1] + R[2,2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (R[2,1] - R[1,2]) / S
            y = (R[0,2] - R[2,0]) / S
            z = (R[1,0] - R[0,1]) / S
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S
        return np.array([x, y, z, w])

    def _rotation_matrix_to_euler(self, R):
        """
        Extract Euler angles (roll, pitch, yaw) from a rotation matrix.
        Convention: Z-Y-X (yaw-pitch-roll).

        Returns:
            np.array([roll, pitch, yaw]) in radians
        """
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0.0

        return np.array([roll, pitch, yaw], dtype=np.float64)

    def calculate_6dof_compensation(self, board_pose_cam, target_pos_base_cm):
        """
        Compute compensation from full 6-DOF board pose.

        Safety behavior:
        - If autobalancing is not enabled/configured, returns (0,0,0) and FIXED_TILT.

        Args:
            board_pose_cam: dict with 'tx','ty','tz' (m) and quaternion 'qx','qy','qz','qw'
            target_pos_base_cm: np.array([x,y,z]) in cm (base frame)

        Returns:
            (compensation_vec_cm, compensated_tilt_deg)
        """
        if not getattr(self, "AUTOBALANCING_ENABLED", False):
            return np.array([0.0, 0.0, 0.0], dtype=np.float64), float(self.FIXED_TILT)

        # Capture reference pose on first use (assumes board is in its "level" reference)
        if self.reference_board_pose is None:
            self.reference_board_pose = {
                'qx': board_pose_cam['qx'],
                'qy': board_pose_cam['qy'],
                'qz': board_pose_cam['qz'],
                'qw': board_pose_cam['qw'],
            }
            return np.array([0.0, 0.0, 0.0], dtype=np.float64), float(self.FIXED_TILT)

        R_ref = self.quaternion_to_matrix([
            self.reference_board_pose['qx'],
            self.reference_board_pose['qy'],
            self.reference_board_pose['qz'],
            self.reference_board_pose['qw'],
        ])
        R_cur = self.quaternion_to_matrix([
            board_pose_cam['qx'], board_pose_cam['qy'], board_pose_cam['qz'], board_pose_cam['qw']
        ])

        # Relative tilt since reference
        R_tilt = R_cur @ R_ref.T
        euler = self._rotation_matrix_to_euler(R_tilt)
        roll_r, pitch_r, yaw_r = float(euler[0]), float(euler[1]), float(euler[2])

        self.drone_attitude['roll'] = float(np.degrees(roll_r))
        self.drone_attitude['pitch'] = float(np.degrees(pitch_r))
        self.drone_attitude['yaw'] = float(np.degrees(yaw_r))

        # --- Position compensation (cm) ---
        comp = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        z_cm = float(target_pos_base_cm[2])

        if self.ROLL_COMPENSATION:
            comp[1] += np.sin(roll_r) * z_cm * self.COMPENSATION_GAIN
        if self.PITCH_COMPENSATION:
            comp[0] += -np.sin(pitch_r) * z_cm * self.COMPENSATION_GAIN
        # yaw compensation intentionally left simple/off by default

        comp = np.clip(comp, -self.MAX_COMPENSATION_CM, self.MAX_COMPENSATION_CM)

        # --- Orientation compensation: wrist tilt from PITCH only (joint 4 / Wrist Roll is locked) ---
        # Roll cannot be executed by the robot, so do not mix it into tilt (avoids conflict with Pitch joint).
        compensated_tilt = float(self.FIXED_TILT - math.degrees(pitch_r) * self.ORIENTATION_COMPENSATION_GAIN)
        compensated_tilt = float(np.clip(
            compensated_tilt,
            self.FIXED_TILT - self.MAX_COMPENSATION_DEG,
            self.FIXED_TILT + self.MAX_COMPENSATION_DEG
        ))

        return comp, compensated_tilt
    
    def update_velocity_estimation(self, pose, current_time):
        """Update velocity estimation from pose history"""
        if not self.PREDICTION_ENABLED:
            return
        
        # Add current pose to history
        pose_entry = {
            'tx': pose['tx'],
            'ty': pose['ty'],
            'tz': pose['tz'],
            'qx': pose['qx'],
            'qy': pose['qy'],
            'qz': pose['qz'],
            'qw': pose['qw'],
            'time': current_time
        }
        self.pose_history.append(pose_entry)
        
        # Need at least 2 poses to estimate velocity
        if len(self.pose_history) < 2:
            return
        
        # Calculate velocity from last two poses
        p1 = self.pose_history[-2]
        p2 = self.pose_history[-1]
        dt = p2['time'] - p1['time']
        
        if dt <= 0.0 or dt > 0.5:  # Skip if invalid or too old
            return
        
        # Linear velocity in BASE frame (cm/s) so prediction is correct when drone yaws
        p1_cam = np.array([p1['tx'], p1['ty'], p1['tz'], 1.0])
        p2_cam = np.array([p2['tx'], p2['ty'], p2['tz'], 1.0])
        p1_base = self.T_calib @ p1_cam
        p2_base = self.T_calib @ p2_cam
        vel_linear_new = ((p2_base[:3] - p1_base[:3]) * 100.0 / dt).astype(np.float64)  # m/s -> cm/s
        
        # Angular velocity (approximate from quaternion difference)
        # Convert quaternions to rotation matrices
        R1 = self.quaternion_to_matrix([p1['qx'], p1['qy'], p1['qz'], p1['qw']])
        R2 = self.quaternion_to_matrix([p2['qx'], p2['qy'], p2['qz'], p2['qw']])
        # Relative rotation
        R_rel = R2 @ R1.T
        # Approximate angular velocity (simplified)
        # For small rotations: omega ≈ angle(R_rel) / dt
        trace = np.trace(R_rel)
        angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        if angle > 0.001:  # Only if significant rotation
            axis = np.array([
                R_rel[2,1] - R_rel[1,2],
                R_rel[0,2] - R_rel[2,0],
                R_rel[1,0] - R_rel[0,1]
            ]) / (2.0 * np.sin(angle))
            vel_angular_new = axis * angle / dt
        else:
            vel_angular_new = np.array([0.0, 0.0, 0.0])
        
        # Smooth velocity with EMA
        alpha = self.VELOCITY_SMOOTHING_ALPHA
        self.velocity_linear = alpha * vel_linear_new + (1.0 - alpha) * self.velocity_linear
        self.velocity_angular = alpha * vel_angular_new + (1.0 - alpha) * self.velocity_angular
    
    def predict_pose(self, base_pose, prediction_time_ms):
        """Predict future pose: velocity is in base frame to avoid tail error when drone yaws."""
        if not self.PREDICTION_ENABLED or prediction_time_ms <= 0.0:
            return base_pose
        
        prediction_time_ms = min(prediction_time_ms, self.MAX_PREDICTION_TIME_MS)
        dt = prediction_time_ms / 1000.0  # seconds
        
        # Current pose (camera frame) → base frame
        p_cam = np.array([base_pose['tx'], base_pose['ty'], base_pose['tz'], 1.0])
        p_base = self.T_calib @ p_cam
        # Predict in base frame (velocity_linear is cm/s in base)
        p_base_pred = p_base[:3] + (self.velocity_linear / 100.0) * dt
        # Base → camera
        p_cam_pred = self.T_calib_inv @ np.append(p_base_pred, 1.0)
        predicted_tx = float(p_cam_pred[0])
        predicted_ty = float(p_cam_pred[1])
        predicted_tz = float(p_cam_pred[2])
        
        # Predict rotation (approximate)
        # For small rotations, use angular velocity
        R_base = self.quaternion_to_matrix([base_pose['qx'], base_pose['qy'], base_pose['qz'], base_pose['qw']])
        # Approximate rotation from angular velocity
        omega_mag = np.linalg.norm(self.velocity_angular)
        if omega_mag > 0.001:
            axis = self.velocity_angular / omega_mag
            angle = omega_mag * dt
            # Rodrigues' rotation formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R_delta = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            R_pred = R_delta @ R_base
            q_pred = self.matrix_to_quaternion(R_pred)
        else:
            q_pred = [base_pose['qx'], base_pose['qy'], base_pose['qz'], base_pose['qw']]
        
        predicted_pose = {
            'tx': predicted_tx,
            'ty': predicted_ty,
            'tz': predicted_tz,
            'qx': q_pred[0],
            'qy': q_pred[1],
            'qz': q_pred[2],
            'qw': q_pred[3],
            'tx_cm': predicted_tx * 100.0,
            'ty_cm': predicted_ty * 100.0,
            'tz_cm': predicted_tz * 100.0,
            'stamp': base_pose.get('stamp', 0.0),
            'predicted': True  # Flag to indicate this is a prediction
        }
        
        return predicted_pose

    def wait_for_vision(self):
        """Wait for vision data (ROS 2 version - uses rclpy.ok())"""
        print("👀 Waiting for ArUco Board...")
        while rclpy.ok():
            if not self.vision_queue.empty():
                print("✅ Vision Acquired! Starting task.")
                self.last_vision_time = time.time()
                return
            time.sleep(0.1)

    # [NOTE: execute_segment, parse_point, and run methods remain largely the same]
    # Only difference is rospy.is_shutdown() -> rclpy.ok()
    # For brevity, I'll include a note that these need to be copied from original
    # and replace rospy.is_shutdown() with rclpy.ok()

    # Copy execute_segment, parse_point, and run methods from original drawing_executor.py
    # Replace: rospy.is_shutdown() -> rclpy.ok()
    # The rest of the logic is identical

    def execute_segment(self, start_pt, end_pt, speed_cm_s):
        """
        Nội suy chuyển động từ start_pt đến end_pt.
        Stroke coordinates: use STROKE_INPUT_METERS (true = meters, false = cm) so duration is correct.
        """
        diff = end_pt[:3] - start_pt[:3]
        dist_same_unit = np.linalg.norm(diff)
        # Distance in cm for duration: if input is meters then *100, else already cm
        dist_cm = (dist_same_unit * 100.0) if self.STROKE_INPUT_METERS else dist_same_unit

        if dist_cm < 0.1: return  # Bỏ qua đoạn quá ngắn (<1mm)

        duration = dist_cm / max(0.1, speed_cm_s)
        steps = int(max(2, duration / self.dt_period))

        for step in range(steps):
            loop_start = time.perf_counter()
            current_time = time.time()

            # 1. Nội suy (Interpolation)
            t = (step + 1) / steps
            current_target_pt = start_pt + (end_pt - start_pt) * t
            self.target_pt_board = current_target_pt

            # 2. Lấy Vision & Đo Queue Latency
            # [BENCHMARK] Đo thời gian lấy từ Queue
            self.profiler.start_timer("Queue_Get_ms")
            vision_updated = False
            if not self.vision_queue.empty():
                self.latest_board_pose = self.vision_queue.get()
                self.last_vision_time = current_time
                vision_updated = True
                # [NEW] Update velocity estimation when new pose arrives
                self.update_velocity_estimation(self.latest_board_pose, current_time)
            t_queue = self.profiler.stop_timer("Queue_Get_ms")

            # [NEW] Enhanced vision loss handling with extrapolation
            vision_lost_time = current_time - self.last_vision_time
            using_extrapolation = False
            active_pose = self.latest_board_pose  # Initialize active_pose
            
            if vision_lost_time > self.vision_timeout:
                # Long vision loss - go to HOME
                print("⚠️ VISION LOST! Hovering...")
                self.servos.apply_angles(self.HOME_POSE)
                if not self.vision_queue.empty():
                    self.latest_board_pose = self.vision_queue.get()
                    self.last_vision_time = time.time()
                    print("✅ Vision Regained!")
                    self.update_velocity_estimation(self.latest_board_pose, current_time)
                    active_pose = self.latest_board_pose
                    break
                time.sleep(0.1)
                continue  # Skip rest of loop iteration
            elif vision_lost_time > (self.EXTRAPOLATION_TIMEOUT_MS / 1000.0) and self.latest_board_pose:
                # Brief vision loss - use extrapolation
                if self.PREDICTION_ENABLED and len(self.pose_history) >= 2:
                    prediction_time_ms = vision_lost_time * 1000.0
                    active_pose = self.predict_pose(self.latest_board_pose, prediction_time_ms)
                    using_extrapolation = True
                    if vision_lost_time > 0.3:  # Only warn if significant delay
                        print(f"⚠️ Using extrapolation (vision delay: {vision_lost_time*1000:.0f}ms)")
            
            # [NEW] Apply prediction to compensate for phase delay
            if active_pose and self.PREDICTION_ENABLED and not using_extrapolation:
                # Calculate phase delay and predict ahead
                vision_ts = active_pose.get('stamp', current_time)
                phase_delay_ms = (current_time - vision_ts) * 1000.0
                if phase_delay_ms > 10.0:  # Only predict if significant delay
                    active_pose = self.predict_pose(active_pose, phase_delay_ms)

            # Khởi tạo biến đo lường
            t_filter = 0.0
            t_calc = 0.0
            t_servo = 0.0
            phase_delay = 0.0
            error_3d = 0.0
            
            val_x = val_reach = val_z = 0.0
            raw_x = raw_y = raw_z = 0.0
            base_raw_x = base_raw_y = base_raw_z = 0.0
            comp_dx = comp_dy = comp_dz = 0.0
            compensated_tilt = self.FIXED_TILT
            p_cam_cm = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            p_base_raw_cm = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            p_base_comp_cm = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            pose_predicted_flag = False

            # [NEW] Use active_pose (may be predicted) if available, otherwise use latest_board_pose
            if 'active_pose' not in locals():
                active_pose = self.latest_board_pose
            
            if active_pose:
                pose_predicted_flag = bool(active_pose.get('predicted', False))
                # [BENCHMARK] Tính Phase Delay (Delay từ lúc chụp ảnh đến lúc xử lý xong tại đây)
                vision_ts = active_pose.get('stamp', current_time)
                phase_delay = (current_time - vision_ts) * 1000.0 # ms

                # Biến đổi tọa độ (using active_pose which may be predicted)
                tx, ty, tz = active_pose['tx'], active_pose['ty'], active_pose['tz']
                q = [active_pose['qx'], active_pose['qy'], 
                     active_pose['qz'], active_pose['qw']]

                rmat = self.quaternion_to_matrix(q)
                T_vision = np.eye(4)
                T_vision[:3, :3] = rmat
                T_vision[:3, 3] = [tx, ty, tz]

                p_cam = T_vision @ current_target_pt
                p_base = self.T_calib @ p_cam

                # Convert for logging (cm)
                p_cam_cm = np.array([p_cam[0] * 100.0, p_cam[1] * 100.0, p_cam[2] * 100.0], dtype=np.float64)
                p_base_raw_cm = np.array([p_base[0] * 100.0, p_base[1] * 100.0, p_base[2] * 100.0], dtype=np.float64)
                base_raw_x, base_raw_y, base_raw_z = float(p_base_raw_cm[0]), float(p_base_raw_cm[1]), float(p_base_raw_cm[2])

                # Start pipeline with base-raw target
                raw_x, raw_y, raw_z = base_raw_x, base_raw_y, base_raw_z

                # 6-DOF compensation (position cm + wrist tilt deg)
                compensation_vec, compensated_tilt = self.calculate_6dof_compensation(
                    active_pose,
                    np.array([raw_x, raw_y, raw_z], dtype=np.float64)
                )
                comp_dx, comp_dy, comp_dz = float(compensation_vec[0]), float(compensation_vec[1]), float(compensation_vec[2])

                raw_x += comp_dx
                raw_y += comp_dy
                raw_z += comp_dz
                p_base_comp_cm = np.array([raw_x, raw_y, raw_z], dtype=np.float64)

                # [BENCHMARK] Đo thời gian chạy bộ lọc (OneEuro)
                self.profiler.start_timer("Filter_Update_ms")
                val_x = self.one_euro_x.update(raw_x, current_time)
                val_y = self.one_euro_y.update(raw_y, current_time)
                val_z = self.one_euro_z.update(raw_z, current_time)
                t_filter = self.profiler.stop_timer("Filter_Update_ms")
                
                # Drawing detection using configurable threshold (optimized: moved from hardcoded value)
                is_drawing = (abs(val_z) < self.DRAWING_THRESHOLD_CM)
                val_reach = val_y if is_drawing else val_y + abs(self.LIFT_HEIGHT_CM)
                
                # [BENCHMARK] Tính Tracking Error (Sai số giữa Lệnh điều khiển và Input thô Vision)
                # Điều này giúp đánh giá xem bộ lọc có làm lệch quá nhiều so với thực tế không
                error_3d = np.sqrt((val_x - raw_x)**2 + (val_y - raw_y)**2 + (val_z - raw_z)**2)

                # 3. Tính IK
                self.profiler.start_timer("Filter_Calc_ms")
                target_angles = self.ik.solve_ik(val_x, val_reach, val_z, compensated_tilt)
                t_calc = self.profiler.stop_timer("Filter_Calc_ms")
                ik_success = 1.0 if target_angles else 0.0

                # 4. Gửi lệnh Servo (sau khi hiệu chỉnh sign + offset quanh HOME)
                if target_angles:
                    # Áp dụng hiệu chỉnh giống như node standalone (sign_shoulder/elbow, offset_*_deg)
                    # DOF3 (elbow) now uses SIGN_ELBOW directly; inversion is configured only in robot_config.yaml.
                    calibrated = [
                        self._apply_output_adjust(target_angles[0], self.SIGN_BASE, self.OFFSET_BASE_DEG),
                        self._apply_output_adjust(target_angles[1], self.SIGN_SHOULDER, self.OFFSET_SHOULDER_DEG),
                        self._apply_output_adjust(target_angles[2], self.SIGN_ELBOW, self.OFFSET_ELBOW_DEG),
                        self._apply_output_adjust(target_angles[3], self.SIGN_WRIST, self.OFFSET_WRIST_DEG),
                    ]
                    self.profiler.start_timer("Servo_Write_ms")
                    smoothed = [self.joint_smoothers[i].update(calibrated[i]) for i in range(4)]
                    self.servos.apply_angles(smoothed)
                    t_servo = self.profiler.stop_timer("Servo_Write_ms")
                else:
                    t_servo = 0.0
                    # Helpful diagnostic when the arm doesn't move (often due to unreachable target)
                    if self.DEBUG_LOG_WAYPOINTS and (step % self.DEBUG_LOG_EVERY_N_STEPS == 0 or step == steps - 1):
                        print(
                            f"[IK_FAIL] step {step+1}/{steps} "
                            f"base_filt_cm=({val_x:+.2f},{val_reach:+.2f},{val_z:+.2f}) "
                            f"tilt_deg={compensated_tilt:+.1f}"
                        )

                # 5. Ghi log vào CSV
                log_kwargs = dict(
                    Timestamp=current_time,
                    Loop_Dt_ms=(time.perf_counter() - loop_start) * 1000.0,
                    Queue_Get_ms=t_queue,
                    Filter_Update_ms=t_filter,
                    Filter_Calc_ms=t_calc,
                    Servo_Write_ms=t_servo,
                    Phase_Delay_ms=phase_delay,
                    Tracking_Error_3D_cm=error_3d,
                    Command_X=val_x, Command_Y=val_reach, Command_Z=val_z,
                    Raw_Vision_X=base_raw_x, Raw_Vision_Y=base_raw_y, Raw_Vision_Z=base_raw_z,
                    Target_X=current_target_pt[0]*100, 
                    Target_Y=current_target_pt[1]*100, 
                    Target_Z=current_target_pt[2]*100,
                    IK_Success=ik_success,
                    Seg_Step=step,
                    Seg_Steps=steps,
                    Using_Extrapolation=1.0 if using_extrapolation else 0.0,
                    Pose_Predicted=1.0 if pose_predicted_flag else 0.0,
                )

                if self.DEBUG_LOG_CSV_EXTRA:
                    log_kwargs.update(dict(
                        Target_Board_X_cm=float(current_target_pt[0] * 100.0),
                        Target_Board_Y_cm=float(current_target_pt[1] * 100.0),
                        Target_Board_Z_cm=float(current_target_pt[2] * 100.0),
                        Target_Cam_X_cm=float(p_cam_cm[0]),
                        Target_Cam_Y_cm=float(p_cam_cm[1]),
                        Target_Cam_Z_cm=float(p_cam_cm[2]),
                        Target_Base_Raw_X_cm=float(p_base_raw_cm[0]),
                        Target_Base_Raw_Y_cm=float(p_base_raw_cm[1]),
                        Target_Base_Raw_Z_cm=float(p_base_raw_cm[2]),
                        Comp_DX_cm=float(comp_dx),
                        Comp_DY_cm=float(comp_dy),
                        Comp_DZ_cm=float(comp_dz),
                        Target_Base_Comp_X_cm=float(p_base_comp_cm[0]),
                        Target_Base_Comp_Y_cm=float(p_base_comp_cm[1]),
                        Target_Base_Comp_Z_cm=float(p_base_comp_cm[2]),
                        Target_Base_Filt_X_cm=float(val_x),
                        Target_Base_Filt_Y_cm=float(val_reach),
                        Target_Base_Filt_Z_cm=float(val_z),
                        Tilt_Fixed_deg=float(self.FIXED_TILT),
                        Tilt_Comp_deg=float(compensated_tilt),
                        Drone_Roll_deg=float(self.drone_attitude.get('roll', 0.0)),
                        Drone_Pitch_deg=float(self.drone_attitude.get('pitch', 0.0)),
                        Drone_Yaw_deg=float(self.drone_attitude.get('yaw', 0.0)),
                    ))

                self.profiler.log_data(**log_kwargs)

                # [DEBUG] Console waypoint trace (rate limited)
                if self.DEBUG_LOG_WAYPOINTS and (step % self.DEBUG_LOG_EVERY_N_STEPS == 0 or step == steps - 1):
                    print(
                        f"[EE_WP] step {step+1:>3}/{steps:<3} "
                        f"board_cm=({current_target_pt[0]*100:+6.2f},{current_target_pt[1]*100:+6.2f},{current_target_pt[2]*100:+6.2f}) "
                        f"base_raw_cm=({base_raw_x:+7.2f},{base_raw_y:+7.2f},{base_raw_z:+7.2f}) "
                        f"comp_cm=({comp_dx:+6.2f},{comp_dy:+6.2f},{comp_dz:+6.2f}) "
                        f"base_filt_cm=({val_x:+7.2f},{val_reach:+7.2f},{val_z:+7.2f}) "
                        f"tilt_deg={compensated_tilt:+6.1f} "
                        f"roll/pitch=({self.drone_attitude.get('roll',0.0):+5.1f},{self.drone_attitude.get('pitch',0.0):+5.1f}) "
                        f"pred={int(pose_predicted_flag)} extrap={int(using_extrapolation)}"
                    )

            # 6. Duy trì 50Hz
            now = time.time()
            sleep_needed = self.next_wake_time - now
            if sleep_needed > 0:
                time.sleep(sleep_needed)
            self.next_wake_time += self.dt_period

    def parse_point(self, raw_pt):
        try:
            pt = np.array(raw_pt, dtype=np.float64)
            if pt.ndim != 1: return None
            if pt.shape[0] == 3: pt = np.append(pt, 1.0) 
            if pt.shape[0] != 4: return None
            return pt
        except: return None

    def run(self):
        print("🚀 Executor Started. Moving to HOME...")
        self.servos.apply_angles(self.HOME_POSE)
        self.ik = KinematicsSolver()

        self.wait_for_vision()

        self.next_wake_time = time.time() + self.dt_period
        
        print(f"🖌️  Total Strokes: {len(self.strokes)}")

        # Vị trí khởi tạo giả định (Board Frame)
        current_board_pos = np.array([0.0, 0.0, -0.05, 1.0]) 

        for stroke_idx, stroke in enumerate(self.strokes):
            print(f"   🔹 Stroke {stroke_idx+1}/{len(self.strokes)}")
            
            valid_points = []
            for raw in stroke:
                p = self.parse_point(raw)
                if p is not None: valid_points.append(p)
            
            if len(valid_points) < 2: continue

            start_pt = valid_points[0]
            
            # --- PHASE 1: BAY ĐẾN ĐIỂM ĐẦU ---
            lift_start = current_board_pos.copy()
            lift_start[2] = -abs(self.LIFT_HEIGHT_CM/100.0)
            self.execute_segment(current_board_pos, lift_start, self.AIR_SPEED_CM_S)
            
            lift_end = start_pt.copy()
            lift_end[2] = -abs(self.LIFT_HEIGHT_CM/100.0)
            self.execute_segment(lift_start, lift_end, self.AIR_SPEED_CM_S)
            
            self.execute_segment(lift_end, start_pt, self.DRAW_SPEED_CM_S)
            current_board_pos = start_pt

            # --- PHASE 2: VẼ NÉT ---
            for i in range(1, len(valid_points)):
                next_pt = valid_points[i]
                self.execute_segment(current_board_pos, next_pt, self.DRAW_SPEED_CM_S)
                current_board_pos = next_pt

            # --- PHASE 3: NHẤC BÚT KẾT THÚC NÉT ---
            lift_finish = current_board_pos.copy()
            lift_finish[2] = -abs(self.LIFT_HEIGHT_CM/100.0)
            self.execute_segment(current_board_pos, lift_finish, self.DRAW_SPEED_CM_S)
            current_board_pos = lift_finish

        print("✅ Task Completed!")
        self.servos.apply_angles(self.HOME_POSE)
        # [BENCHMARK] In báo cáo tóm tắt
        self.profiler.print_summary()


def main(args=None):
    """ROS 2 main function"""
    rclpy.init(args=args)
    artist = None
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='json', choices=['json', 'tri', 'square', 'rect', 'penta', 'hexa', 'circle', 'star', 'line'])
    parser.add_argument('--size', type=float, default=7.0)
    parser.add_argument('--scale', type=float, default=1.0)
    args_parsed = parser.parse_args(args)
    
    try:
        gen = ShapeGenerator(safe_zone_cm=args_parsed.size)
    except NameError:
        rclpy.shutdown()
        sys.exit(1)
    
    input_data = None
    is_file = False
    if args_parsed.mode == 'json':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        input_data = os.path.join(current_dir, '..', 'drone_task.json')
        is_file = True
    elif args_parsed.mode == 'tri':    input_data = gen.polygon(3, args_parsed.scale)
    elif args_parsed.mode == 'square': input_data = gen.rectangle(args_parsed.scale, args_parsed.scale)
    elif args_parsed.mode == 'rect':   input_data = gen.rectangle(args_parsed.scale, args_parsed.scale*0.6)
    elif args_parsed.mode == 'penta':  input_data = gen.polygon(5, args_parsed.scale)
    elif args_parsed.mode == 'hexa':   input_data = gen.polygon(6, args_parsed.scale)
    elif args_parsed.mode == 'circle': input_data = gen.circle(36, args_parsed.scale)
    elif args_parsed.mode == 'star':   input_data = gen.star(args_parsed.scale)
    elif args_parsed.mode == 'line':   input_data = gen.line(0, args_parsed.scale)

    try:
        print(f"🎨 STARTING: Mode={args_parsed.mode.upper()} | Size={args_parsed.size}cm")
        artist = PBVSArtist(input_data, is_file=is_file)
        
        # ROS 2: Use executor to handle callbacks while running main loop
        executor = MultiThreadedExecutor()
        executor.add_node(artist)
        
        # Run in separate thread to allow main loop to execute
        import threading
        def spin_thread():
            executor.spin()
        
        spin_thread_obj = threading.Thread(target=spin_thread, daemon=True)
        spin_thread_obj.start()
        
        # Run main drawing loop
        artist.run()
        
    except KeyboardInterrupt:
        pass
    finally:
        if artist is not None:
            artist.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
