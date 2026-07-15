#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ROS 2 Humble Version
#
# Purpose:
# - Demonstrate "angle compensation" + tracking by driving the arm toward
#   a single tracking point: the board CENTER (board frame origin).
# - Useful for tuning installation / calibration parameters without drawing paths.
#
# Inputs:
# - /target_pose (geometry_msgs/PoseStamped): board pose in camera frame (from vision_node_ros2.py)
#
# Outputs:
# - Servo commands via drivers/i2c_manager.py ServoController
# - /pbvs/monitor_metrics (std_msgs/Float32MultiArray): [delay_ms, err_cm, vec_cm(x,y,z), target_base_cm(x,y,z), cmd_base_cm(x,y,z), roll_deg, pitch_deg, yaw_deg]
#
# Notes:
# - This node uses the same core pipeline as drawing_executor_ros2.py:
#   T_vision (from PoseStamped) -> T_cam_to_base calibration -> CM -> 6DOF compensation -> filters -> IK -> servo calibration.

import os
import sys
import time
import yaml
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.kinematics import KinematicsSolver
from core.filters import EMASmoother, KalmanFilter1D, OutlierRejector, OneEuroFilter
from drivers.i2c_manager import ServoController


class CenterTrackingExecutor(Node):
    def __init__(self):
        super().__init__("center_tracking_executor")

        config_path = None
        try:
            from ament_index_python.packages import get_package_share_directory
            share_dir = get_package_share_directory('visual_servoing')
            config_path = os.path.join(share_dir, 'config', 'robot_config.yaml')
        except Exception:
            pass

        if not config_path or not os.path.exists(config_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.abspath(os.path.join(current_dir, "..", "config", "robot_config.yaml"))
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))

        with open(config_path, "r") as f:
            self.full_config = yaml.safe_load(f) or {}
        self.get_logger().info(f"Loaded config: {config_path}")

        # ---- config ----
        ctrl_cfg = (self.full_config.get("control") or {})
        geom_cfg = (ctrl_cfg.get("geometry") or {})
        pred_cfg = (ctrl_cfg.get("prediction") or {})
        bal_cfg = (ctrl_cfg.get("autobalancing") or {})
        dbg_cfg = (ctrl_cfg.get("debug") or {})

        self.dt_period = float(ctrl_cfg.get("dt_period", 0.02))  # 50 Hz default
        self.FIXED_TILT = float(geom_cfg.get("fixed_tilt", -35.0))

        self.DEBUG_LOG_EVERY_N = int(max(1, dbg_cfg.get("log_every_n_steps", 10)))

        self.PREDICTION_ENABLED = bool(pred_cfg.get("enabled", True))
        self.MAX_PREDICTION_TIME_MS = float(pred_cfg.get("max_prediction_time_ms", 50.0))

        self.AUTOBALANCING_ENABLED = bool(bal_cfg.get("enabled", False))
        self.COMPENSATION_GAIN = float(bal_cfg.get("compensation_gain", 1.0))
        self.ORIENTATION_COMPENSATION_GAIN = float(bal_cfg.get("orientation_compensation_gain", 1.0))
        self.MAX_COMPENSATION_CM = float(bal_cfg.get("max_compensation_cm", 5.0))
        self.MAX_COMPENSATION_DEG = float(bal_cfg.get("max_compensation_deg", 15.0))
        self.ROLL_COMPENSATION = bool(bal_cfg.get("roll_compensation", True))
        self.PITCH_COMPENSATION = bool(bal_cfg.get("pitch_compensation", True))
        self.YAW_COMPENSATION = bool(bal_cfg.get("yaw_compensation", False))

        # ---- servo calibration ----
        robot_cfg = (self.full_config.get("robot") or {})
        servo_cal = (robot_cfg.get("servo_calibration") or {})
        self.HOME_DEG = float(servo_cal.get("home_deg", 90.0))
        self.SIGN_BASE = float(servo_cal.get("sign_base", 1.0))
        self.SIGN_SHOULDER = float(servo_cal.get("sign_shoulder", -1.0))
        self.SIGN_ELBOW = float(servo_cal.get("sign_elbow", -1.0))
        self.SIGN_WRIST = float(servo_cal.get("sign_wrist", 1.0))
        self.OFFSET_BASE_DEG = float(servo_cal.get("offset_base_deg", 0.0))
        self.OFFSET_SHOULDER_DEG = float(servo_cal.get("offset_shoulder_deg", 30.0))
        self.OFFSET_ELBOW_DEG = float(servo_cal.get("offset_elbow_deg", -30.0))
        self.OFFSET_WRIST_DEG = float(servo_cal.get("offset_wrist_deg", 0.0))

        serv_cfg = (robot_cfg.get("servos") or {})
        fixed_channels = serv_cfg.get("fixed_channels", [])
        fixed_degs_raw = list(serv_cfg.get("fixed_degs", [100.0, 45.0]))
        pen_deg = serv_cfg.get("pen_deg", None)
        if pen_deg is not None and len(fixed_degs_raw) >= 2:
            fixed_degs = [float(fixed_degs_raw[0]), float(pen_deg)]
        else:
            fixed_degs = [float(d) for d in fixed_degs_raw]
        off_channels = serv_cfg.get("off_channels", [])
        shoulder_mirror_enabled = bool(serv_cfg.get("shoulder_mirror_enabled", False))
        shoulder_mirror_channel = serv_cfg.get("shoulder_mirror_channel", None)
        shoulder_mirror_angle_max = float(serv_cfg.get("shoulder_mirror_angle_max", 180.0))

        self.servos = ServoController(
            fixed_channels=fixed_channels,
            fixed_degs=fixed_degs,
            off_channels=off_channels,
            shoulder_mirror_enabled=shoulder_mirror_enabled,
            shoulder_mirror_channel=shoulder_mirror_channel,
            shoulder_mirror_angle_max=shoulder_mirror_angle_max,
        )

        # ---- calibration matrix ----
        calib_path = None
        try:
            from ament_index_python.packages import get_package_share_directory
            share_dir = get_package_share_directory('visual_servoing')
            calib_path = os.path.join(share_dir, 'config', 'T_cam_to_base_THEORETICAL.npy')
        except Exception:
            pass

        if not calib_path or not os.path.exists(calib_path):
            calib_path = os.path.abspath(os.path.join(current_dir, "..", "config", "T_cam_to_base_THEORETICAL.npy"))

        self.T_calib = np.load(calib_path)
        self.get_logger().info(f"Loaded calibration: {calib_path}")

        # ---- IK + filters (cm) ----
        self.ik = KinematicsSolver()
        self.outlier_x = OutlierRejector(max_jump=5.0)
        self.outlier_y = OutlierRejector(max_jump=5.0)
        self.outlier_z = OutlierRejector(max_jump=5.0)
        self.kalman_x = KalmanFilter1D(R=0.5, Q=0.01)
        self.kalman_y = KalmanFilter1D(R=0.5, Q=0.01)
        self.kalman_z = KalmanFilter1D(R=0.5, Q=0.01)
        self.one_euro_x = OneEuroFilter(min_cutoff=0.01, beta=0.01)
        self.one_euro_y = OneEuroFilter(min_cutoff=0.01, beta=0.01)
        self.one_euro_z = OneEuroFilter(min_cutoff=0.01, beta=0.01)

        self.joint_smoothers = [EMASmoother(0.2) for _ in range(4)]
        for s in self.joint_smoothers:
            s.value = self.HOME_DEG

        # ---- tracking target: board center on drawing plane (board frame) ----
        # Match drawing_executor_ros2.py, which uses a board-plane Z around -0.05 m.
        # If we use Z=0.0 here, the transformed target often ends up far "behind"
        # the physical board and below the IK floor (z_floor), so the arm saturates
        # and appears not to move with the table. Using -0.05 keeps targets on the
        # same drawing surface the PBVS executor uses.
        self.target_center_board = np.array([0.0, 0.0, -0.05, 1.0], dtype=np.float64)

        # ---- state ----
        self.latest_pose = None
        self.reference_board_pose = None
        self.drone_attitude = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self.last_cmd_base_cm = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.last_target_base_cm = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.step = 0

        # ---- ROS I/O ----
        self.sub = self.create_subscription(PoseStamped, "/target_pose", self._on_pose, 1)
        self.pub_metrics = self.create_publisher(Float32MultiArray, "/pbvs/monitor_metrics", 1)

        # periodic control loop
        self.timer = self.create_timer(self.dt_period, self._tick)

        self.get_logger().info("Center tracking executor started.")

    # ---------- math helpers ----------
    def quaternion_to_matrix(self, q):
        x, y, z, w = q
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,      2*x*z + 2*w*y],
            [2*x*y + 2*w*z,         1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x],
            [2*x*z - 2*w*y,         2*y*z + 2*w*x,      1 - 2*x*x - 2*y*y]
        ], dtype=np.float64)

    def _rotation_matrix_to_euler(self, R):
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

    # ---------- servo calibration helpers ----------
    def _apply_sign_around_home(self, deg_calc: float, sign: float) -> float:
        return self.HOME_DEG + float(sign) * (float(deg_calc) - self.HOME_DEG)

    def _apply_output_adjust(self, deg_calc: float, sign: float, offset_deg: float) -> float:
        return self._apply_sign_around_home(deg_calc, sign) + float(offset_deg)

    # ---------- 6DOF compensation ----------
    def calculate_6dof_compensation(self, board_pose_cam, target_pos_base_cm):
        if not self.AUTOBALANCING_ENABLED:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64), float(self.FIXED_TILT)

        if self.reference_board_pose is None:
            self.reference_board_pose = {
                "qx": board_pose_cam["qx"],
                "qy": board_pose_cam["qy"],
                "qz": board_pose_cam["qz"],
                "qw": board_pose_cam["qw"],
            }
            return np.array([0.0, 0.0, 0.0], dtype=np.float64), float(self.FIXED_TILT)

        R_ref = self.quaternion_to_matrix([
            self.reference_board_pose["qx"],
            self.reference_board_pose["qy"],
            self.reference_board_pose["qz"],
            self.reference_board_pose["qw"],
        ])
        R_cur = self.quaternion_to_matrix([
            board_pose_cam["qx"], board_pose_cam["qy"], board_pose_cam["qz"], board_pose_cam["qw"]
        ])

        R_tilt = R_cur @ R_ref.T
        euler = self._rotation_matrix_to_euler(R_tilt)
        roll_r, pitch_r, yaw_r = float(euler[0]), float(euler[1]), float(euler[2])
        self.drone_attitude["roll"] = float(np.degrees(roll_r))
        self.drone_attitude["pitch"] = float(np.degrees(pitch_r))
        self.drone_attitude["yaw"] = float(np.degrees(yaw_r))

        comp = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        z_cm = float(target_pos_base_cm[2])
        if self.ROLL_COMPENSATION:
            comp[1] += np.sin(roll_r) * z_cm * self.COMPENSATION_GAIN
        if self.PITCH_COMPENSATION:
            comp[0] += -np.sin(pitch_r) * z_cm * self.COMPENSATION_GAIN
        comp = np.clip(comp, -self.MAX_COMPENSATION_CM, self.MAX_COMPENSATION_CM)

        # tilt compensation (keep pen more normal to board)
        board_normal_cam = R_cur @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        camera_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        cos_tilt = float(np.clip(np.dot(board_normal_cam, camera_z), -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(cos_tilt)))
        if board_normal_cam[0] > 0:
            tilt_deg = -tilt_deg

        compensated_tilt = float(self.FIXED_TILT + tilt_deg * self.ORIENTATION_COMPENSATION_GAIN)
        compensated_tilt = float(np.clip(
            compensated_tilt,
            self.FIXED_TILT - self.MAX_COMPENSATION_DEG,
            self.FIXED_TILT + self.MAX_COMPENSATION_DEG,
        ))

        return comp, compensated_tilt

    # ---------- ROS callbacks ----------
    def _on_pose(self, msg: PoseStamped):
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
        self.latest_pose = {
            "stamp": stamp_sec,
            "tx": float(msg.pose.position.x),
            "ty": float(msg.pose.position.y),
            "tz": float(msg.pose.position.z),
            "qx": float(msg.pose.orientation.x),
            "qy": float(msg.pose.orientation.y),
            "qz": float(msg.pose.orientation.z),
            "qw": float(msg.pose.orientation.w),
        }

    def _tick(self):
        self.step += 1
        if not self.latest_pose:
            return

        now = time.time()
        phase_delay_ms = (now - float(self.latest_pose.get("stamp", now))) * 1000.0

        # Build T_vision from current board pose in camera frame
        q = [self.latest_pose["qx"], self.latest_pose["qy"], self.latest_pose["qz"], self.latest_pose["qw"]]
        R = self.quaternion_to_matrix(q)
        T_vision = np.eye(4, dtype=np.float64)
        T_vision[:3, :3] = R
        T_vision[:3, 3] = [self.latest_pose["tx"], self.latest_pose["ty"], self.latest_pose["tz"]]

        # target: board center -> camera -> base
        p_cam = T_vision @ self.target_center_board
        p_base = self.T_calib @ p_cam
        target_base_cm = np.array([p_base[0] * 100.0, p_base[1] * 100.0, p_base[2] * 100.0], dtype=np.float64)
        self.last_target_base_cm = target_base_cm.copy()

        # compensation (cm + tilt deg)
        comp_vec, compensated_tilt = self.calculate_6dof_compensation(self.latest_pose, target_base_cm)
        target_comp_cm = target_base_cm + comp_vec

        # filter chain (cm)
        x = self.outlier_x.check(float(target_comp_cm[0]))
        y = self.outlier_y.check(float(target_comp_cm[1]))
        z = self.outlier_z.check(float(target_comp_cm[2]))

        x = self.kalman_x.update(x)
        y = self.kalman_y.update(y)
        z = self.kalman_z.update(z)

        cmd_x = self.one_euro_x.update(x, now)
        cmd_y = self.one_euro_y.update(y, now)
        cmd_z = self.one_euro_z.update(z, now)
        cmd_base_cm = np.array([cmd_x, cmd_y, cmd_z], dtype=np.float64)
        self.last_cmd_base_cm = cmd_base_cm.copy()

        # vector + error
        vec_cm = target_base_cm - cmd_base_cm
        err_cm = float(np.linalg.norm(vec_cm))

        # IK + servo
        angles = self.ik.solve_ik(float(cmd_x), float(cmd_y), float(cmd_z), float(compensated_tilt))
        if angles is not None:
            calibrated = [
                self._apply_output_adjust(angles[0], self.SIGN_BASE, self.OFFSET_BASE_DEG),
                self._apply_output_adjust(angles[1], self.SIGN_SHOULDER, self.OFFSET_SHOULDER_DEG),
                self._apply_output_adjust(angles[2], self.SIGN_ELBOW, self.OFFSET_ELBOW_DEG),
                self._apply_output_adjust(angles[3], self.SIGN_WRIST, self.OFFSET_WRIST_DEG),
            ]
            smoothed = [self.joint_smoothers[i].update(calibrated[i]) for i in range(4)]
            self.servos.apply_angles(smoothed)

        # publish monitor metrics
        m = Float32MultiArray()
        m.data = [
            float(phase_delay_ms),
            float(err_cm),
            float(vec_cm[0]), float(vec_cm[1]), float(vec_cm[2]),
            float(target_base_cm[0]), float(target_base_cm[1]), float(target_base_cm[2]),
            float(cmd_base_cm[0]), float(cmd_base_cm[1]), float(cmd_base_cm[2]),
            float(self.drone_attitude.get("roll", 0.0)),
            float(self.drone_attitude.get("pitch", 0.0)),
            float(self.drone_attitude.get("yaw", 0.0)),
        ]
        self.pub_metrics.publish(m)

        # lightweight console log
        if (self.step % self.DEBUG_LOG_EVERY_N) == 0:
            self.get_logger().info(
                "delay_ms={:.0f} err_cm={:.2f} target_cm=({:+.2f},{:+.2f},{:+.2f}) cmd_cm=({:+.2f},{:+.2f},{:+.2f}) roll/pitch=({:+.1f},{:+.1f})".format(
                    phase_delay_ms,
                    err_cm,
                    target_base_cm[0], target_base_cm[1], target_base_cm[2],
                    cmd_base_cm[0], cmd_base_cm[1], cmd_base_cm[2],
                    float(self.drone_attitude.get("roll", 0.0)),
                    float(self.drone_attitude.get("pitch", 0.0)),
                )
            )


def main(args=None):
    rclpy.init(args=args)
    node = CenterTrackingExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
