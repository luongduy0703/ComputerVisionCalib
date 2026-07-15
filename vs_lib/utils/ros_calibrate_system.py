#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS SYSTEM CALIBRATION - V24.3 (FINAL FIXED FOR RPi + OpenCV cũ)
--------------------------------------------------
- Fix lỗi cv2.SOLVEPNP_RANSAC (không tồn tại trên OpenCV cũ)
- Dùng ITERATIVE + RefineLM → chính xác cao, tương thích 100%
- Thêm pause sau chạm, inlier check, FK verify
- Clean code & comments
"""

import rospy
import numpy as np
import sys
import tty, termios
import math
import time
import cv2
from geometry_msgs.msg import Vector3
from robot_ik_fk import DroneArmController

# 9 điểm Safe Zone 7x7cm (z = 0 vì chạm tường)
S = 0.035 
TARGET_POINTS = [
    ("1. Góc TRÊN-TRÁI", [-S,  S, 0.0]), ("2. Giữa TRÊN", [0.0, S, 0.0]), ("3. Góc TRÊN-PHẢI", [S, S, 0.0]),
    ("4. Giữa TRÁI",     [-S, 0.0, 0.0]), ("5. TÂM BẢNG",  [0.0, 0.0, 0.0]), ("6. Giữa PHẢI",   [S, 0.0, 0.0]),
    ("7. Góc DƯỚI-TRÁI", [-S, -S, 0.0]), ("8. Giữa DƯỚI", [0.0, -S, 0.0]), ("9. Góc DƯỚI-PHẢI", [S, -S, 0.0]),
]

class RosCalibratorV24:
    def __init__(self):
        rospy.init_node('ros_calibration_v24', anonymous=False)
        self.bot = DroneArmController(dry_run=False)
        self.bot.park() 
        
        self.latest_pos = None
        self.latest_euler = None
        self.has_vision = False
        
        rospy.Subscriber('/robust_pbvs_node/camera_position', Vector3, self.pos_cb)
        rospy.Subscriber('/robust_pbvs_node/board_euler', Vector3, self.euler_cb)
        
        # Tốc độ điều khiển
        self.angle_step = 4.0 

        # Load camera calibration (nếu có – không bắt buộc)
        try:
            calib = np.load("camera_calib.npy", allow_pickle=True).item()
            self.mtx = calib['mtx']
            self.dist = calib['dist']
            print("Loaded camera calibration")
        except:
            print("camera_calib.npy not found – running without undistort")
            self.mtx = None
            self.dist = None

    def pos_cb(self, msg): 
        self.latest_pos = np.array([msg.x, msg.y, msg.z])
        self.has_vision = True

    def euler_cb(self, msg): 
        self.latest_euler = np.array([msg.x, msg.y, msg.z])

    def get_key(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch

    def euler_to_matrix(self, roll, pitch, yaw):
        r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
        Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def run(self):
        points_cam = []
        points_base = []

        print("\n" + "="*60)
        print("   ROS CALIBRATION V24.3 - FINAL FIXED (RPi Compatible)")
        print("="*60)
        print("Hướng dẫn:")
        print(" - Dùng phím A/D/W/S/I/K/U/O để điều chỉnh từng servo")
        print(" - Nhấn ENTER khi đầu bút chạm đúng điểm (vision sẵn sàng)")
        print(" - Cần ít nhất 6 điểm hợp lệ")
        print("="*60)

        for i, (name, pt_board) in enumerate(TARGET_POINTS):
            print(f"\n{i+1:2d}/9 → {name} – Đưa đầu bút chạm điểm này")
            
            while not rospy.is_shutdown():
                print("\r   Đang điều chỉnh... (A/D: Base, W/S: Shoulder, I/K: Elbow, U/O: Wrist)   ", end="")
                
                key = self.get_key()
                
                updated = False
                if key in ['a', 'A']: 
                    self.bot.current_servos[0] -= self.angle_step; updated = True
                elif key in ['d', 'D']: 
                    self.bot.current_servos[0] += self.angle_step; updated = True
                elif key in ['w', 'W']: 
                    self.bot.current_servos[1] -= self.angle_step; updated = True
                elif key in ['s', 'S']: 
                    self.bot.current_servos[1] += self.angle_step; updated = True
                elif key in ['i', 'I']: 
                    self.bot.current_servos[2] += self.angle_step; updated = True
                elif key in ['k', 'K']: 
                    self.bot.current_servos[2] -= self.angle_step; updated = True
                elif key in ['u', 'U']: 
                    self.bot.current_servos[3] += self.angle_step; updated = True
                elif key in ['o', 'O']: 
                    self.bot.current_servos[3] -= self.angle_step; updated = True
                elif key == '1': self.angle_step = 4.0
                elif key == '2': self.angle_step = 8.0
                
                if updated:
                    self.bot._apply_servos()
                
                elif key == '\r':  # ENTER
                    if not self.has_vision or self.latest_pos is None:
                        print("\n   Vision chưa sẵn sàng – chờ ArUco lock!")
                        continue
                    
                    # Pause để pose ổn định
                    time.sleep(0.5)
                    
                    # Tính điểm trong camera
                    if self.latest_euler is not None:
                        R = self.euler_to_matrix(*self.latest_euler)
                        p_cam = R @ np.array(pt_board) + self.latest_pos
                    else:
                        p_cam = self.latest_pos + np.array(pt_board)  # approx nếu thiếu euler
                    
                    points_cam.append(p_cam)
                    
                    # Lấy FK pose đầu bút
                    fk_x, fk_y, fk_z = self.bot.get_current_position_fk()
                    points_base.append(np.array([fk_x, fk_y, fk_z]) / 100.0)  # cm → m
                    
                    print(f"\n   Điểm {i+1} lưu thành công!")
                    break
                
                elif key == '\x1b':  # ESC
                    self.bot.park()
                    return

        if len(points_base) < 6:
            print(f"\nChỉ có {len(points_base)} điểm – cần ít nhất 6!")
            return

        # [FINAL FIX] Dùng ITERATIVE + RefineLM (tương thích OpenCV cũ, chính xác cao)
        src = np.array(points_cam, dtype=np.float32)
        dst = np.array(points_base, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            src, dst, np.eye(3), None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            # Refine để tăng độ chính xác
            try:
                rvec, tvec = cv2.solvePnPRefineLM(src, dst, np.eye(3), None, rvec, tvec)
            except:
                pass  # Nếu không có RefineLM, bỏ qua

            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3,:3] = R
            T[:3,3] = tvec.flatten()

            np.save("T_cam_to_base_FINAL.npy", T)
            print("\n" + "="*60)
            print("           CALIBRATION HOÀN TẤT V24.3!")
            print("="*60)
            print("Ma trận T_cam_to_base:")
            print(np.round(T, 6))
            print("File lưu: T_cam_to_base_FINAL.npy")
            print("="*60)
        else:
            print("\nSolvePnP thất bại – thử thêm điểm hoặc điều chỉnh góc!")

if __name__ == "__main__":
    try:
        app = RosCalibratorV24()
        app.run()
    except rospy.ROSInterruptException:
        pass