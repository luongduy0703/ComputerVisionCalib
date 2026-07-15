#!/usr/bin/env python3
import numpy as np
import math

# --- THÔNG SỐ THỰC TẾ ---
TILT_DEG = 40.0          # Camera chúi xuống 40 độ
YAW_DEG = 0.0           # Bù lệch ngang 
CAM_HEIGHT = 0.055       # 5.5 cm
CAM_OFFSET_X = 0.0       
CAM_OFFSET_Y = 0.105      # 10.5 cm 

def create_theoretical_matrix():
    alpha = math.radians(TILT_DEG)
    gamma = math.radians(YAW_DEG)
    
    # 1. Trục Z của Camera (Trục quang học): Hướng về phía trước (Y+) và chúi xuống (Z-)
    z_cam = np.array([
        math.sin(gamma) * math.cos(alpha),
        math.cos(gamma) * math.cos(alpha),
        -math.sin(alpha)
    ])
    
    # 2. Trục X của Camera: Hướng sang phải (X+) và hơi xoay theo Yaw
    x_cam = np.array([
        math.cos(gamma),
        -math.sin(gamma),
        0.0
    ])
    
    # 3. Trục Y của Camera: Vuông góc với X và Z
    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / np.linalg.norm(y_cam)
    
    # Tạo ma trận xoay R
    R = np.array([x_cam, y_cam, z_cam]).T
    
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = [CAM_OFFSET_X, CAM_OFFSET_Y, CAM_HEIGHT]
    
    print(f"✅ Ma trận mới với Yaw={YAW_DEG} và Tilt={TILT_DEG}")
    print(np.round(T, 4))
    np.save("T_cam_to_base_THEORETICAL.npy", T) 

if __name__ == "__main__":
    create_theoretical_matrix()
