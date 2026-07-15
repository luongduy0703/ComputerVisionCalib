#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AeroScript - Headless Pen Tip Coordinate Extractor (Raspberry Pi 4 Optimized)
=============================================================================
- Chạy không cần GUI (headless) để tiết kiệm CPU cho Pi 4.
- Hỗ trợ tự động phát hiện và load model NCNN, TFLite hoặc PyTorch.
- Xuất trực tiếp tọa độ (X, Y, Z) của đầu bút (pen tip) ra stdout.
- Có tham số cấu hình giãn khoảng cách lấy mẫu (sampling interval) để giảm tải CPU.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
import os

# ==========================================
# CẤU HÌNH HÌNH HỌC 3D (Đồng bộ từ pen_geometry.py)
# ==========================================
PEN_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],         # Điểm 0: Ngòi bút (Gốc tọa độ vật thể)
    [0.0, 64.0, 0.0],        # Điểm 1: Đuôi bút
    [-11.5, 44.0, 0.0],      # Điểm 2: Mép trái nắp
    [11.5, 44.0, 0.0]        # Điểm 3: Mép phải nắp
], dtype=np.float32)

# ==========================================
# CẤU HÌNH CAMERA (Mặc định cho Webcam 640x480)
# ==========================================
fx, fy = 770.0, 770.0
cx, cy = 320.0, 240.0
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)


def get_model():
    """Tự động tìm kiếm định dạng mô hình tối ưu nhất để tải."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Thứ tự ưu tiên: NCNN -> TFLite -> PyTorch (.pt)
    candidates = [
        # 1. NCNN (Thường xuất ra thư mục best_ncnn_model)
        (os.path.join(script_dir, 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best_ncnn_model'), 'NCNN'),
        # 2. TFLite (Thường xuất ra file best_saved_model hoặc best_float32.tflite/best_int8.tflite)
        (os.path.join(script_dir, 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best_saved_model'), 'TFLite'),
        (os.path.join(script_dir, 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best_int8.tflite'), 'TFLite (INT8)'),
        # 3. PyTorch gốc
        (os.path.join(script_dir, 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best.pt'), 'PyTorch (Raw)')
    ]
    
    for path, model_type in candidates:
        if os.path.exists(path):
            print(f"Detected and loading {model_type} model from: {path}")
            return YOLO(path)
            
    # Dự phòng tìm ở thư mục hiện tại
    fallback_pt = 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best.pt'
    if os.path.exists(fallback_pt):
        print(f"Loading fallback PyTorch model: {fallback_pt}")
        return YOLO(fallback_pt)
        
    raise FileNotFoundError("Không tìm thấy mô hình YOLOv8-pose nào để chạy!")


def main():
    parser = argparse.ArgumentParser(description="Headless Pen Tip Tracker for Pi 4")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index")
    parser.add_argument("--interval", type=float, default=3.0, 
                        help="Thời gian giãn lấy mẫu AI (giây). Mặc định: 3.0s")
    parser.add_argument("--conf", type=float, default=0.55, help="YOLO confidence threshold")
    args = parser.parse_args()

    print(f"⚙️ Khởi động hệ thống tracking đầu bút...")
    print(f"⏱️ Khoảng thời gian lấy mẫu: {args.interval} giây")
    
    model = get_model()

    print(f"📸 Mở Camera index {args.cam}...")
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"❌ Không thể mở Camera {args.cam}!")
        return

    # Thiết lập camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Cấu hình phơi sáng tự động và các thông số chống tối ảnh
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)     # 3 = Aperture Priority (Auto)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Thử lại với giá trị float cho một số backend OpenCV khác
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)           # Bật Auto White Balance (Cân bằng trắng tự động)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)      # Khôi phục độ sáng về mức mặc định (tránh bị chỉnh tay tối trước đó)
    cap.set(cv2.CAP_PROP_GAIN, -1)             # Thiết lập gain tự động/mặc định
    
    print("🚀 Bắt đầu vòng lặp đọc và xử lý. Ấn Ctrl+C để thoát.")
    
    last_inference_time = 0.0
    lost_logged = False      # Khống chế log LOST chỉ in 1 lần

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            now = time.time()
            # Kiểm tra xem đã đến chu kỳ lấy mẫu tiếp theo chưa
            if now - last_inference_time >= args.interval:
                last_inference_time = now
                
                # Chạy inference
                # Note: imgsz=320 để giảm tải xử lý ảnh
                results = model(frame, conf=args.conf, imgsz=320, verbose=False)
                
                detected = False
                
                if len(results[0]) > 0 and results[0].keypoints is not None:
                    kpts = results[0].keypoints.xy[0].cpu().numpy()
                    confs = results[0].keypoints.conf[0].cpu().numpy()
                    
                    # Cần đủ 4 keypoints có độ tin cậy tốt
                    valid = (len(kpts) == 4 and np.all(confs > 0.45) and not np.any(kpts == 0.0))
                    
                    if valid:
                        # Dùng solvePnP định vị
                        success, rvec, tvec = cv2.solvePnP(
                            PEN_3D_POINTS, kpts.astype(np.float32),
                            camera_matrix, dist_coeffs,
                            flags=cv2.SOLVEPNP_IPPE
                        )
                        
                        if success and tvec[2][0] > 0:
                            # Vì point 0 của PEN_3D_POINTS là ngòi bút [0.0, 0.0, 0.0],
                            # tvec chính là tọa độ 3D của ngòi bút (pen tip) trong hệ tọa độ camera.
                            x = tvec[0][0]
                            y = tvec[1][0]
                            z = tvec[2][0]
                            
                            print(f"[PEN_TIP] Time: {time.strftime('%H:%M:%S')} | X: {x:6.1f} mm | Y: {y:6.1f} mm | Z: {z:6.1f} mm")
                            detected = True
                            lost_logged = False  # Reset cờ khi phát hiện lại
                
                if not detected:
                    if not lost_logged:
                        print(f"[PEN_TIP] Time: {time.strftime('%H:%M:%S')} | LOST_TARGET")
                        lost_logged = True
            
            # Giữ cho vòng lặp không chiếm dụng 100% CPU khi không inference
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑 Đang dừng chương trình...")
    finally:
        cap.release()
        print("👋 Đã giải phóng Camera. Tạm biệt!")


if __name__ == '__main__':
    main()
