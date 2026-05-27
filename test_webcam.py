import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. BỘ LỌC EMA (Khử rung 2D)
# ==========================================
class EMAFilter:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.filtered_points = None

    def update(self, current_points):
        if self.filtered_points is None:
            self.filtered_points = current_points.copy()
        else:
            self.filtered_points = self.alpha * current_points + (1 - self.alpha) * self.filtered_points
        return self.filtered_points

# ==========================================
# 2. CẤU HÌNH HÌNH HỌC 3D (Đơn vị: mm)
# ==========================================
# Thứ tự BẮT BUỘC phải khớp với lúc bạn dán nhãn trên Roboflow: 
# (Ví dụ: 0: Ngòi, 1: Đuôi, 2: Mép trái, 3: Mép phải)
PEN_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],         # Điểm 0: Ngòi bút (Đây chính là gốc tọa độ)
    [0.0, 145.0, 0.0],       # Điểm 1: Đuôi bút (Giả sử dài 145mm)
    [-8.0, 100.0, 0.0],      # Điểm 2: Mép trái nắp
    [8.0, 100.0, 0.0]        # Điểm 3: Mép phải nắp
], dtype=np.float32)

# ==========================================
# 3. CẤU HÌNH CAMERA (Giả định cho Webcam Laptop 640x480)
# ==========================================
fx, fy = 600.0, 600.0
cx, cy = 320.0, 240.0
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((4, 1)) 

def main():
    print("⏳ Đang tải mô hình AI và Hệ thống PnP...")
    model_path = '/home/luongduy/AeroScript_Vision/runs/pose/Aero_Models/pen_pose_v1-7/weights/best.pt'
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    
    ema_filter = EMAFilter(alpha=0.4)
    
    print("✅ Hệ thống đã sẵn sàng! Hãy cầm thước kẻ ra đo nào.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, conf=0.6, verbose=False)
        annotated_frame = frame.copy() # Bỏ khung vẽ mặc định của YOLO cho dễ nhìn
        
        if len(results[0]) > 0 and results[0].keypoints is not None:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            
            # Nếu AI nhìn thấy đủ 4 điểm
            if len(kpts) == 4:
                # 1. Lọc mượt tọa độ 2D
                smoothed_kpts = ema_filter.update(kpts)
                
                # Vẽ 4 điểm màu xanh lá lên ảnh
                for x, y in smoothed_kpts:
                    cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                # 2. Giải mã 3D bằng SolvePnP
                success, rvec, tvec = cv2.solvePnP(
                    PEN_3D_POINTS, 
                    smoothed_kpts, 
                    camera_matrix, 
                    dist_coeffs, 
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Trích xuất tọa độ X, Y, Z (mm) của ngòi bút
                    x_dist = tvec[0][0]
                    y_dist = tvec[1][0]
                    z_dist = tvec[2][0]
                    
                    # Vẽ hệ trục tọa độ 3D mọc ra từ ngòi bút (Dài 30mm)
                    cv2.drawFrameAxes(annotated_frame, camera_matrix, dist_coeffs, rvec, tvec, 30)
                    
                    # Hiện thông số lên góc trái màn hình
                    cv2.putText(annotated_frame, f"X (Ngang): {x_dist:6.1f} mm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(annotated_frame, f"Y (Doc)  : {y_dist:6.1f} mm", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Z (Sau)  : {z_dist:6.1f} mm", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # In ra Terminal để tiện copy/lưu log
                    print(f"Ngòi bút -> X: {x_dist:6.1f} | Y: {y_dist:6.1f} | Z: {z_dist:6.1f} (mm)")

        else:
            ema_filter.filtered_points = None
            
        cv2.imshow('AeroScript 3D Pose', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
