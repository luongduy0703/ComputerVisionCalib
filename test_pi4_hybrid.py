#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AeroScript - Hybrid Pen Tip Tracker (YOLOv8-Pose + Lucas-Kanade Optical Flow)
=============================================================================
- Chạy tối ưu cho Raspberry Pi 4:
  * YOLOv8-Pose chạy định kỳ (mặc định 3s/lần) làm mốc sửa lỗi trôi (drift correction).
  * Lucas-Kanade Optical Flow bám đuôi 4 keypoints ở tần số camera thực tế (20-30 FPS).
  * solvePnP chạy ở mọi frame để xuất tọa độ đầu bút thời gian thực với độ trễ thấp.
- Tiết kiệm 80-90% tải CPU so với chạy YOLO liên tục.
- Hỗ trợ chế độ headless (mặc định) và GUI (để debug trực quan).
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
import os

from collections import deque

# ==========================================
# CẤU HÌNH HÌNH HỌC 3D (Đồng bộ từ pen_geometry.py)
# ==========================================
PEN_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],         # Điểm 0: Ngòi bút
    [0.0, 64.0, 0.0],        # Điểm 1: Đuôi bút
    [-11.5, 44.0, 0.0],      # Điểm 2: Mép trái nắp
    [11.5, 44.0, 0.0]        # Điểm 3: Mép phải nắp
], dtype=np.float32)

# ==========================================
# CẤU HÌNH CAMERA (Mặc định cho Webcam 640x480)
# ==========================================
#fx, fy = 770.0, 770.0
fx, fy = 600.0, 600.0
cx, cy = 320.0, 240.0
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# Tham số cấu hình Lucas-Kanade Optical Flow
LK_PARAMS = dict(
    winSize=(21, 21),       # Kích thước cửa sổ tìm kiếm (lớn hơn giúp bắt chuyển động nhanh tốt hơn)
    maxLevel=3,             # Số tầng kim tự tháp ảnh (pyramid levels)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
)

# ==========================================
# DASHBOARD RENDERER (Vẽ biểu đồ bằng OpenCV)
# ==========================================
class Dashboard:
    """Vẽ dashboard bên phải khung hình camera với các biểu đồ real-time."""

    PANEL_W = 420          # Chiều rộng panel dashboard
    HISTORY_LEN = 120      # Số điểm dữ liệu lưu lại cho biểu đồ cuộn
    TRAIL_LEN = 500        # Số điểm quỹ đạo 2D lưu lại

    # Bảng màu
    COL_BG       = (25, 25, 30)       # Nền dashboard
    COL_GRID     = (50, 50, 55)       # Đường kẻ lưới
    COL_X        = (80, 180, 255)     # Cam nhạt cho X
    COL_Y        = (80, 255, 120)     # Xanh lá cho Y
    COL_Z        = (100, 100, 255)    # Đỏ nhạt cho Z
    COL_TRAIL    = (255, 200, 80)     # Vàng nhạt cho quỹ đạo
    COL_TEXT     = (220, 220, 220)    # Chữ trắng xám
    COL_LABEL    = (140, 140, 150)    # Chữ label nhạt
    COL_ACCENT   = (0, 200, 255)     # Viền highlight

    def __init__(self, cam_h):
        self.cam_h = cam_h
        self.x_hist = deque(maxlen=self.HISTORY_LEN)
        self.y_hist = deque(maxlen=self.HISTORY_LEN)
        self.z_hist = deque(maxlen=self.HISTORY_LEN)
        self.trail = deque(maxlen=self.TRAIL_LEN)
        self.sample_count = 0

    def push(self, x, y, z):
        """Thêm một điểm dữ liệu mới vào lịch sử."""
        self.x_hist.append(x)
        self.y_hist.append(y)
        self.z_hist.append(z)
        self.trail.append((x, y))
        self.sample_count += 1

    def render(self, current_x, current_y, current_z, has_detection):
        """Render toàn bộ panel dashboard và trả về ảnh numpy."""
        panel = np.full((self.cam_h, self.PANEL_W, 3), self.COL_BG, dtype=np.uint8)

        # --- Viền trái ---
        cv2.line(panel, (0, 0), (0, self.cam_h), self.COL_ACCENT, 2)

        # --- Tiêu đề ---
        cv2.putText(panel, "AeroScript Dashboard", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COL_ACCENT, 2)
        cv2.line(panel, (15, 38), (self.PANEL_W - 15, 38), self.COL_GRID, 1)

        # ========================================
        # PHẦN 1: Chỉ số kỹ thuật số (Digital Readout)
        # ========================================
        y0 = 65
        if has_detection:
            status_color = (0, 255, 100)
            status_text = "TRACKING"
        else:
            status_color = (0, 0, 200)
            status_text = "NO TARGET"

        cv2.putText(panel, status_text, (15, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        cv2.putText(panel, f"Samples: {self.sample_count}", (220, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COL_LABEL, 1)

        y0 += 30
        self._draw_readout(panel, "X (Ngang)", current_x, "mm", self.COL_X, 15, y0)
        y0 += 25
        self._draw_readout(panel, "Y (Doc)  ", current_y, "mm", self.COL_Y, 15, y0)
        y0 += 25
        self._draw_readout(panel, "Z (Sau)  ", current_z, "mm", self.COL_Z, 15, y0)

        cv2.line(panel, (15, y0 + 12), (self.PANEL_W - 15, y0 + 12), self.COL_GRID, 1)

        # ========================================
        # PHẦN 2: Biểu đồ sóng cuộn X, Y, Z theo thời gian
        # ========================================
        chart_top = y0 + 25
        cv2.putText(panel, "Time-Series (X, Y, Z)", (15, chart_top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COL_LABEL, 1)
        chart_top += 8

        chart_h = 130
        chart_left = 15
        chart_right = self.PANEL_W - 15
        chart_w = chart_right - chart_left

        # Vẽ nền biểu đồ
        cv2.rectangle(panel, (chart_left, chart_top),
                      (chart_right, chart_top + chart_h), (35, 35, 40), -1)
        # Đường kẻ giữa (zero-line)
        mid_y = chart_top + chart_h // 2
        cv2.line(panel, (chart_left, mid_y), (chart_right, mid_y), self.COL_GRID, 1)
        # Đường 1/4 và 3/4
        cv2.line(panel, (chart_left, chart_top + chart_h // 4),
                 (chart_right, chart_top + chart_h // 4), (40, 40, 42), 1)
        cv2.line(panel, (chart_left, chart_top + 3 * chart_h // 4),
                 (chart_right, chart_top + 3 * chart_h // 4), (40, 40, 42), 1)

        if len(self.x_hist) > 1:
            # Tìm range tự động
            all_vals = list(self.x_hist) + list(self.y_hist) + list(self.z_hist)
            v_min = min(all_vals)
            v_max = max(all_vals)
            v_range = max(v_max - v_min, 20.0)  # Tối thiểu 20mm range
            v_center = (v_max + v_min) / 2.0

            # Vẽ nhãn range
            cv2.putText(panel, f"{v_center + v_range / 2:.0f}",
                        (chart_right + 2, chart_top + 12),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, self.COL_LABEL, 1)
            cv2.putText(panel, f"{v_center - v_range / 2:.0f}",
                        (chart_right + 2, chart_top + chart_h),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, self.COL_LABEL, 1)

            self._draw_line_series(panel, self.x_hist, chart_left, chart_top,
                                   chart_w, chart_h, v_center, v_range, self.COL_X)
            self._draw_line_series(panel, self.y_hist, chart_left, chart_top,
                                   chart_w, chart_h, v_center, v_range, self.COL_Y)
            self._draw_line_series(panel, self.z_hist, chart_left, chart_top,
                                   chart_w, chart_h, v_center, v_range, self.COL_Z)

        # Chú thích
        legend_y = chart_top + chart_h + 15
        for i, (label, col) in enumerate([("X", self.COL_X), ("Y", self.COL_Y), ("Z", self.COL_Z)]):
            lx = 15 + i * 90
            cv2.line(panel, (lx, legend_y - 4), (lx + 20, legend_y - 4), col, 2)
            cv2.putText(panel, label, (lx + 25, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        cv2.line(panel, (15, legend_y + 12), (self.PANEL_W - 15, legend_y + 12), self.COL_GRID, 1)

        # ========================================
        # PHẦN 3: Quỹ đạo 2D (Bảng vẽ thu nhỏ X-Y)
        # ========================================
        canvas_top = legend_y + 25
        cv2.putText(panel, "2D Trajectory (X-Y)", (15, canvas_top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COL_LABEL, 1)
        canvas_top += 8

        canvas_size = min(self.PANEL_W - 30, self.cam_h - canvas_top - 10)
        canvas_size = max(canvas_size, 80)
        canvas_left = 15
        canvas_right = canvas_left + canvas_size
        canvas_bottom = canvas_top + canvas_size

        # Nền canvas
        cv2.rectangle(panel, (canvas_left, canvas_top),
                      (canvas_right, canvas_bottom), (35, 35, 40), -1)
        # Lưới chữ thập
        cmx = canvas_left + canvas_size // 2
        cmy = canvas_top + canvas_size // 2
        cv2.line(panel, (cmx, canvas_top), (cmx, canvas_bottom), self.COL_GRID, 1)
        cv2.line(panel, (canvas_left, cmy), (canvas_right, cmy), self.COL_GRID, 1)

        if len(self.trail) > 1:
            trail_arr = np.array(list(self.trail))
            tx, ty = trail_arr[:, 0], trail_arr[:, 1]
            t_xmin, t_xmax = tx.min(), tx.max()
            t_ymin, t_ymax = ty.min(), ty.max()
            t_range = max(t_xmax - t_xmin, t_ymax - t_ymin, 20.0)
            t_cx = (t_xmax + t_xmin) / 2.0
            t_cy = (t_ymax + t_ymin) / 2.0

            margin = 10
            draw_size = canvas_size - 2 * margin
            pts = []
            for px, py in self.trail:
                sx = int(cmx + (px - t_cx) / t_range * draw_size)
                sy = int(cmy + (py - t_cy) / t_range * draw_size)
                sx = np.clip(sx, canvas_left + 2, canvas_right - 2)
                sy = np.clip(sy, canvas_top + 2, canvas_bottom - 2)
                pts.append((sx, sy))

            # Vẽ đường quỹ đạo
            for i in range(1, len(pts)):
                alpha = int(80 + 175 * i / len(pts))
                col = (
                    int(self.COL_TRAIL[0] * alpha / 255),
                    int(self.COL_TRAIL[1] * alpha / 255),
                    int(self.COL_TRAIL[2] * alpha / 255),
                )
                cv2.line(panel, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)

            # Vẽ vị trí hiện tại (chấm to)
            cv2.circle(panel, pts[-1], 4, (0, 255, 255), -1)

        # Nhãn trục
        cv2.putText(panel, "X ->", (canvas_right - 35, canvas_bottom + 12),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, self.COL_LABEL, 1)
        cv2.putText(panel, "Y", (canvas_left - 2, canvas_top - 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, self.COL_LABEL, 1)

        return panel

    # --- Helper: Vẽ chỉ số dạng số ---
    def _draw_readout(self, img, label, value, unit, color, x, y):
        cv2.putText(img, f"{label}:", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COL_LABEL, 1)
        if value is not None:
            cv2.putText(img, f"{value:8.1f} {unit}", (x + 130, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        else:
            cv2.putText(img, "  --- ", (x + 130, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)

    # --- Helper: Vẽ đường line-series ---
    def _draw_line_series(self, img, data, left, top, w, h, center, vrange, color):
        n = len(data)
        if n < 2:
            return
        pts = []
        for i, v in enumerate(data):
            px = int(left + i * w / (self.HISTORY_LEN - 1))
            normalized = (v - center) / vrange  # -0.5 .. 0.5
            py = int(top + h / 2 - normalized * h)
            py = np.clip(py, top + 1, top + h - 1)
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)

def get_model():
    """Tự động tìm kiếm định dạng mô hình tối ưu nhất để tải."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        (os.path.join(script_dir, 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best_ncnn_model'), 'NCNN'),
        (os.path.join(script_dir, 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best_saved_model'), 'TFLite'),
        (os.path.join(script_dir, 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best_int8.tflite'), 'TFLite (INT8)'),
        (os.path.join(script_dir, 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best.pt'), 'PyTorch (Raw)')
    ]
    for path, model_type in candidates:
        if os.path.exists(path):
            print(f"Detected and loading {model_type} model from: {path}")
            return YOLO(path)
            
    fallback_pt = 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best.pt'
    if os.path.exists(fallback_pt):
        print(f"Loading fallback PyTorch model: {fallback_pt}")
        return YOLO(fallback_pt)
        
    raise FileNotFoundError("Không tìm thấy mô hình YOLOv8-pose nào để chạy!")


def main():
    parser = argparse.ArgumentParser(description="Hybrid YOLO + LK Optical Flow Pen Tip Tracker")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--interval", type=float, default=3.0, 
                        help="Khoảng thời gian kích hoạt YOLO để sửa sai số trôi (giây)")
    parser.add_argument("--conf", type=float, default=0.55, help="YOLO confidence threshold")
    parser.add_argument("--gui", action="store_true", help="Bật cửa sổ hiển thị trực quan (Debug)")
    parser.add_argument("--out-interval", type=float, default=0.1, 
                        help="Khoảng thời gian xuất/in tọa độ ra terminal (giây). Mặc định: 0.1s (10Hz)")
    args = parser.parse_args()

    print(f"⚙️ Khởi động hệ thống tracking lai (Hybrid)...")
    print(f"⏱️ Khoảng cách chạy YOLO sửa trôi: {args.interval} giây")
    print(f"🖥️ Chế độ: {'GUI Debug' if args.gui else 'Headless'}")

    model = get_model()

    print(f"📸 Mở Camera index {args.cam}...")
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"❌ Không thể mở Camera {args.cam}!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Cấu hình phơi sáng tự động và các thông số chống tối ảnh
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)     # 3 = Aperture Priority (Auto)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Thử lại với giá trị float cho một số backend OpenCV khác
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)           # Bật Auto White Balance (Cân bằng trắng tự động)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)      # Khôi phục độ sáng về mức mặc định (tránh bị chỉnh tay tối trước đó)
    cap.set(cv2.CAP_PROP_GAIN, -1)             # Thiết lập gain tự động/mặc định

    # Khởi tạo dashboard nếu bật GUI
    dashboard = Dashboard(cam_h=480) if args.gui else None

    # Các biến trạng thái bám đuôi
    prev_gray = None
    tracked_kpts = None      # 4 keypoints dạng numpy array shape (4, 1, 2)
    is_tracking = False
    last_yolo_time = 0.0
    lost_logged = False      # Khống chế log LOST chỉ in 1 lần
    last_print_time = 0.0    # Khống chế tốc độ in tọa độ ra terminal

    print("🚀 Bắt đầu xử lý. Ấn Ctrl+C (hoặc 'q' trong GUI) để thoát.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            now = time.time()
            
            # Cờ xác định có chạy YOLO ở frame này không
            run_yolo = False
            
            if not is_tracking or (now - last_yolo_time >= args.interval):
                run_yolo = True

            # ----------------------------------------------------
            # BƯỚC 1: CẬP NHẬT KEYPOINTS (YOLO HOẶC OPTICAL FLOW)
            # ----------------------------------------------------
            tracking_source = "NONE"
            
            if run_yolo:
                # Chạy YOLOv8-pose để phát hiện tuyệt đối
                t_start = time.time()
                results = model(frame, conf=args.conf, imgsz=320, verbose=False)
                t_yolo = (time.time() - t_start) * 1000.0
                
                detected = False
                if len(results[0]) > 0 and results[0].keypoints is not None:
                    kpts = results[0].keypoints.xy[0].cpu().numpy()
                    confs = results[0].keypoints.conf[0].cpu().numpy()
                    
                    valid = (len(kpts) == 4 and np.all(confs > 0.45) and not np.any(kpts == 0.0))
                    if valid:
                        # Gán lại điểm bám đuôi chuẩn (reshape về (4, 1, 2) cho hàm calcOpticalFlowPyrLK)
                        tracked_kpts = np.expand_dims(kpts, axis=1).astype(np.float32)
                        is_tracking = True
                        last_yolo_time = now
                        detected = True
                        tracking_source = f"YOLO (took {t_yolo:.1f}ms)"
                
                if not detected:
                    # Nếu YOLO xịt mà trước đó đang tracking thì giữ tạm trạng thái
                    if is_tracking:
                        run_yolo = False  # Fallback sang dùng Optical Flow dưới đây
                    else:
                        is_tracking = False
                        tracked_kpts = None
            
            # Nếu không chạy YOLO (hoặc YOLO xịt nhưng đang có điểm cũ), chạy Optical Flow
            if not run_yolo and is_tracking and prev_gray is not None and tracked_kpts is not None:
                t_start = time.time()
                # Tính toán Optical Flow từ frame trước sang frame hiện tại
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, tracked_kpts, None, **LK_PARAMS
                )
                t_of = (time.time() - t_start) * 1000.0
                
                # Kiểm tra xem cả 4 điểm có được bám thành công không (status == 1)
                if status is not None and np.all(status == 1):
                    tracked_kpts = next_pts
                    tracking_source = f"Optical Flow (took {t_of:.1f}ms)"
                else:
                    # Mất dấu một trong các điểm
                    is_tracking = False
                    tracked_kpts = None
                    tracking_source = "LOST (Optical Flow failed)"

            # Lưu ảnh xám cho frame tiếp theo
            prev_gray = gray.copy()

            # ----------------------------------------------------
            # BƯỚC 2: TÍNH TOÁN PNP & XUẤT TỌA ĐỘ
            # ----------------------------------------------------
            pen_tip_coords = None
            
            if is_tracking and tracked_kpts is not None:
                # Chuyển đổi shape về (4, 2) cho solvePnP
                pts_2d = tracked_kpts.reshape(4, 2)
                
                success, rvec, tvec = cv2.solvePnP(
                    PEN_3D_POINTS, pts_2d,
                    camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE
                )
                
                if success and tvec[2][0] > 0:
                    x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
                    pen_tip_coords = (x, y, z)
                    
                    # Log tọa độ ra stdout theo chu kỳ mong muốn
                    if now - last_print_time >= args.out_interval:
                        print(f"[{tracking_source}] Pen Tip -> X: {x:6.1f} | Y: {y:6.1f} | Z: {z:6.1f} mm")
                        last_print_time = now
                    
                    lost_logged = False  # Reset cờ
                    
                    # Đẩy dữ liệu vào dashboard
                    if args.gui:
                        dashboard.push(x, y, z)
                    
                    # Vẽ debug nếu bật GUI
                    if args.gui:
                        for idx, pt in enumerate(pts_2d):
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), -1)
                            cv2.putText(frame, str(idx), (int(pt[0])+8, int(pt[1])-8), 
                                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), 1)
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 30)
            
            if pen_tip_coords is None:
                if not lost_logged:
                    print(f"[{tracking_source}] Pen Tip -> LOST")
                    lost_logged = True
                if args.gui:
                    cv2.putText(frame, "LOST TARGET", (15, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ----------------------------------------------------
            # BƯỚC 3: HIỂN THỊ GUI DEBUG (NẾU ĐƯỢC YÊU CẦU)
            # ----------------------------------------------------
            if args.gui:
                has_det = (pen_tip_coords is not None)
                cur_x = pen_tip_coords[0] if has_det else None
                cur_y = pen_tip_coords[1] if has_det else None
                cur_z = pen_tip_coords[2] if has_det else None
                
                panel = dashboard.render(cur_x, cur_y, cur_z, has_det)
                combined = np.hstack([frame, panel])
                cv2.imshow("AeroScript Hybrid Tracking Debug", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Tiết kiệm CPU hơn nữa bằng sleep nhỏ ở chế độ headless
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Đang dừng chương trình...")
    finally:
        cap.release()
        if args.gui:
            cv2.destroyAllWindows()
        print("👋 Đã giải phóng Camera. Tạm biệt!")


if __name__ == '__main__':
    main()
