import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
from collections import deque

# ==========================================
# 1. BỘ LỌC KALMAN 2D (Khử rung tọa độ keypoint trên ảnh)
# ==========================================
# Theo dõi 8 trạng thái: (x0, y0, x1, y1, x2, y2, x3, y3)
# cho 4 keypoints trên mặt phẳng ảnh.
# ==========================================
class KeypointKalmanFilter:
    """
    Kalman Filter cho 4 keypoints 2D.
    State: [x0, y0, x1, y1, x2, y2, x3, y3,
            vx0, vy0, vx1, vy1, vx2, vy2, vx3, vy3]
    Measurement: [x0, y0, x1, y1, x2, y2, x3, y3]
    """
    def __init__(self, process_noise=1e-2, measurement_noise=2.0):
        n_states = 16   # 8 positions + 8 velocities
        n_meas = 8      # 8 positions
        self.kf = cv2.KalmanFilter(n_states, n_meas, 0)
        self.initialized = False

        # Transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.eye(n_states, dtype=np.float32)
        for i in range(8):
            self.kf.transitionMatrix[i, i + 8] = 1.0  # x += vx * dt

        # Measurement matrix: chỉ đo được vị trí, không đo vận tốc
        self.kf.measurementMatrix = np.zeros((n_meas, n_states), dtype=np.float32)
        for i in range(8):
            self.kf.measurementMatrix[i, i] = 1.0

        # Ma trận nhiễu quá trình (Process noise) — mức độ tin tưởng vào mô hình chuyển động
        self.kf.processNoiseCov = np.eye(n_states, dtype=np.float32) * process_noise

        # Ma trận nhiễu đo lường (Measurement noise) — mức độ tin tưởng vào kết quả YOLO
        self.kf.measurementNoiseCov = np.eye(n_meas, dtype=np.float32) * measurement_noise

        # Error covariance ban đầu
        self.kf.errorCovPost = np.eye(n_states, dtype=np.float32) * 100.0

    def update(self, keypoints_4x2):
        """
        Nhận 4 keypoints (4, 2) từ YOLO, trả về keypoints đã được lọc mượt.
        """
        meas = keypoints_4x2.flatten().astype(np.float32)  # (8,)

        if not self.initialized:
            # Khởi tạo state ban đầu = measurement, velocity = 0
            state = np.zeros(16, dtype=np.float32)
            state[:8] = meas
            self.kf.statePost = state.reshape(16, 1)
            self.initialized = True
            return keypoints_4x2.copy()

        # Predict → Correct
        self.kf.predict()
        corrected = self.kf.correct(meas.reshape(8, 1))

        # Trích xuất 8 giá trị vị trí đã lọc
        filtered = corrected[:8, 0].reshape(4, 2)
        return filtered

    def reset(self):
        self.initialized = False

# ==========================================
# 2. BỘ LỌC KALMAN 3D (Khử rung tọa độ 3D từ solvePnP)
# ==========================================
# State: [x, y, z, vx, vy, vz]  (vị trí + vận tốc)
# Measurement: [x, y, z]
#
# Ưu điểm so với EMA:
#   - Dự đoán vị trí tiếp theo dựa trên vận tốc hiện tại
#   - Tự động cân bằng giữa dự đoán và đo lường
#   - Loại bỏ nhiễu tối ưu theo lý thuyết xác suất
# ==========================================
class PoseKalmanFilter:
    """
    Kalman Filter 6 trạng thái cho tọa độ 3D của ngòi bút.
    Khi phát hiện bước nhảy bất thường (outlier), bộ lọc sẽ tự
    khởi tạo lại tại vị trí mới thay vì cố ngoại suy.
    """
    def __init__(self, process_noise=0.5, measurement_noise=15.0, max_jump_mm=250.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_jump_mm = max_jump_mm
        self.initialized = False
        self.last_meas = None
        self._init_kf()

    def _init_kf(self):
        """Khởi tạo (hoặc reset) bộ lọc Kalman nội bộ."""
        n_states = 6   # x, y, z, vx, vy, vz
        n_meas = 3     # x, y, z
        self.kf = cv2.KalmanFilter(n_states, n_meas, 0)

        # Transition: constant velocity model
        self.kf.transitionMatrix = np.eye(n_states, dtype=np.float32)
        self.kf.transitionMatrix[0, 3] = 1.0  # x += vx
        self.kf.transitionMatrix[1, 4] = 1.0  # y += vy
        self.kf.transitionMatrix[2, 5] = 1.0  # z += vz

        # Measurement matrix: chỉ đo x, y, z
        self.kf.measurementMatrix = np.zeros((n_meas, n_states), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1.0
        self.kf.measurementMatrix[1, 1] = 1.0
        self.kf.measurementMatrix[2, 2] = 1.0

        # Process noise (Q)
        self.kf.processNoiseCov = np.eye(n_states, dtype=np.float32) * self.process_noise

        # Measurement noise (R)
        self.kf.measurementNoiseCov = np.eye(n_meas, dtype=np.float32) * self.measurement_noise

        # Error covariance ban đầu
        self.kf.errorCovPost = np.eye(n_states, dtype=np.float32) * 500.0

    def update(self, tvec):
        """
        Nhận tvec (3, 1) từ solvePnP, trả về tvec đã lọc mượt.
        Nếu bước nhảy quá lớn → reset bộ lọc tại vị trí mới.
        """
        meas = np.array([tvec[0][0], tvec[1][0], tvec[2][0]], dtype=np.float32)

        if not self.initialized:
            self._init_at(meas)
            return tvec.copy()

        # Phát hiện bước nhảy bất thường → reset tại vị trí mới
        jump = np.linalg.norm(meas - self.last_meas)
        if jump > self.max_jump_mm:
            self._init_at(meas)
            return tvec.copy()

        self.last_meas = meas.copy()

        # Predict → Correct
        self.kf.predict()
        corrected = self.kf.correct(meas.reshape(3, 1))

        result = corrected[:3, 0].reshape(3, 1)
        return result

    def _init_at(self, meas):
        """Khởi tạo bộ lọc tại một vị trí đo lường cụ thể."""
        self._init_kf()
        state = np.zeros(6, dtype=np.float32)
        state[:3] = meas
        self.kf.statePost = state.reshape(6, 1)
        self.last_meas = meas.copy()
        self.initialized = True

    def reset(self):
        self._init_kf()
        self.initialized = False
        self.last_meas = None

# ==========================================
# 3. DASHBOARD RENDERER (Vẽ biểu đồ bằng OpenCV)
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


# ==========================================
# 4. CẤU HÌNH HÌNH HỌC 3D (Đơn vị: mm)
# ==========================================
# (Đồng bộ từ pen_geometry.py)
PEN_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],         # Điểm 0: Ngòi bút
    [0.0, 64.0, 0.0],        # Điểm 1: Đuôi bút
    [-11.5, 44.0, 0.0],      # Điểm 2: Mép trái nắp
    [11.5, 44.0, 0.0]        # Điểm 3: Mép phải nắp
], dtype=np.float32)

# ==========================================
# 5. CẤU HÌNH CAMERA (Giả định cho Webcam Laptop 640x480)
# ==========================================
fx, fy = 600.0, 600.0
cx, cy = 320.0, 240.0
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="AeroScript Webcam Pose Estimation + Dashboard")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index (try 0, 1, or 2)")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="Khoảng cách giữa các lần chạy AI (giây). Mặc định: 0.0 (liên tục)")
    parser.add_argument("--raw", action="store_true",
                        help="Tắt bộ lọc Kalman, xuất kết quả thô từ solvePnP")
    args = parser.parse_args()

    print("⏳ Đang tải mô hình AI và Hệ thống PnP...")
    model_path = '/home/luongduy/AeroScript_Vision/runs/pose/Aero_Models/pen_pose_v4-3/weights/best.pt'
    model = YOLO(model_path)

    print(f"📸 Đang mở Camera index {args.cam}...")
    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("⚠️ V4L2 thất bại, thử backend mặc định...")
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"❌ Không thể mở Camera {args.cam}!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Khởi tạo bộ lọc Kalman và dashboard
    kf_2d = KeypointKalmanFilter(process_noise=1e-2, measurement_noise=2.0)
    kf_3d = PoseKalmanFilter(process_noise=0.5, measurement_noise=15.0, max_jump_mm=250.0)
    dashboard = Dashboard(cam_h=480)

    # Đếm số frame inference liên tiếp không phát hiện bút
    lost_count = 0
    LOST_RESET_THRESHOLD = 10  # Reset bộ lọc sau 10 lần inference liên tiếp mất dấu

    if args.raw:
        print("✅ Hệ thống + Dashboard sẵn sàng! [CHẾ ĐỘ RAW — không lọc Kalman]")
    else:
        print("✅ Hệ thống + Dashboard + Kalman Filter sẵn sàng!")
    if args.interval > 0:
        print(f"⏱️ Chế độ tiết kiệm CPU: AI chạy mỗi {args.interval}s")
    print("Ấn 'q' để thoát.")

    frame_count = 0
    last_inference_time = 0.0
    last_valid_pose = None
    current_xyz = (None, None, None)
    has_detection = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không đọc được frame.")
            break

        frame_count += 1
        if frame_count < 10 and np.mean(frame) < 5:
            print("⚠️ Camera đen. Thử --cam khác.")

        now = time.time()
        annotated_frame = frame.copy()
        run_inference = (now - last_inference_time >= args.interval)

        if run_inference:
            last_inference_time = now
            detected_this_frame = False

            results = model(frame, conf=0.55, imgsz=320, verbose=False)

            if len(results[0]) > 0 and results[0].keypoints is not None:
                kpts = results[0].keypoints.xy[0].cpu().numpy()
                confs = results[0].keypoints.conf[0].cpu().numpy()

                valid = (len(kpts) == 4 and
                         np.all(confs > 0.45) and
                         not np.any(kpts == 0.0))

                if valid:
                    if args.raw:
                        # === CHẾ ĐỘ RAW: Không lọc, dùng keypoints thô ===
                        use_kpts = np.ascontiguousarray(kpts, dtype=np.float32)
                    else:
                        # === CHẾ ĐỘ KALMAN: Lọc keypoints 2D ===
                        use_kpts = kf_2d.update(kpts)

                    success, rvec, tvec = cv2.solvePnP(
                        PEN_3D_POINTS, use_kpts,
                        camera_matrix, dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE
                    )

                    if success and tvec[2][0] > 0:
                        if args.raw:
                            # === RAW: Dùng tvec thô trực tiếp ===
                            use_tvec = tvec
                        else:
                            # === KALMAN: Lọc tvec 3D ===
                            use_tvec = kf_3d.update(tvec)

                        last_valid_pose = (use_kpts, use_tvec, rvec)

                        x_d = use_tvec[0][0]
                        y_d = use_tvec[1][0]
                        z_d = use_tvec[2][0]
                        current_xyz = (x_d, y_d, z_d)
                        has_detection = True
                        detected_this_frame = True
                        lost_count = 0

                        dashboard.push(x_d, y_d, z_d)

                        # Tính các chỉ số chẩn đoán
                        dist_3d = np.sqrt(x_d**2 + y_d**2 + z_d**2) # Khoảng cách đường thẳng (mm)
                        # Khoảng cách pixel giữa ngòi (0) và đuôi (1) từ YOLO
                        tip_tail_px = np.linalg.norm(kpts[0] - kpts[1]) 

                        tag = "RAW" if args.raw else "KF"
                        print(f"[{tag}] X: {x_d:6.1f} | Y: {y_d:6.1f} | Z: {z_d:6.1f} mm | Dist: {dist_3d:6.1f} mm | Tip-Tail: {tip_tail_px:5.1f} px")

            # Xử lý khi mất dấu: KHÔNG gọi predict, chỉ đóng băng rồi reset
            if not detected_this_frame:
                lost_count += 1
                if lost_count >= LOST_RESET_THRESHOLD:
                    # Mất dấu quá lâu → reset bộ lọc để sẵn sàng cho lần phát hiện tiếp
                    kf_2d.reset()
                    kf_3d.reset()
                    last_valid_pose = None
                    has_detection = False
                    current_xyz = (None, None, None)
                # Nếu mất < threshold: giữ nguyên last_valid_pose (đóng băng hiển thị)

        # --- Vẽ kết quả lên camera ---
        if last_valid_pose is not None:
            pts2d, tvec_s, rvec_s = last_valid_pose
            for x, y in pts2d:
                cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.drawFrameAxes(annotated_frame, camera_matrix, dist_coeffs,
                              rvec_s, tvec_s, 30)

        # --- Render dashboard ---
        panel = dashboard.render(current_xyz[0], current_xyz[1], current_xyz[2], has_detection)

        # --- Ghép camera + dashboard ---
        combined = np.hstack([annotated_frame, panel])

        cv2.imshow('AeroScript 3D Pose + Dashboard', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
