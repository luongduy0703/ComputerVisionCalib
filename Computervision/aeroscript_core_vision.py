"""
AeroScript Core Vision — Module Nhận thức Cốt lõi (v3 - ArUco Multi-Marker)
=============================================================================
Kết hợp 3 thành phần:
  1. YOLO Tracking      : Khóa mục tiêu bút sơn (painting_pen)
  2. ArUco Multi-Marker  : Nhận diện 4 marker ở 4 góc khung làm việc
  3. Pose Estimation     : Ước lượng tư thế tường & tính mặt phẳng nén lò xo

Tác giả : AeroScript Team
Ngày    : 2026-05-05
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════
#  CẤU HÌNH
# ═══════════════════════════════════════════════════════════

# --- Đường dẫn ---
MODEL_PATH = "/home/luongduy/AeroScript_Vision/runs/detect/runs/detect/aeroscript_pen_model/weights/best.pt"

# --- YOLO ---
YOLO_CONFIDENCE = 0.6

# --- ArUco Marker ---
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_250   # Dictionary 4x4, phù hợp marker đã in
MARKER_LENGTH_CM = 2.0                      # ← ĐIỀN KÍCH THƯỚC THỰC CỦA MARKER (cm)
                                            # Ví dụ: 5.0 nếu mỗi marker 5x5 cm

# --- Camera ---
CAMERA_INDEX   = 0
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
WARMUP_FRAMES  = 30

# --- Cơ khí lò xo ---
DELTA_D_CM     = 1.0            # Khoảng nén lò xo (cm) — tường ảo đâm xuyên

# --- Hiển thị ---
COLOR_PEN_BOX  = (0, 255, 0)    # Xanh lá — bounding box bút
COLOR_PEN_CTR  = (0, 0, 255)    # Đỏ     — tâm bút
COLOR_TEXT     = (0, 255, 255)   # Vàng   — text thông số
COLOR_LIVE     = (0, 255, 0)    # Xanh lá — dữ liệu LIVE
COLOR_LOST     = (0, 0, 255)    # Đỏ     — cảnh báo mất dấu
FONT           = cv2.FONT_HERSHEY_SIMPLEX

WINDOW_MAIN    = "AeroScript - Core Vision"


# ═══════════════════════════════════════════════════════════
#  MA TRẬN NỘI TẠI CAMERA (giả định, chưa calib)
# ═══════════════════════════════════════════════════════════

def build_camera_matrix(w=640, h=480):
    """
    Tạo ma trận nội tại K giả định cho camera chưa calib.
    Focal length ước lượng ≈ chiều rộng ảnh (heuristic phổ biến).
    """
    fx = fy = float(w)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)
    return K


# ═══════════════════════════════════════════════════════════
#  BƯỚC A: TRACKING BÚT (YOLO) — HOÀN TOÀN ĐỘC LẬP
# ═══════════════════════════════════════════════════════════

def track_pen(model, frame, display):
    """
    Chạy YOLO để tìm bounding box bút sơn.

    Returns:
        pen_center : (cx, cy) tuple hoặc None nếu không tìm thấy
    """
    results = model(frame, conf=YOLO_CONFIDENCE, verbose=False)
    pen_center = None

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names.get(cls_id, "?")

            # Vẽ bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), COLOR_PEN_BOX, 2)
            cv2.putText(display, f"{label} {conf:.2f}",
                        (x1, y1 - 8), FONT, 0.5, COLOR_PEN_BOX, 2)

            # Tính tâm bounding box (xấp xỉ vị trí mũi bút)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(display, (cx, cy), 6, COLOR_PEN_CTR, -1)
            pen_center = (cx, cy)
            break  # Lấy detection confidence cao nhất
        if pen_center:
            break

    return pen_center


# ═══════════════════════════════════════════════════════════
#  BƯỚC B: NHẬN DIỆN ĐA MARKER (ArUco Multi-Marker)
#  API OpenCV >= 4.8: dùng ArucoDetector thay detectMarkers()
# ═══════════════════════════════════════════════════════════

def detect_aruco_markers(frame_gray, aruco_detector):
    """
    Tìm tất cả ArUco marker trong khung hình.
    Sử dụng ArucoDetector (API mới OpenCV 4.8+).

    Returns:
        corners    : Tuple các mảng góc (4 điểm) cho mỗi marker
        ids        : Mảng ID marker tìm được (Nx1)
        n_detected : Số marker tìm được (0–4)
    """
    corners, ids, _ = aruco_detector.detectMarkers(frame_gray)

    n_detected = 0 if ids is None else len(ids)
    return corners, ids, n_detected


# ═══════════════════════════════════════════════════════════
#  BƯỚC C + D: ƯỚC LƯỢNG TƯ THẾ ĐA MARKER → PHÁP TUYẾN
#              → MẶT PHẲNG NÉN LÒ XO
# ═══════════════════════════════════════════════════════════

def compute_multi_marker_pose(corners, ids, K, dist_coeffs, marker_length,
                              display=None):
    """
    Ước lượng tư thế 3D từ TẤT CẢ marker nhìn thấy.
    Thay thế estimatePoseSingleMarkers (bị loại bỏ ở OpenCV >= 4.8)
    bằng cách tự tạo object_points và gọi solvePnP cho từng marker.

    -----------------------------------------------------------------
    TOÁN HỌC TỔNG HỢP ĐA MARKER:

    1. Với mỗi marker, tạo 4 điểm 3D (object_points) dựa trên
       marker_length, rồi chạy solvePnP → rvec, tvec.

    2. Vector pháp tuyến n⃗:
       Vì tất cả marker nằm trên CÙNG MỘT MẶT PHẲNG (tờ giấy), ta chỉ
       cần lấy rvec của marker đầu tiên, đổi thành Ma trận xoay R (3×3)
       bằng Rodrigues, rồi trích xuất CỘT THỨ 3:

           R = Rodrigues(rvec_0)
           n⃗ = R[:, 2]

       Giải thích: Cột thứ 3 của R chính là hướng trục Z của marker
       trong hệ tọa độ camera. Vì marker nằm phẳng trên tường, trục Z
       này vuông góc với mặt phẳng tường → đó chính là PHÁP TUYẾN.

    3. Khoảng cách Z_wall:
       Lấy TRUNG BÌNH CỘNG trục Z của tất cả marker đang nhìn thấy.
       Trung bình nhiều marker giúp giảm nhiễu.

    4. Nén lò xo:
           Z_target = Z_wall + Δd
    -----------------------------------------------------------------

    Returns:
        normal_vec : Vector pháp tuyến (3,) hoặc None
        z_wall     : Khoảng cách trung bình Z tới tường (cm)
        z_target   : Z mặt phẳng nén lò xo (cm)
    """
    # ---------------------------------------------------------------
    # Tạo object_points cho 1 marker: 4 góc trên mặt phẳng Z=0
    #   Thứ tự: top-left, top-right, bottom-right, bottom-left
    # ---------------------------------------------------------------
    half = marker_length / 2.0
    obj_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float64)

    rvecs = []
    tvecs = []

    for i in range(len(ids)):
        img_pts = corners[i][0].astype(np.float64)  # (4, 2)
        try:
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_pts, K, dist_coeffs
            )
        except cv2.error:
            continue
        if not success:
            continue
        rvecs.append(rvec)
        tvecs.append(tvec)

    if len(rvecs) == 0:
        return None, 0.0, 0.0

    # ---------------------------------------------------------------
    # VẼ TRỤC 3D cho TẤT CẢ marker (trực quan hóa sự đồng bộ)
    # Mỗi marker hiển thị 3 trục RGB: X(đỏ) Y(xanh lá) Z(xanh dương)
    # ---------------------------------------------------------------
    if display is not None:
        for i in range(len(rvecs)):
            cv2.drawFrameAxes(display, K, dist_coeffs,
                              rvecs[i], tvecs[i], marker_length * 0.5)
            # Ghi ID marker lên ảnh
            c = corners[i][0]
            cx_m = int(c[:, 0].mean())
            cy_m = int(c[:, 1].mean())
            marker_id = int(ids[i][0])
            cv2.putText(display, f"ID:{marker_id}",
                        (cx_m - 15, cy_m - 15), FONT, 0.5, (255, 0, 255), 2)

    # ---------------------------------------------------------------
    # TRÍCH XUẤT PHÁP TUYẾN từ marker đầu tiên (rvecs[0])
    #
    #   R = Rodrigues(rvecs[0])
    #   R có 3 cột: [r1 | r2 | r3]
    #     • r1 (cột 1) = trục X marker (hướng ngang)
    #     • r2 (cột 2) = trục Y marker (hướng dọc)
    #     • r3 (cột 3) = trục Z marker = PHÁP TUYẾN mặt phẳng
    #
    #   Tất cả marker cùng nằm trên 1 mặt phẳng → pháp tuyến giống nhau
    #   → Chỉ cần lấy 1 marker là đủ.
    # ---------------------------------------------------------------
    R, _ = cv2.Rodrigues(rvecs[0])
    normal_vec = R[:, 2]  # CỘT THỨ 3 = Pháp tuyến

    # ---------------------------------------------------------------
    # TRUNG BÌNH Z_WALL từ tất cả marker
    #   Lấy tvec[2] (thành phần Z) của mỗi marker rồi tính mean.
    #   mean() giúp giảm nhiễu khi có nhiều marker.
    # ---------------------------------------------------------------
    z_values = [float(tv[2][0]) for tv in tvecs]
    z_wall = float(np.mean(z_values))

    # Nén lò xo: Z_target = Z_wall + Δd
    z_target = z_wall + DELTA_D_CM

    return normal_vec, z_wall, z_target


# ═══════════════════════════════════════════════════════════
#  HIỂN THỊ THÔNG SỐ LÊN FRAME (HUD) — Với Fail-safe
# ═══════════════════════════════════════════════════════════

def draw_hud(display, pen_center, n_markers, normal_vec, z_wall, z_target, K,
             pose_frozen=False, frame_count=0):
    """
    Vẽ overlay thông số debug lên frame chính.

    n_markers  : Số ArUco marker tìm thấy (0–4)
    pose_frozen: True khi 0 marker → dùng last_good
    frame_count: tạo hiệu ứng nhấp nháy cảnh báo
    """
    y = 25

    # --- Trạng thái bút (LUÔN hiển thị, ĐỘC LẬP với ArUco) ---
    if pen_center:
        cx, cy = pen_center
        # Tính toạ độ vật lý X, Y (cm) nếu có z_wall hợp lệ
        if z_wall > 0:
            fx, fy = K[0, 0], K[1, 1]
            cx0, cy0 = K[0, 2], K[1, 2]
            x_cm = (cx - cx0) * z_wall / fx
            y_cm = (cy - cy0) * z_wall / fy
            txt = f"Pen: ({x_cm:.1f}cm, {y_cm:.1f}cm) [Z={z_wall:.1f}cm]"
        else:
            txt = f"Pen: ({cx}px, {cy}px) [No Depth]"
        cv2.putText(display, txt, (10, y), FONT, 0.55, COLOR_TEXT, 2)
    else:
        cv2.putText(display, "Pen: TRACKING LOST", (10, y),
                    FONT, 0.55, COLOR_LOST, 2)
    y += 28

    # --- Trạng thái ArUco ---
    marker_color = COLOR_LIVE if n_markers > 0 else COLOR_LOST
    cv2.putText(display, f"ArUco markers: {n_markers}/4", (10, y),
                FONT, 0.55, marker_color, 2)
    y += 28

    # --- Pháp tuyến & Z_target ---
    if normal_vec is not None:
        nx, ny, nz = normal_vec
        data_color = COLOR_LIVE if not pose_frozen else COLOR_TEXT
        status_tag = "[LIVE]" if not pose_frozen else "[FROZEN]"

        cv2.putText(display,
                    f"Normal: ({nx:.3f}, {ny:.3f}, {nz:.3f}) {status_tag}",
                    (10, y), FONT, 0.55, data_color, 2)
        y += 28
        cv2.putText(display,
                    f"Z_wall: {z_wall:.1f}cm  Z_target: {z_target:.1f}cm",
                    (10, y), FONT, 0.55, data_color, 2)
        y += 28
        cv2.putText(display, f"Delta_d: {DELTA_D_CM} cm (nen lo xo)",
                    (10, y), FONT, 0.50, (180, 180, 180), 1)
    else:
        cv2.putText(display, "Wall pose: NO DATA YET",
                    (10, y), FONT, 0.55, COLOR_LOST, 2)

    # ---------------------------------------------------------------
    # CẢNH BÁO NHẤP NHÁY ĐỎ — khi 0 marker → Pose đóng băng
    # Nhấp nháy mỗi 15 frame (~0.5s ở 30fps)
    # ---------------------------------------------------------------
    if pose_frozen and (frame_count // 15) % 2 == 0:
        h_disp, w_disp = display.shape[:2]
        overlay = display.copy()
        cv2.rectangle(overlay, (w_disp - 420, 5), (w_disp - 5, 65),
                      (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        cv2.putText(display, "WARNING: No markers visible!",
                    (w_disp - 415, 30), FONT, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "Pose FROZEN - Robot HOLD",
                    (w_disp - 415, 55), FONT, 0.6, (0, 255, 255), 2)


# ═══════════════════════════════════════════════════════════
#  VÒNG LẶP CHÍNH
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  AeroScript Core Vision — v3 ArUco Multi-Marker")
    print("  (Fail-safe: Pose Freeze khi 0 marker)")
    print("=" * 60)

    # --- Kiểm tra MARKER_LENGTH_CM ---
    if MARKER_LENGTH_CM <= 0:
        print("\n  ⚠️  CẢNH BÁO: MARKER_LENGTH_CM chưa được cấu hình!")
        print("     Hãy mở file và điền kích thước marker thực (cm).")
        print("     Ví dụ: MARKER_LENGTH_CM = 5.0  # marker 5×5 cm")
        print("     Tạm dùng giá trị mặc định 5.0 cm.\n")
        marker_len = 5.0
    else:
        marker_len = MARKER_LENGTH_CM

    # --- 1. Load YOLO ---
    print("[INFO] Đang load mô hình YOLO...")
    model = YOLO(MODEL_PATH)
    print(f"[INFO] YOLO đã sẵn sàng. Classes: {model.names}")

    # --- 2. Khởi tạo ArUco (API mới: ArucoDetector) ---
    print("[INFO] Đang khởi tạo ArUco detector...")
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    print(f"[INFO] ArUco dictionary: DICT_4X4_250")
    print(f"[INFO] Marker size: {marker_len} cm")

    # --- 3. Camera matrix (giả định) ---
    K = build_camera_matrix(FRAME_WIDTH, FRAME_HEIGHT)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    print(f"[INFO] Camera matrix (giả định {FRAME_WIDTH}x{FRAME_HEIGHT}):")
    print(K)

    # --- 4. Mở camera ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[LỖI] Không mở được camera!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print(f"[INFO] Warm-up camera ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        cap.read()

    print("[INFO] Hệ thống sẵn sàng. Nhấn 'q' để thoát.\n")

    # ═══════════════════════════════════════════════
    #  BIẾN NHỚ TRẠNG THÁI — Fail-safe
    #  Lưu trạng thái tốt nhất gần nhất để dùng khi
    #  bị che khuất hoàn toàn (0 marker).
    # ═══════════════════════════════════════════════
    last_good_normal   = np.array([0.0, 0.0, 1.0])  # Mặc định: tường vuông góc camera
    last_good_z_wall   = 0.0
    last_good_z_target = 0.0
    pose_frozen = False     # True = đang dùng last_good
    frame_count = 0         # Đếm frame cho hiệu ứng nhấp nháy

    # ═══════════════════════════════════════════════
    #  PIPELINE LOOP
    # ═══════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CẢNH BÁO] Mất frame camera!")
            continue

        frame_count += 1
        display = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ─────────────────────────────────────────
        # Bước A: Tracking bút (YOLO)
        # → HOÀN TOÀN ĐỘC LẬP với ArUco/PnP.
        #   Dù marker bị che, bút vẫn được track.
        # ─────────────────────────────────────────
        pen_center = track_pen(model, frame, display)

        # ─────────────────────────────────────────
        # Bước B: Nhận diện ArUco Markers
        # ─────────────────────────────────────────
        corners, ids, n_markers = detect_aruco_markers(
            frame_gray, aruco_detector
        )

        # Vẽ viền marker lên display (nếu tìm thấy)
        if n_markers > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

        # Biến hiển thị
        display_normal   = None
        display_z_wall   = 0.0
        display_z_target = 0.0

        # ─────────────────────────────────────────
        # Bước C + D: KHÓA AN TOÀN (Fail-safe)
        #
        # n_markers > 0 → Ước lượng tư thế, cập nhật
        # n_markers == 0 → ĐÓNG BĂNG, dùng last_good
        # ─────────────────────────────────────────
        if n_markers > 0:
            # === CÓ MARKER: Ước lượng tư thế ===
            normal_vec, z_wall, z_target = compute_multi_marker_pose(
                corners, ids, K, dist_coeffs, marker_len, display
            )

            if normal_vec is not None:
                # Thành công → Cập nhật last_good
                last_good_normal = normal_vec.copy()
                last_good_z_wall = z_wall
                last_good_z_target = z_target
                pose_frozen = False

                display_normal   = normal_vec
                display_z_wall   = z_wall
                display_z_target = z_target
            else:
                # estimatePose fail → Đóng băng
                pose_frozen = True
                display_normal   = last_good_normal
                display_z_wall   = last_good_z_wall
                display_z_target = last_good_z_target
        else:
            # === 0 MARKER: Bị che khuất → Đóng băng ===
            pose_frozen = True
            display_normal   = last_good_normal
            display_z_wall   = last_good_z_wall
            display_z_target = last_good_z_target

        # ─────────────────────────────────────────
        # Hiển thị HUD
        # ─────────────────────────────────────────
        draw_hud(display, pen_center, n_markers,
                 display_normal, display_z_wall, display_z_target, K,
                 pose_frozen=pose_frozen, frame_count=frame_count)
        cv2.imshow(WINDOW_MAIN, display)

        # Thoát bằng phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Dọn dẹp ---
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã thoát AeroScript Core Vision.")


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
