"""
AeroScript Vision Pipeline — Bước 1 + 2 (Tích hợp)
====================================================
Bước 1: YOLOv8 phát hiện vật thể → cắt ROI
Bước 2: Classical CV tìm đường viền + tâm trên ROI

Tác giả : AeroScript Team
Ngày    : 2026-05-04
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO
from step2_contour_detection import process_roi_and_find_contour

# ──────────────────────────────────────────────
# CẤU HÌNH
# ──────────────────────────────────────────────
MODEL_PATH     = "/home/luongduy/AeroScript_Vision/runs/detect/runs/detect/aeroscript_pen_model/weights/best.pt"  # Mô hình painting_pen đã train
CONFIDENCE     = 0.6                # Ngưỡng tin cậy tối thiểu
PADDING        = 20                 # Mở rộng lề (pixel) quanh bounding box
CAMERA_INDEX   = 0                  # Webcam mặc định
WARMUP_FRAMES  = 30                 # Số frame bỏ qua để camera khởi động

# Tên cửa sổ hiển thị (4 cửa sổ)
WINDOW_MAIN    = "AeroScript - Camera"
WINDOW_ROI     = "AeroScript - ROI"
WINDOW_MASK    = "AeroScript - Mask"
WINDOW_CONTOUR = "AeroScript - Contour"

# Màu sắc hiển thị (BGR)
BOX_COLOR    = (0, 255, 0)        # Xanh lá – khung bbox
LABEL_COLOR  = (0, 255, 255)      # Vàng – nhãn
BOX_THICK    = 2
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.6


def main() -> None:
    # ──────────────────────────────────────────
    # 1. KHỞI TẠO MÔ HÌNH & CAMERA
    # ──────────────────────────────────────────
    print("[INFO] Đang load mô hình YOLO …")
    model = YOLO(MODEL_PATH)

    # Ép mô hình chạy trên GPU (device=0 → CUDA GPU đầu tiên)
    # Nếu không có GPU, YOLO sẽ tự fallback về CPU kèm cảnh báo.
    model.to(device=0)
    print("[INFO] Mô hình đã sẵn sàng trên GPU.")

    # Dùng backend V4L2 (Linux) để tương thích tốt hơn với webcam
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback: thử mở không chỉ định backend
        print("[CẢNH BÁO] V4L2 thất bại, thử backend mặc định …")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[LỖI] Không mở được webcam. Kiểm tra kết nối camera.")
            sys.exit(1)

    # Đặt độ phân giải rõ ràng để tránh camera trả frame rỗng
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── WARM-UP: Bỏ qua vài frame đầu vì webcam cần thời gian khởi động ──
    # Nhiều webcam USB trả về frame đen trong ~0.5-1 giây đầu tiên.
    print(f"[INFO] Đang warm-up camera ({WARMUP_FRAMES} frames) …")
    for _ in range(WARMUP_FRAMES):
        cap.read()

    print("[INFO] Webcam đã sẵn sàng. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CẢNH BÁO] Không đọc được frame, bỏ qua …")
            continue

        # Lấy kích thước khung hình — dùng để chặn tràn viền sau này
        frame_h, frame_w = frame.shape[:2]

        # ──────────────────────────────────────
        # BƯỚC 1: SUY LUẬN YOLO (INFERENCE)
        # ──────────────────────────────────────
        # verbose=False để không in log mỗi frame → tăng hiệu suất
        results = model.predict(source=frame, conf=CONFIDENCE, verbose=False)

        # Bản sao để vẽ — giữ nguyên frame gốc cho việc cắt ROI
        display = frame.copy()

        roi_found = False  # Cờ đánh dấu đã tìm được ROI hợp lệ

        # ──────────────────────────────────────
        # BƯỚC 1: LỌC & CẮT ROI
        # ──────────────────────────────────────
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                conf = float(box.conf[0])

                # Lọc theo ngưỡng tin cậy (đã set ở predict,
                # nhưng kiểm tra lại cho chắc chắn)
                if conf < CONFIDENCE:
                    continue

                # Lấy tọa độ xyxy (pixel, float) → ép sang int
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Lấy nhãn lớp
                cls_id = int(box.cls[0])
                label  = model.names[cls_id]

                # ── Vẽ bbox lên frame hiển thị ──
                cv2.rectangle(display, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICK)
                cv2.putText(
                    display,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 8),
                    FONT, FONT_SCALE, LABEL_COLOR, 2,
                )

                # ── Thêm padding & CHỐNG TRÀN VIỀN ──
                # max(0, …)  : đảm bảo tọa độ không bị ÂM
                # min(w/h, …): đảm bảo tọa độ không VƯỢT QUÁ kích thước ảnh
                px1 = max(0, x1 - PADDING)
                py1 = max(0, y1 - PADDING)
                px2 = min(frame_w, x2 + PADDING)
                py2 = min(frame_h, y2 + PADDING)

                # ── Cắt ROI bằng Numpy slicing (hiệu suất cao) ──
                roi = frame[py1:py2, px1:px2]

                # ── CHỐNG MẢNG RỖNG ──
                # Khi UAV lắc mạnh, vật thể có thể trôi ra ngoài hoàn toàn
                # → roi.size == 0.  Phải kiểm tra trước khi imshow để tránh crash.
                if roi.size == 0:
                    continue

                # ════════════════════════════════════
                # Hiển thị ROI gốc (Bước 1)
                cv2.imshow(WINDOW_ROI, roi)

                # ════════════════════════════════════
                # BƯỚC 2: Tìm đường viền trên ROI
                # ════════════════════════════════════
                mask, contour_roi = process_roi_and_find_contour(roi)

                # Hiển thị Mask nhị phân (luôn có nếu ROI hợp lệ)
                if mask is not None:
                    cv2.imshow(WINDOW_MASK, mask)

                # Hiển thị ảnh ROI đã vẽ viền + tâm
                if contour_roi is not None:
                    cv2.imshow(WINDOW_CONTOUR, contour_roi)
                else:
                    # Không tìm được viền → hiện thông báo trên ROI
                    no_contour = roi.copy()
                    cv2.putText(
                        no_contour, "No contour",
                        (10, 30), FONT, 0.7, (0, 0, 255), 2,
                    )
                    cv2.imshow(WINDOW_CONTOUR, no_contour)

                roi_found = True
                break  # Chỉ xử lý 1 ROI (vật thể confidence cao nhất)

            if roi_found:
                break

        # Nếu không tìm được ROI hợp lệ nào → hiện ảnh đen nhỏ thay thế
        if not roi_found:
            placeholder = np.zeros((120, 240, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "No target",
                (40, 70),
                FONT, 0.7, (0, 0, 255), 2,
            )
            cv2.imshow(WINDOW_ROI, placeholder)
            cv2.imshow(WINDOW_CONTOUR, placeholder)
            # Mask là ảnh grayscale → placeholder riêng
            cv2.imshow(WINDOW_MASK, np.zeros((120, 240), dtype=np.uint8))

        # ──────────────────────────────────────
        # HIỂN THỊ CAMERA TOÀN CẢNH
        # ──────────────────────────────────────
        cv2.imshow(WINDOW_MAIN, display)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ──────────────────────────────────────────
    # GIẢI PHÓNG TÀI NGUYÊN
    # ──────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã thoát chương trình.")


if __name__ == "__main__":
    main()
