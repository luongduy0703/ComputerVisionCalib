"""
AeroScript Vision Pipeline — Bước 2: Phát hiện đường viền vật thể
=================================================================
Sử dụng Classical CV (lọc màu HSV, khử nhiễu hình thái học, findContours)
để trích xuất đường viền chính xác và tọa độ tâm của bề mặt cần sơn.

Đầu vào : roi_frame (mảng ảnh BGR đã cắt từ Bước 1)
Đầu ra  : mask (ảnh nhị phân sạch), output_roi (ảnh đã vẽ viền + tâm)

Tác giả : AeroScript Team
Ngày    : 2026-05-04
"""

import cv2
import numpy as np
from typing import Optional, Tuple

# ──────────────────────────────────────────────
# CẤU HÌNH PIPELINE
# ──────────────────────────────────────────────

# --- Dải màu HSV mục tiêu (ví dụ: màu XANH LÁ) ---
# Điều chỉnh giá trị này tùy theo màu bề mặt cần sơn thực tế.
# Công cụ hữu ích: chạy script hsv_tuner để chỉnh realtime.
HSV_LOWER = np.array([35, 80, 80])     # Cận dưới [H, S, V]
HSV_UPPER = np.array([85, 255, 255])   # Cận trên [H, S, V]

# --- Kernel cho phép hình thái học ---
MORPH_KERNEL_SIZE = (5, 5)

# --- Ngưỡng diện tích tối thiểu (pixel²) ---
# Viền nhỏ hơn giá trị này bị coi là rác/nhiễu và bị loại bỏ.
MIN_CONTOUR_AREA = 500

# --- Màu vẽ kết quả (BGR) ---
CONTOUR_COLOR = (0, 255, 0)       # Xanh lá – đường viền
CENTER_COLOR  = (0, 0, 255)       # Đỏ     – điểm tâm
CONTOUR_THICK = 2
CENTER_RADIUS = 6


def process_roi_and_find_contour(
    roi_frame: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Pipeline xử lý ảnh ROI để tìm đường viền chính xác của bề mặt cần sơn.

    Pipeline:
        BGR → HSV → Lọc màu → Opening → Closing → findContours
        → Viền lớn nhất → Tâm (cx, cy)

    Args:
        roi_frame: Mảng ảnh BGR (numpy array) đã cắt từ Bước 1.

    Returns:
        (mask, output_roi):
            - mask       : Ảnh nhị phân sau khi lọc nhiễu hình thái học.
            - output_roi : Ảnh ROI gốc đã vẽ đường viền và chấm tâm.
        Trả về (None, None) nếu không tìm được viền hợp lệ.
    """
    # ──────────────────────────────────────────
    # 1. ĐỔI HỆ MÀU: BGR → HSV
    # ──────────────────────────────────────────
    # HSV tách riêng kênh màu sắc (Hue) khỏi độ sáng (Value),
    # giúp lọc màu ổn định hơn khi ánh sáng thay đổi (UAV bay ngoài trời).
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # ──────────────────────────────────────────
    # 2. LỌC MÀU (COLOR SEGMENTATION)
    # ──────────────────────────────────────────
    # inRange tạo mặt nạ nhị phân: pixel nằm trong dải HSV → 255 (trắng),
    # ngoài dải → 0 (đen). Kết quả là ảnh binary thô, còn nhiều nhiễu.
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # ──────────────────────────────────────────
    # 3. KHỬ NHIỄU HÌNH THÁI HỌC (MORPHOLOGICAL OPS)
    # ──────────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)

    # ── 3a. OPENING (Erode → Dilate) ──
    # Tác dụng: Xóa các đốm nhiễu NHỎ (speckles) ở vùng nền xung quanh.
    # Erode thu nhỏ tất cả vùng trắng → đốm nhỏ biến mất hoàn toàn.
    # Dilate phục hồi lại kích thước vùng trắng lớn (vật thể chính).
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # ── 3b. CLOSING (Dilate → Erode) ──
    # Tác dụng: Vá các lỗ thủng BÊN TRONG vật thể.
    # Nguyên nhân lỗ thủng: bề mặt bị phản quang/cháy sáng (specular highlight)
    # khiến giá trị V trong HSV vượt ngưỡng → pixel bị "mất" khi inRange.
    # Dilate lấp đầy lỗ nhỏ → Erode co lại biên để giữ kích thước gốc.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ──────────────────────────────────────────
    # 4. TRÍCH XUẤT ĐƯỜNG VIỀN (FIND CONTOURS)
    # ──────────────────────────────────────────
    # RETR_EXTERNAL: chỉ lấy viền ngoài cùng (bỏ viền lồng bên trong).
    # CHAIN_APPROX_SIMPLE: nén các điểm thẳng hàng → tiết kiệm bộ nhớ.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Không tìm được viền nào → trả về mask (vẫn hữu ích cho debug)
    if not contours:
        return mask, None

    # Chọn viền có DIỆN TÍCH LỚN NHẤT → loại trừ rác/nhiễu còn sót
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    # Viền quá nhỏ → nhiễu, không phải vật thể thực
    if area < MIN_CONTOUR_AREA:
        return mask, None

    # ──────────────────────────────────────────
    # 5. TÍNH TỌA ĐỘ TÂM (IMAGE MOMENTS)
    # ──────────────────────────────────────────
    # Moments là tập hợp các đại lượng thống kê mô tả hình dạng viền.
    # m10/m00 = tọa độ x tâm,  m01/m00 = tọa độ y tâm.
    M = cv2.moments(largest_contour)

    # ── CHỐNG CHIA CHO 0 ──
    # M["m00"] = diện tích viền. Nếu = 0 (viền suy biến/1 pixel) → bỏ qua.
    if M["m00"] == 0:
        return mask, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # ──────────────────────────────────────────
    # 6. VẼ KẾT QUẢ LÊN ẢNH
    # ──────────────────────────────────────────
    # Tạo bản copy để không làm thay đổi roi_frame gốc
    output_roi = roi_frame.copy()

    # Vẽ đường viền lớn nhất
    cv2.drawContours(output_roi, [largest_contour], -1, CONTOUR_COLOR, CONTOUR_THICK)

    # Chấm điểm tâm
    cv2.circle(output_roi, (cx, cy), CENTER_RADIUS, CENTER_COLOR, -1)

    # Ghi tọa độ tâm lên ảnh (hữu ích khi debug)
    cv2.putText(
        output_roi,
        f"Center ({cx}, {cy})",
        (cx + 10, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        CENTER_COLOR,
        1,
    )

    return mask, output_roi


# ──────────────────────────────────────────────
# TEST ĐỘC LẬP (chạy trực tiếp file này để kiểm tra)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Cho phép test nhanh bằng ảnh tĩnh:
    #   python step2_contour_detection.py path/to/roi_image.jpg
    if len(sys.argv) < 2:
        print("Cách dùng: python step2_contour_detection.py <đường_dẫn_ảnh_ROI>")
        print("Hoặc import hàm process_roi_and_find_contour() vào Bước 1.")
        sys.exit(0)

    img_path = sys.argv[1]
    roi = cv2.imread(img_path)
    if roi is None:
        print(f"[LỖI] Không đọc được ảnh: {img_path}")
        sys.exit(1)

    mask, result = process_roi_and_find_contour(roi)

    cv2.imshow("Mask (Binary)", mask)

    if result is not None:
        cv2.imshow("Contour + Center", result)
        print("[OK] Đã tìm thấy viền và tâm vật thể.")
    else:
        print("[CẢNH BÁO] Không tìm được viền hợp lệ. Thử điều chỉnh HSV_LOWER/HSV_UPPER.")

    print("Nhấn phím bất kỳ để đóng …")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
