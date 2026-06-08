import cv2
import numpy as np

def main():
    print("📸 Khởi động chẩn đoán camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("⚠️ Không mở được camera với V4L2, thử backend mặc định...")
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("❌ LỖI: Không thể mở /dev/video0!")
        return

    # Thiết lập cơ bản
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("⏳ Đang chụp thử 10 frames để khởi động camera...")
    frame = None
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Lỗi đọc ở frame thứ {i}")
            break
        print(f"Frame {i}: Độ sáng trung bình = {np.mean(frame):.2f}")

    if frame is not None:
        filename = "diagnose_camera.jpg"
        cv2.imwrite(filename, frame)
        print(f"💾 Đã lưu ảnh chụp thử vào file: {filename}")
        
        # Phân tích
        mean_val = np.mean(frame)
        if mean_val < 5.0:
            print("🚨 KẾT LUẬN: Ảnh chụp ra BỊ ĐEN HOÀN TOÀN.")
            print("👉 Vui lòng kiểm tra:")
            print("   1. Nắp che camera vật lý (shutter) trên viền màn hình laptop có đang đóng không?")
            print("   2. Có phím tắt camera (Fn + phím camera) nào đang tắt camera không?")
            print("   3. Có ứng dụng khác (trình duyệt, Zoom, Teams, OBS...) đang mở camera không?")
        else:
            print("🎉 KẾT LUẬN: Camera hoạt động bình thường, ảnh chụp ra có ánh sáng!")
            print("👉 Lỗi đen màn hình hiển thị có thể do hệ thống giao diện (GUI/Qt) của máy bạn.")
            
    cap.release()

if __name__ == '__main__':
    main()
