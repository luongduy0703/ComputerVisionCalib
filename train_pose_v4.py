from ultralytics import YOLO
import torch

def main():
    print("🚀 BẮT ĐẦU HUẤN LUYỆN TRÊN BỘ DỮ LIỆU V4 (COVIP_training.v4i)...")
    
    # Tự động chọn thiết bị huấn luyện (GPU nếu có, ngược lại dùng CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Thiết bị sử dụng: {device.upper()}")
    if device == 'cpu':
        print("⚠️ Cảnh báo: Huấn luyện trên CPU với 2445 ảnh có thể mất rất nhiều thời gian!")
        print("💡 Gợi ý: Bạn có thể giảm số lượng epochs (ví dụ: epochs=50 hoặc 100) để chạy thử trước.")

    # Tải mô hình YOLOv8n-pose pre-trained
    model = YOLO('yolov8n-pose.pt') 
    
    # Đường dẫn file data.yaml của bộ dữ liệu V4
    data_yaml_path = '/home/luongduy/AeroScript_Vision/datasets/pen_pose/COVIP_training.v4i.yolov8/data.yaml'
    
    # Tiến trình huấn luyện
    # Bạn có thể điều chỉnh epochs và batch tùy theo cấu hình máy
    model.train(
        data=data_yaml_path,
        epochs=150,               # Mặc định 100 epochs (giảm từ 150 để tối ưu thời gian)
        patience=30,
        imgsz=320,                # Độ phân giải ảnh đầu vào
        batch=16,                 # Tăng lên 16 nếu dùng GPU mạnh, giữ 8 hoặc 16 cho CPU
        device=device,            # Thiết bị train
        project='Aero_Models',    # Thư mục lưu kết quả huấn luyện
        name='pen_pose_v4',       # Tên phiên bản huấn luyện
        verbose=True
    )
    
    print("\n✅ Huấn luyện thành công! File trọng số tốt nhất được lưu tại: Aero_Models/pen_pose_v4/weights/best.pt")

    print("\n⏳ ĐANG TỰ ĐỘNG XUẤT MÔ HÌNH SANG ĐỊNH DẠNG TFLITE CHO RASPBERRY PI 4...")
    best_model_path = 'Aero_Models/pen_pose_v4/weights/best.pt'
    best_model = YOLO(best_model_path)
    
    best_model.export(
        format='tflite',
        int8=True,       # Lượng tử hóa int8 giúp tăng tốc tối đa trên Raspberry Pi
        imgsz=320,
        simplify=True    
    )
    print("🎉 HOÀN TẤT! Đã xuất file TFLite thành công.")

if __name__ == '__main__':
    main()
