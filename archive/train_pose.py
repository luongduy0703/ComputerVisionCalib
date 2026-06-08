from ultralytics import YOLO

def main():
    print("🚀 BẮT ĐẦU HUẤN LUYỆN TRONG AEROSCRIPT_VISION...")
    # Tải mô hình Nano nhẹ nhất
    model = YOLO('yolov8n-pose.pt') 
    
    # Quá trình Train
    results = model.train(
        data='/home/luongduy/AeroScript_Vision/datasets/pen_pose/data.yaml', # Đường dẫn mới
        epochs=150,
        imgsz=320,
        batch=8,
        project='Aero_Models', # Sẽ tạo một thư mục tên Aero_Models trong dự án
        name='pen_pose_v1'
    )
    print("✅ Huấn luyện xong file best.pt!")

    print("⏳ ĐANG NÉN XUỐNG TFLITE CHO RASPBERRY PI 4...")
    best_model = YOLO('Aero_Models/pen_pose_v1/weights/best.pt')
    
    best_model.export(
        format='tflite',
        int8=True,       
        imgsz=320,
        simplify=True    
    )
    print("🎉 HOÀN TẤT! Đã có mô hình cho Laptop và mạch nhúng.")

if __name__ == '__main__':
    main()
