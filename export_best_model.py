from ultralytics import YOLO

def main():
    # Sử dụng đúng đường dẫn tới trọng số best.pt đã train xong ở thư mục pen_pose_v4-3
    best_model_path = '/home/luongduy/AeroScript_Vision/runs/pose/Aero_Models/pen_pose_v4-3/weights/best.pt'
    print(f"📦 Đang tải mô hình đã train: {best_model_path}")
    
    model = YOLO(best_model_path)
    
    print("⏳ Đang tiến hành xuất mô hình sang định dạng TFLite...")
    model.export(
        format='tflite',
        int8=True,       # Lượng tử hóa int8 để tối ưu tốc độ trên Raspberry Pi 4
        imgsz=320,
        simplify=True,
        data='/home/luongduy/AeroScript_Vision/datasets/pen_pose/COVIP_training.v4i.yolov8/data.yaml' # Calib INT8
    )
    print("🎉 HOÀN TẤT! Mô hình TFLite đã được sinh ra cùng thư mục với best.pt.")

if __name__ == '__main__':
    main()
