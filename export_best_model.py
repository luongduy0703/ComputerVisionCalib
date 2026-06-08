from ultralytics import YOLO

def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    best_model_path = os.path.join(script_dir, 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best.pt')
    if not os.path.exists(best_model_path):
        best_model_path = 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best.pt'
    print(f"📦 Đang tải mô hình đã train: {best_model_path}")
    
    model = YOLO(best_model_path)
    
    print("⏳ Đang tiến hành xuất mô hình sang định dạng TFLite...")
    data_yaml_path = os.path.join(script_dir, 'datasets/pen_pose/COVIP_training.v4i.yolov8/data.yaml')
    if not os.path.exists(data_yaml_path):
        data_yaml_path = 'datasets/pen_pose/COVIP_training.v4i.yolov8/data.yaml'
        
    model.export(
        format='tflite',
        int8=True,       # Lượng tử hóa int8 để tối ưu tốc độ trên Raspberry Pi 4
        imgsz=320,
        simplify=True,
        data=data_yaml_path # Calib INT8
    )
    print("🎉 HOÀN TẤT! Mô hình TFLite đã được sinh ra cùng thư mục với best.pt.")

if __name__ == '__main__':
    main()
