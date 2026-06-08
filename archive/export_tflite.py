from ultralytics import YOLO
import sys

def main():
    model_path = '/home/luongduy/AeroScript_Vision/runs/pose/Aero_Models/pen_pose_v1-9/weights/best.pt'
    print(f"📦 Đang tải mô hình từ: {model_path}")
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")
        sys.exit(1)
        
    print("⏳ ĐANG NÉN XUỐNG TFLITE CHO RASPBERRY PI 4...")
    try:
        model.export(
            format='tflite',
            int8=True,       
            imgsz=320,
            simplify=True    
        )
        print("🎉 HOÀN TẤT! Đã có mô hình cho Laptop và mạch nhúng TFLite.")
    except Exception as e:
        print(f"❌ Lỗi khi export TFLite: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
