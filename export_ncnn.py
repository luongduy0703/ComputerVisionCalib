from ultralytics import YOLO
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    best_model_path = os.path.join(script_dir, 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best.pt')
    if not os.path.exists(best_model_path):
        best_model_path = 'runs/pose/Aero_Models/pen_pose_v4-3/weights/best.pt'
        
    print(f"📦 Loading model: {best_model_path}")
    model = YOLO(best_model_path)
    
    print("⏳ Exporting model to NCNN format...")
    model.export(format='ncnn', imgsz=320, simplify=True)
    print("🎉 Export complete! The NCNN model should be in the same folder as best.pt.")

if __name__ == '__main__':
    main()
