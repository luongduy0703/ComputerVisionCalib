import cv2
import numpy as np
from ultralytics import YOLO
import glob

model_path = '/home/luongduy/AeroScript_Vision/runs/pose/Aero_Models/pen_pose_v1-9/weights/best.pt'
model = YOLO(model_path)

val_images = glob.glob('/home/luongduy/AeroScript_Vision/datasets/pen_pose/merged_pen_pose/val/images/*.jpg')
print(f"Found {len(val_images)} validation images.")

found = False
for img_path in val_images:
    results = model(img_path, conf=0.2, verbose=False)
    if len(results[0]) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy[0]) > 0:
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        confs = results[0].keypoints.conf[0].cpu().numpy()
        
        # Check if they are valid keypoints (non-zero)
        if not np.any(kpts == 0.0):
            print(f"\nSUCCESS: Detected keypoints in {img_path}:")
            for i, (kp, conf) in enumerate(zip(kpts, confs)):
                print(f"  Point {i}: x={kp[0]:.2f}, y={kp[1]:.2f}, conf={conf:.2f}")
            found = True
            break

if not found:
    print("Could not find any image with 4 valid keypoints detected.")
