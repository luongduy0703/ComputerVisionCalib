import cv2
import numpy as np
from ultralytics import YOLO
import glob

model_path = '/home/luongduy/AeroScript_Vision/runs/pose/Aero_Models/pen_pose_v1-9/weights/best.pt'
model = YOLO(model_path)

val_images = glob.glob('/home/luongduy/AeroScript_Vision/datasets/pen_pose/merged_pen_pose/val/images/*.jpg')
print(f"Analyzing {len(val_images)} validation images...\n")

print(f"{'Image':<50} | {'Tip-Tail (px)':<15} | {'Cap-Cap (px)':<15} | {'Ratio':<10}")
print("-" * 100)

count = 0
for img_path in val_images[:15]:
    results = model(img_path, conf=0.3, verbose=False)
    if len(results[0]) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy[0]) > 0:
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        
        # Check if they are valid keypoints (non-zero)
        if not np.any(kpts == 0.0):
            d_tip_tail = np.linalg.norm(kpts[0] - kpts[1])
            d_cap_cap = np.linalg.norm(kpts[2] - kpts[3])
            ratio = d_tip_tail / (d_cap_cap + 1e-6)
            img_name = img_path.split('/')[-1]
            print(f"{img_name[:50]:<50} | {d_tip_tail:<15.2f} | {d_cap_cap:<15.2f} | {ratio:<10.2f}")
            count += 1

print(f"\nSuccessfully analyzed {count} images.")
