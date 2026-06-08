import glob
import os
import numpy as np

# Find label files
label_paths = glob.glob('/home/luongduy/AeroScript_Vision/datasets/pen_pose/ComputerVision.v1i.yolov8/train/labels/*.txt')
print(f"Found {len(label_paths)} train labels.")

print(f"{'Label File':<50} | {'GT Tip-Tail (norm)':<18} | {'GT Cap-Cap (norm)':<18} | {'Ratio':<10}")
print("-" * 105)

count = 0
for path in label_paths[:15]:
    with open(path, 'r') as f:
        lines = f.readlines()
    if not lines:
        continue
    
    parts = lines[0].strip().split()
    # YOLO pose format: class x_c y_c w h kp0_x kp0_y kp0_v kp1_x kp1_y kp1_v ...
    if len(parts) < 17: # needs at least 4 keypoints
        continue
    
    try:
        kp0 = np.array([float(parts[5]), float(parts[6])])
        kp1 = np.array([float(parts[8]), float(parts[9])])
        kp2 = np.array([float(parts[11]), float(parts[12])])
        kp3 = np.array([float(parts[14]), float(parts[15])])
        
        # Scale to 640x480 to compare with predicted pixels
        kp0_px = kp0 * [640.0, 480.0]
        kp1_px = kp1 * [640.0, 480.0]
        kp2_px = kp2 * [640.0, 480.0]
        kp3_px = kp3 * [640.0, 480.0]
        
        d_tip_tail = np.linalg.norm(kp0_px - kp1_px)
        d_cap_cap = np.linalg.norm(kp2_px - kp3_px)
        ratio = d_tip_tail / (d_cap_cap + 1e-6)
        
        file_name = os.path.basename(path)
        print(f"{file_name[:50]:<50} | {d_tip_tail:<18.2f} | {d_cap_cap:<18.2f} | {ratio:<10.2f}")
        count += 1
    except Exception as e:
        continue

print(f"\nAnalyzed {count} ground truth label files.")
