import glob
import os
import numpy as np

# Find label files
label_paths = glob.glob('/home/luongduy/AeroScript_Vision/datasets/pen_pose/ComputerVision.v1i.yolov8/train/labels/*.txt')
print(f"Found {len(label_paths)} train labels.")

swapped_count = 0
inconsistent_count = 0
good_count = 0

for path in label_paths:
    with open(path, 'r') as f:
        lines = f.readlines()
    if not lines:
        continue
    
    parts = lines[0].strip().split()
    if len(parts) < 17:
        continue
    
    try:
        # Get the 4 keypoints (x, y normalized)
        kp0 = np.array([float(parts[5]), float(parts[6])])
        kp1 = np.array([float(parts[8]), float(parts[9])])
        kp2 = np.array([float(parts[11]), float(parts[12])])
        kp3 = np.array([float(parts[14]), float(parts[15])])
        
        # Calculate pixel distances (assuming 640x480)
        d01 = np.linalg.norm((kp0 - kp1) * [640, 480])
        d23 = np.linalg.norm((kp2 - kp3) * [640, 480])
        
        # Typically, Tip-to-Tail (d01) should be much larger than Cap-to-Cap (d23)
        # If d23 > d01, they are likely swapped!
        if d23 > d01:
            swapped_count += 1
        elif d01 / (d23 + 1e-6) < 1.5:
            inconsistent_count += 1
        else:
            good_count += 1
            
    except Exception as e:
        continue

print(f"Analysis of training labels:")
print(f"  Total analyzed files: {good_count + swapped_count + inconsistent_count}")
print(f"  Good files (Tip-Tail > 1.5 * Cap-Cap): {good_count}")
print(f"  Swapped files (Cap-Cap > Tip-Tail): {swapped_count}")
print(f"  Inconsistent files (1.0 <= Tip-Tail / Cap-Cap <= 1.5): {inconsistent_count}")
